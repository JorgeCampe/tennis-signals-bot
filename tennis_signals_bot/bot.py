#!/usr/bin/env python3
"""Bot de senales de tenis (Kalshi) — app INDEPENDIENTE con banca en paper.

Es un bot aparte del dashboard Flask (localhost). No coloca ordenes reales: lleva
una banca simulada que arranca en $250 (como el bot de temperatura), apuesta las
senales del modelo contra el precio NETO de Kalshi (comision descontada), y
liquida cada apuesta con los resultados reales de ESPN. Guarda su propio historial
y genera un dashboard HTML autonomo (dashboard.html) que se abre en el navegador.

Reusa el "cerebro" del proyecto NBA (Elo v2 calibrado + ML + Kalshi + resultados
ESPN) por sys.path, pero es totalmente separado: su propia carpeta, sus propios
datos y su propia interfaz.

  python bot.py           # corre la simulacion y regenera el dashboard
  python bot.py --open    # ademas abre dashboard.html en el navegador

Honesto: el modelo NO le gana al mercado sharp. Estas senales solo valen si los
250/Challengers de Kalshi estan mal preciados. La banca en paper es el juez: si
no crece con el tiempo, no hay edge real. Usar como paper, no como consejo.
"""
import os
import sys
import json
import random
import argparse
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent                      # carpeta NBA: reusar los modelos
sys.path.insert(0, str(ROOT))

import config                                           # noqa: E402
from ml import kalshi                                   # noqa: E402
from ml import kalshi_signals                           # noqa: E402
from ml import tennis_results                           # noqa: E402

# ----- parametros del bot (editables) --------------------------------------
START = 250.0            # banca inicial (como el bot de temperatura)
MIN_EDGE = 0.05          # ventaja minima del modelo vs Kalshi para apostar
KELLY_FRAC = 0.25        # 1/4 de Kelly
MAX_STAKE_FRAC = 0.10    # tope de 10% de la banca por apuesta (seguridad)
MAX_EXPOSURE_FRAC = 0.60  # tope de banca total en juego a la vez
MIN_STAKE = 1.0          # apuesta minima en $
MAX_ODDS = config.MAX_ODDS_SIM   # tope de cuota (guarda favorito-longshot)
# ---------------------------------------------------------------------------

DATA = HERE / "data"
POS = DATA / "positions.csv"
EQ = DATA / "equity.csv"
SIG = DATA / "signals.csv"
DASH = HERE / "dashboard.html"

POS_COLS = ["id", "opened_ts", "tour", "tournament", "surface", "commence_time",
            "player_a", "player_b", "pick", "model_prob", "kalshi_prob", "odds",
            "edge_pct", "kelly_pct", "stake", "status", "settled_ts", "pnl"]

if sys.platform == "win32":
    for _s in (sys.stdout, sys.stderr):
        try:
            _s.reconfigure(encoding="utf-8")
        except (AttributeError, ValueError):
            pass


# ---------- utilidades ------------------------------------------------------
def _safe_write_csv(df, path):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8", newline="") as f:
        df.to_csv(f, index=False)
        f.flush()
        os.fsync(f.fileno())


def _load_positions():
    if POS.exists():
        try:
            return pd.read_csv(POS).to_dict("records")
        except Exception:
            pass
    return []


def _lastname(x):
    t = kalshi.norm_full(x).split()
    return t[-1] if t else ""


def _match_key(pa, pb):
    return "|".join(sorted([_lastname(pa), _lastname(pb)]))


def _settle_pick(pick, pa, pb, commence_time, res):
    """'won'/'lost'/None comparando con ESPN. Cruza por apellidos en +-4 dias."""
    if res is None or getattr(res, "empty", True):
        return None
    d0 = pd.to_datetime(commence_time, utc=True, errors="coerce")
    want = {_lastname(pa), _lastname(pb)}
    for _, row in res.iterrows():
        wl = str(row.get("w", "")).split()[-1] if str(row.get("w", "")) else ""
        ll = str(row.get("l", "")).split()[-1] if str(row.get("l", "")) else ""
        if {wl, ll} != want:
            continue
        if pd.notna(d0):
            dd = pd.to_datetime(row.get("date"), utc=True, errors="coerce")
            if pd.notna(dd) and abs((dd - d0).days) > 4:
                continue
        return "won" if _lastname(pick) == wl else "lost"
    return None


def _montecarlo(open_pos, equity, n=5000):
    """Distribucion de la banca si las apuestas abiertas resuelven segun el MODELO.
    Es una PROYECCION (asume que la prob del modelo es correcta), no resultado real."""
    if not open_pos:
        return None
    bets = []
    for p in open_pos:
        pm = float(p["model_prob"])
        o = float(p["odds"])
        st = float(p["stake"])
        bets.append((pm, st * (o - 1.0), -st))
    ends = []
    for _ in range(n):
        tot = 0.0
        for pm, win_amt, lose_amt in bets:
            tot += win_amt if random.random() < pm else lose_amt
        ends.append(equity + tot)
    ends.sort()

    def pct(q):
        i = min(len(ends) - 1, max(0, int(q * len(ends))))
        return round(ends[i], 2)

    return {
        "expected": round(sum(ends) / len(ends), 2),
        "p5": pct(0.05), "p50": pct(0.50), "p95": pct(0.95),
        "prob_profit": round(100.0 * sum(1 for e in ends if e > equity) / len(ends), 1),
        "n_bets": len(bets),
        "stake_total": round(sum(-b[2] for b in bets), 2),
    }


def _gross_and_fee(net_odds):
    """De la cuota NETA (con fee) saca la cuota BRUTA (mercado) y el fee como %
    del desembolso. Invierte: costo = 1/neta = precio + FEE*precio*(1-precio)."""
    try:
        o = float(net_odds)
    except (TypeError, ValueError):
        return None, None
    if o <= 1:
        return None, None
    fr = config.KALSHI_FEE_RATE
    cost = 1.0 / o
    disc = (1 + fr) ** 2 - 4 * fr * cost            # fr*p^2 - (1+fr)*p + cost = 0
    if fr <= 0 or disc < 0:
        return round(o, 2), 0.0
    p = ((1 + fr) - disc ** 0.5) / (2 * fr)         # raiz menor = el precio
    if not (0 < p < 1):
        return round(o, 2), 0.0
    return round(1.0 / p, 2), round(fr * p * (1 - p) * 100.0, 1)   # fee segun Kalshi (max 1.75%)


# ---------- motor -----------------------------------------------------------
def run(open_browser=False):
    now = datetime.now(timezone.utc)
    now_iso = now.isoformat()
    positions = _load_positions()

    # 1) resultados frescos de ESPN y liquidar apuestas abiertas
    tennis_results.refresh_if_stale(minutes=15)
    res = tennis_results.load_results()
    for p in positions:
        if str(p.get("status")) != "open":
            continue
        r = _settle_pick(p["pick"], p["player_a"], p["player_b"], p.get("commence_time"), res)
        if r == "won":
            p["status"] = "won"
            p["pnl"] = round(float(p["stake"]) * (float(p["odds"]) - 1), 2)
            p["settled_ts"] = now_iso
        elif r == "lost":
            p["status"] = "lost"
            p["pnl"] = -round(float(p["stake"]), 2)
            p["settled_ts"] = now_iso
        else:
            d0 = pd.to_datetime(p.get("commence_time"), utc=True, errors="coerce")
            if pd.notna(d0) and (now - d0).days >= 5:      # nunca aparecio -> anular
                p["status"] = "void"
                p["pnl"] = 0.0
                p["settled_ts"] = now_iso

    # equity = 250 + P&L realizado ; cash = equity - lo que esta en juego
    settled = [p for p in positions if str(p.get("status")) in ("won", "lost", "void")]
    realized = sum(float(p.get("pnl") or 0) for p in settled)
    equity = round(START + realized, 2)
    open_pos = [p for p in positions if str(p.get("status")) == "open"]
    at_risk = round(sum(float(p["stake"]) for p in open_pos), 2)
    cash = round(equity - at_risk, 2)

    # 2) senales actuales (modelo vs Kalshi), dimensionadas a la banca de hoy
    try:
        sigs = kalshi_signals.signals(min_edge=MIN_EDGE, bankroll=equity)
    except Exception as e:
        print(f"  (no se pudieron leer senales de Kalshi: {e})")
        sigs = []

    # 3) colocar nuevas apuestas — solo partidos GENUINAMENTE indecisos.
    # Se saltan los que ESPN ya tiene como terminados: apostar un mercado ya
    # resuelto (Kalshi a veces tarda en cerrar) inflaria la banca con hindsight.
    decided = set()
    if res is not None and not res.empty:
        for _, row in res.iterrows():
            wl = str(row.get("w", "")).split()[-1] if str(row.get("w", "")) else ""
            ll = str(row.get("l", "")).split()[-1] if str(row.get("l", "")) else ""
            if wl and ll:
                decided.add("|".join(sorted([wl, ll])))

    open_keys = {_match_key(p["player_a"], p["player_b"]) for p in open_pos}
    nid = max([int(p.get("id", 0)) for p in positions], default=0)
    cap = round(equity * MAX_EXPOSURE_FRAC, 2)     # tope de exposicion total
    exposure = at_risk
    avail = cash
    placed = 0
    for s in sigs:
        k = _match_key(s["player_a"], s["player_b"])
        if k in open_keys or k in decided:
            continue
        f = float(s["kelly_pct"]) / 100.0
        stake = round(equity * min(f * KELLY_FRAC, MAX_STAKE_FRAC), 2)
        if stake < MIN_STAKE:
            continue
        room = min(avail, round(cap - exposure, 2))    # respeta cash y tope total
        if stake > room:
            stake = round(room, 2)
        if stake < MIN_STAKE:
            break                              # sin cash / sin cupo de exposicion
        nid += 1
        positions.append({
            "id": nid, "opened_ts": now_iso, "tour": s["tour"], "tournament": s["tournament"],
            "surface": s["surface"], "commence_time": s["commence_time"],
            "player_a": s["player_a"], "player_b": s["player_b"], "pick": s["pick"],
            "model_prob": s["p_model"], "kalshi_prob": s["kalshi_prob"], "odds": s["odds"],
            "edge_pct": s["edge_pct"], "kelly_pct": s["kelly_pct"], "stake": stake,
            "status": "open", "settled_ts": "", "pnl": "",
        })
        open_keys.add(k)
        exposure = round(exposure + stake, 2)
        avail = round(avail - stake, 2)
        placed += 1

    # recomputar tras colocar
    open_pos = [p for p in positions if str(p.get("status")) == "open"]
    at_risk = round(sum(float(p["stake"]) for p in open_pos), 2)
    cash = round(equity - at_risk, 2)
    wins = sum(1 for p in settled if p["status"] == "won")
    losses = sum(1 for p in settled if p["status"] == "lost")

    # 4) persistir estado
    _safe_write_csv(pd.DataFrame(positions, columns=POS_COLS), POS)
    _safe_write_csv(pd.DataFrame(sigs, columns=kalshi_signals.COLS), SIG)
    eq_hist = []
    if EQ.exists():
        try:
            eq_hist = pd.read_csv(EQ).to_dict("records")
        except Exception:
            eq_hist = []
    eq_hist.append({"ts": now_iso, "equity": equity, "cash": cash, "at_risk": at_risk,
                    "realized_cum": round(realized, 2), "wins": wins, "losses": losses,
                    "n_open": len(open_pos)})
    _safe_write_csv(pd.DataFrame(eq_hist), EQ)

    # 5) proyeccion Monte Carlo + dashboard
    mc = _montecarlo(open_pos, equity)
    _write_dashboard(equity, cash, at_risk, realized, wins, losses,
                     open_pos, settled, sigs, eq_hist, mc, now_iso, placed)

    roi = (equity / START - 1) * 100
    print(f"Banca ${equity:.2f} (inicio ${START:.0f}, ROI {roi:+.1f}%) | "
          f"cash ${cash:.2f} | en juego ${at_risk:.2f} ({len(open_pos)} abiertas) | "
          f"record {wins}-{losses} | {placed} nuevas apuestas | {len(sigs)} senales")
    print(f"Dashboard: {DASH}")
    if open_browser:
        import webbrowser
        webbrowser.open(DASH.as_uri())
    return 0


# ---------- dashboard HTML autonomo ----------------------------------------
def _write_dashboard(equity, cash, at_risk, realized, wins, losses,
                     open_pos, settled, sigs, eq_hist, mc, now_iso, placed):
    payload = {
        "start": START, "equity": equity, "cash": cash, "at_risk": at_risk,
        "realized": round(realized, 2), "wins": wins, "losses": losses,
        "roi": round((equity / START - 1) * 100, 1),
        "min_edge": round(MIN_EDGE * 100), "kelly_frac": KELLY_FRAC,
        "updated": now_iso, "placed": placed,
        "equity_curve": [{"ts": r["ts"], "equity": r["equity"]} for r in eq_hist],
        "open": [_pos_view(p) for p in sorted(open_pos, key=lambda x: -float(x.get("edge_pct", 0)))],
        "closed": [_pos_view(p) for p in sorted(settled, key=lambda x: str(x.get("settled_ts", "")), reverse=True)],
        "signals": [_sig_view(s) for s in sigs],
        "mc": mc,
    }
    html = _DASHBOARD_TEMPLATE.replace("/*DATA*/", json.dumps(payload, ensure_ascii=False))
    with open(DASH, "w", encoding="utf-8", newline="") as f:
        f.write(html)
        f.flush()
        os.fsync(f.fileno())


def _pos_view(p):
    rival = p["player_b"] if p["pick"] == p["player_a"] else p["player_a"]
    return {
        "tour": p.get("tour"), "tournament": p.get("tournament"), "surface": p.get("surface"),
        "pick": p.get("pick"), "rival": rival, "odds": p.get("odds"),
        "gross": _gross_and_fee(p.get("odds"))[0], "fee_pct": _gross_and_fee(p.get("odds"))[1],
        "model": round(float(p.get("model_prob", 0)) * 100, 1),
        "edge": p.get("edge_pct"), "stake": p.get("stake"),
        "status": p.get("status"), "pnl": p.get("pnl"),
        "opened": p.get("opened_ts"), "settled": p.get("settled_ts"), "commence": p.get("commence_time"),
    }


def _sig_view(s):
    g, fp = _gross_and_fee(s["odds"])
    return {
        "tour": s["tour"], "tournament": s["tournament"], "surface": s["surface"],
        "pick": s["pick"], "odds": s["odds"], "gross": g, "fee_pct": fp,
        "model": round(s["p_model"] * 100, 1), "kalshi": round(s["kalshi_prob"] * 100, 1),
        "edge": s["edge_pct"], "stake": s["stake"], "volume": s.get("volume"),
        "commence": s.get("commence_time"),
        "rival": s["player_b"] if s["pick"] == s["player_a"] else s["player_a"],
    }


_DASHBOARD_TEMPLATE = r"""<!DOCTYPE html>
<html lang="es">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Bot de Senales — Tenis (Kalshi)</title>
<script src="https://cdnjs.cloudflare.com/ajax/libs/Chart.js/4.4.1/chart.umd.min.js"></script>
<style>
:root{
  --bg:#0b0f16; --card:#141b26; --card2:#1b2432; --line:#243040; --txt:#e6edf5;
  --dim:#8b97a8; --up:#22c55e; --down:#ef4444; --accent:#37d3a4; --accent2:#3b82f6;
}
*{box-sizing:border-box}
body{margin:0;background:var(--bg);color:var(--txt);
  font-family:-apple-system,BlinkMacSystemFont,"Segoe UI",Roboto,Helvetica,Arial,sans-serif;}
.wrap{max-width:1120px;margin:0 auto;padding:22px 18px 60px;}
.top{display:flex;align-items:center;justify-content:space-between;flex-wrap:wrap;gap:10px;margin-bottom:18px;}
.brand{display:flex;align-items:center;gap:12px;font-weight:700;font-size:19px;}
.brand .dot{width:10px;height:10px;border-radius:50%;background:var(--accent);box-shadow:0 0 10px var(--accent);}
.fresh{color:var(--dim);font-size:13px;}
.hero{display:grid;grid-template-columns:1.15fr 1fr;gap:16px;margin-bottom:16px;}
@media(max-width:820px){.hero{grid-template-columns:1fr}}
.card{background:var(--card);border:1px solid var(--line);border-radius:16px;padding:18px;}
.bankroll{font-size:44px;font-weight:800;letter-spacing:-1px;line-height:1;}
.sub{color:var(--dim);font-size:13px;margin-top:6px;}
.pill{display:inline-block;padding:3px 10px;border-radius:999px;font-weight:700;font-size:13px;}
.up{color:var(--up)} .down{color:var(--down)}
.pill.up{background:rgba(34,197,94,.13)} .pill.down{background:rgba(239,68,68,.13)}
.kpis{display:grid;grid-template-columns:repeat(4,1fr);gap:12px;margin:4px 0 18px;}
@media(max-width:820px){.kpis{grid-template-columns:repeat(2,1fr)}}
.kpi{background:var(--card);border:1px solid var(--line);border-radius:14px;padding:14px;}
.kpi .l{color:var(--dim);font-size:12px;text-transform:uppercase;letter-spacing:.4px;}
.kpi .v{font-size:23px;font-weight:800;margin-top:6px;}
h2{font-size:15px;margin:26px 0 10px;color:var(--txt);}
h2 span{color:var(--dim);font-weight:500;font-size:13px;}
table{width:100%;border-collapse:collapse;font-size:13.5px;}
th,td{padding:9px 10px;text-align:left;border-bottom:1px solid var(--line);white-space:nowrap;}
th{color:var(--dim);font-weight:600;font-size:12px;text-transform:uppercase;letter-spacing:.3px;}
tr:last-child td{border-bottom:none}
.tag{font-size:11px;padding:2px 7px;border-radius:6px;background:var(--card2);color:var(--dim);}
.tag.atp{color:#7dd3fc}.tag.wta{color:#f0abfc}
.badge{font-weight:700;padding:2px 8px;border-radius:6px;font-size:12px;}
.badge.win{background:rgba(34,197,94,.15);color:var(--up)}
.badge.lose{background:rgba(239,68,68,.15);color:var(--down)}
.badge.open{background:rgba(59,130,246,.15);color:#93c5fd}
.badge.void{background:rgba(139,151,168,.15);color:var(--dim)}
.mc{display:grid;grid-template-columns:repeat(4,1fr);gap:12px;margin-top:6px}
@media(max-width:820px){.mc{grid-template-columns:repeat(2,1fr)}}
.mc .v{font-size:20px;font-weight:800;margin-top:4px}
.note{color:var(--dim);font-size:12.5px;line-height:1.5;margin-top:10px;
  border-left:3px solid var(--line);padding-left:10px;}
.right{text-align:right}
.empty{color:var(--dim);padding:16px 4px;font-size:14px;}
.scroll{overflow-x:auto;border:1px solid var(--line);border-radius:14px;}
.scroll table th,.scroll table td{border-bottom:1px solid var(--line)}
canvas{max-height:230px}
</style>
</head>
<body>
<div class="wrap">
  <div class="top">
    <div class="brand"><span class="dot"></span>Bot de Senales · Tenis <span style="color:var(--dim);font-weight:500">Kalshi</span></div>
    <div class="fresh" id="fresh"></div>
  </div>

  <div class="hero">
    <div class="card">
      <div class="sub">Banca simulada (paper)</div>
      <div class="bankroll" id="bankroll">—</div>
      <div class="sub" id="roiline"></div>
      <canvas id="curve" style="margin-top:14px"></canvas>
    </div>
    <div class="card">
      <div class="sub" style="margin-bottom:6px">Proyeccion Monte Carlo · apuestas abiertas</div>
      <div id="mcbox"></div>
      <div class="note">Simula 5.000 escenarios asumiendo que la probabilidad del <b>modelo</b> es correcta.
        Es una proyeccion, no resultado real. La banca de arriba solo se mueve con partidos ya liquidados por ESPN.</div>
    </div>
  </div>

  <div class="kpis" id="kpis"></div>

  <div class="sub" style="margin:2px 0 -8px">Cuota = precio de mercado (Kalshi) · Fee = comisión · Neta = cuota después del fee (a la que apuesta el bot).</div>

  <h2>Senales de hoy <span id="sigcount"></span></h2>
  <div class="scroll"><table id="sigtab">
    <thead><tr><th>Tour</th><th>Torneo</th><th>Fecha (Perú)</th><th>Pick</th><th>Rival</th><th class="right">Cuota</th><th class="right">Fee</th><th class="right">Neta</th>
      <th class="right">Modelo</th><th class="right">Kalshi</th><th class="right">Ventaja</th><th class="right">Stake</th></tr></thead>
    <tbody></tbody></table></div>

  <h2>Posiciones abiertas <span id="opencount"></span></h2>
  <div class="scroll"><table id="opentab">
    <thead><tr><th>Tour</th><th>Torneo</th><th>Fecha (Perú)</th><th>Pick</th><th>Rival</th><th class="right">Cuota</th><th class="right">Fee</th><th class="right">Neta</th>
      <th class="right">Modelo</th><th class="right">Ventaja</th><th class="right">Stake</th></tr></thead>
    <tbody></tbody></table></div>

  <h2>Historial liquidado <span id="histcount"></span></h2>
  <div class="scroll"><table id="histtab">
    <thead><tr><th>Resultado</th><th>Torneo</th><th>Pick</th><th class="right">Cuota</th>
      <th class="right">Stake</th><th class="right">P&L</th></tr></thead>
    <tbody></tbody></table></div>
</div>

<script>
const D = /*DATA*/;
const money = v => (v<0?'-$':'$') + Math.abs(v).toFixed(2);
const peru = iso => { if(!iso) return '\u2014'; const d=new Date(iso); return isNaN(d)?'\u2014':new Intl.DateTimeFormat('es-PE',{timeZone:'America/Lima',weekday:'short',day:'2-digit',month:'short',hour:'2-digit',minute:'2-digit',hour12:false}).format(d); };
const el = id => document.getElementById(id);

// freshness
(function(){
  const t = new Date(D.updated), s = Math.max(0,(Date.now()-t)/60000);
  const ago = s<1?'recien':(s<60?Math.round(s)+' min':Math.round(s/60)+' h');
  el('fresh').textContent = 'Actualizado hace ' + ago + '  ·  edge min ' + D.min_edge + '%  ·  ¼ Kelly';
})();

// bankroll + roi
el('bankroll').textContent = money(D.equity);
const up = D.roi>=0;
el('roiline').innerHTML = '<span class="pill '+(up?'up':'down')+'">'+(up?'▲ ':'▼ ')+D.roi+'%</span>'
  + ' &nbsp;vs inicio $'+D.start.toFixed(0)+' &nbsp;·&nbsp; realizado '
  + '<span class="'+(D.realized>=0?'up':'down')+'">'+money(D.realized)+'</span>';

// kpis
const total = D.wins + D.losses;
const wr = total ? (100*D.wins/total).toFixed(0)+'%' : '—';
const kpis = [
  ['En juego', money(D.at_risk), D.open.length+' abiertas'],
  ['Cash libre', money(D.cash), ''],
  ['Record', D.wins+'–'+D.losses, 'aciertos '+wr],
  ['Senales hoy', String(D.signals.length), D.placed+' apostadas'],
];
el('kpis').innerHTML = kpis.map(k=>`<div class="kpi"><div class="l">${k[0]}</div><div class="v">${k[1]}</div><div class="sub">${k[2]}</div></div>`).join('');

// montecarlo
if(D.mc){
  const m = D.mc, pu = m.prob_profit>=50;
  el('mcbox').innerHTML = `<div class="mc">
    <div><div class="l" style="color:var(--dim);font-size:12px">Esperada</div><div class="v">${money(m.expected)}</div></div>
    <div><div class="l" style="color:var(--dim);font-size:12px">Prob. ganar</div><div class="v ${pu?'up':'down'}">${m.prob_profit}%</div></div>
    <div><div class="l" style="color:var(--dim);font-size:12px">Malo (P5)</div><div class="v down">${money(m.p5)}</div></div>
    <div><div class="l" style="color:var(--dim);font-size:12px">Bueno (P95)</div><div class="v up">${money(m.p95)}</div></div>
  </div><div class="sub" style="margin-top:8px">${m.n_bets} apuestas abiertas · $${m.stake_total.toFixed(2)} en juego</div>`;
} else {
  el('mcbox').innerHTML = '<div class="empty">Sin apuestas abiertas para proyectar.</div>';
}

// equity curve
(function(){
  const c = D.equity_curve;
  const labels = c.map(p=>{const d=new Date(p.ts);return (d.getMonth()+1)+'/'+d.getDate()+' '+String(d.getHours()).padStart(2,'0')+':'+String(d.getMinutes()).padStart(2,'0');});
  const data = c.map(p=>p.equity);
  if(data.length===1){labels.unshift('inicio');data.unshift(D.start);}
  const ctx = el('curve');
  new Chart(ctx,{type:'line',
    data:{labels,datasets:[{data,borderColor:'#37d3a4',backgroundColor:'rgba(55,211,164,.10)',
      fill:true,tension:.25,pointRadius:data.length>30?0:2,borderWidth:2}]},
    options:{plugins:{legend:{display:false}},
      scales:{x:{grid:{color:'#1b2432'},ticks:{color:'#8b97a8',maxTicksLimit:8}},
              y:{grid:{color:'#1b2432'},ticks:{color:'#8b97a8',callback:v=>'$'+v}}}}});
})();

// tables
function tourTag(t){return `<span class="tag ${(''+t).toLowerCase()}">${t}</span>`;}
function fill(id,rows,cols){
  const tb = el(id).querySelector('tbody');
  if(!rows.length){tb.innerHTML=`<tr><td colspan="${cols}" class="empty">Nada por ahora.</td></tr>`;return;}
  tb.innerHTML = rows.join('');
}
fill('sigtab', D.signals.map(s=>`<tr>
  <td>${tourTag(s.tour)}</td><td>${s.tournament}</td><td style="color:var(--dim);white-space:nowrap">${peru(s.commence)}</td><td><b>${s.pick}</b></td><td style="color:var(--dim)">${s.rival}</td>
  <td class="right">${(s.gross!=null?s.gross:s.odds).toFixed(2)}</td><td class="right" style="color:var(--dim)">${s.fee_pct!=null?s.fee_pct+'%':'\u2014'}</td><td class="right">${s.odds.toFixed(2)}</td><td class="right">${s.model}%</td><td class="right">${s.kalshi}%</td>
  <td class="right up">+${s.edge}%</td><td class="right">${money(s.stake)}</td></tr>`), 12);
el('sigcount').textContent = D.signals.length ? `(${D.signals.length})` : '';

fill('opentab', D.open.map(p=>`<tr>
  <td>${tourTag(p.tour)}</td><td>${p.tournament}</td><td style="color:var(--dim);white-space:nowrap">${peru(p.commence)}</td><td><b>${p.pick}</b></td><td style="color:var(--dim)">${p.rival}</td>
  <td class="right">${(p.gross!=null?+p.gross:+p.odds).toFixed(2)}</td><td class="right" style="color:var(--dim)">${p.fee_pct!=null?p.fee_pct+'%':'\u2014'}</td><td class="right">${(+p.odds).toFixed(2)}</td><td class="right">${p.model}%</td>
  <td class="right up">+${p.edge}%</td><td class="right">${money(+p.stake)}</td></tr>`), 11);
el('opencount').textContent = D.open.length ? `(${D.open.length})` : '';

fill('histtab', D.closed.map(p=>{
  const st = p.status, pnl = (p.pnl===''||p.pnl==null)?0:+p.pnl;
  const badge = st==='won'?'<span class="badge win">GANO</span>':st==='lost'?'<span class="badge lose">PERDIO</span>':'<span class="badge void">ANUL.</span>';
  return `<tr><td>${badge}</td><td>${p.tournament}</td><td><b>${p.pick}</b></td>
    <td class="right">${(+p.odds).toFixed(2)}</td><td class="right">${money(+p.stake)}</td>
    <td class="right ${pnl>=0?'up':'down'}">${money(pnl)}</td></tr>`;
}), 6);
el('histcount').textContent = D.closed.length ? `(${D.wins}–${D.losses})` : '';
</script>
</body>
</html>"""


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="Bot de senales de tenis (Kalshi) con banca en paper.")
    ap.add_argument("--open", action="store_true", help="abre dashboard.html al terminar")
    args = ap.parse_args()
    sys.exit(run(open_browser=args.open))
