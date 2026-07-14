"""Bot de SEÑALES de tenis para Kalshi (no coloca ordenes; solo recomienda).

Analogo al bot de temperatura, pero para tenis: lee los mercados de Kalshi,
estima cada partido con el MODELO PROPIO (Elo v2 calibrado + ML, promediados) y
compara esa probabilidad INDEPENDIENTE contra el precio neto de Kalshi. Donde el
modelo ve ventaja suficiente, emite una señal con su tamaño (¼ Kelly).

Honesto: el modelo NO le gana al mercado sharp (probado). Estas señales solo
valen si los 250 en Kalshi son lo bastante finos como para estar mal preciados —
el tracker de acierto lo dira. Usar como paper hasta comprobar edge real.

  python ml/kalshi_signals.py
"""
import os
import sys
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import config                                              # noqa: E402
from ml import kalshi                                      # noqa: E402
from ml.tennis_models import predict_pair                 # noqa: E402

OUT = ROOT / "ml" / "outputs" / "kalshi_signals.csv"
KELLY_FRAC = 0.25
MAX_ODDS = config.MAX_ODDS_SIM
COLS = ["ts", "tour", "tournament", "surface", "commence_time", "player_a", "player_b",
        "pick", "p_model", "kalshi_prob", "odds", "edge_pct", "kelly_pct", "stake",
        "volume", "event_ticker"]

if sys.platform == "win32":
    for _s in (sys.stdout, sys.stderr):
        try:
            _s.reconfigure(encoding="utf-8")
        except (AttributeError, ValueError):
            pass


def _model_prob(tour, a, b, surface):
    """Prob INDEPENDIENTE de que gane A: promedio de Elo v2 calibrado y ML.
    No usa el Blend (ese incorpora el mercado -> seria circular contra Kalshi)."""
    pe, pml = predict_pair(tour, a, b, surface, 3)
    vals = [x for x in (pe, pml) if x is not None]
    return sum(vals) / len(vals) if vals else None


def signals(min_edge=None, bankroll=None, min_volume=None):
    min_edge = config.KALSHI_SIGNAL_MIN_EDGE if min_edge is None else min_edge
    bankroll = config.KALSHI_BANKROLL if bankroll is None else bankroll
    ms = kalshi.matches(min_volume=min_volume)
    now = datetime.now(timezone.utc).isoformat()
    out = []
    for m in ms:
        a, b = m["player_a"], m["player_b"]
        pa = _model_prob(m["tour"], a, b, m["surface"])
        if pa is None:
            continue                                    # jugador desconocido para el modelo
        oa, ob = m["odds_a"], m["odds_b"]               # ya netas (comision Kalshi descontada)
        # ventaja de cada lado; elegir el mejor por Kelly
        opts = []
        for side, name, p_side, o in (("A", a, pa, oa), ("B", b, 1 - pa, ob)):
            if o and o > 1:
                edge = p_side * o - 1
                f = (p_side * o - 1) / (o - 1)
                opts.append((side, name, p_side, o, edge, f))
        if not opts:
            continue
        side, name, p_side, o, edge, f = max(opts, key=lambda x: x[4])
        if edge < min_edge or o > MAX_ODDS or f <= 0:
            continue
        stake = round(bankroll * min(f * KELLY_FRAC, 0.25), 2)
        out.append({
            "ts": now, "tour": m["tour"], "tournament": m["tournament"],
            "surface": m["surface"], "commence_time": m["commence_time"],
            "player_a": a, "player_b": b, "pick": name,
            "p_model": round(p_side, 3), "kalshi_prob": round(1 / o, 3),
            "odds": round(o, 2), "edge_pct": round(edge * 100, 1),
            "kelly_pct": round(f * 100, 1), "stake": stake,
            "volume": m["volume"], "event_ticker": m["event_ticker"],
        })
    out.sort(key=lambda x: x["edge_pct"], reverse=True)
    return out


def save(sigs):
    df = pd.DataFrame(sigs, columns=COLS)
    OUT.parent.mkdir(parents=True, exist_ok=True)
    with open(OUT, "w", encoding="utf-8", newline="") as f:
        df.to_csv(f, index=False)
        f.flush()
        os.fsync(f.fileno())


def load():
    """Señales de la última corrida. El bot solo mira mercados ABIERTOS de Kalshi
    (status=open), que son tradeables incluso en vivo, así que se muestran todas;
    solo se descartan las claramente terminadas (>4 h desde el saque)."""
    if not OUT.exists():
        return []
    try:
        d = pd.read_csv(OUT)
    except Exception:
        return []
    if d.empty:
        return []
    # el bot solo mira mercados ABIERTOS de Kalshi (status=open) y un partido
    # decidido tendria cuota extrema (filtrada por el tope), asi que todas las
    # señales son mercados tradeables. Se muestran todas, ordenadas por ventaja.
    d = d.sort_values("edge_pct", ascending=False)
    return d.where(pd.notna(d), None).to_dict("records")


def main():
    sigs = signals()
    save(sigs)
    print(f"Señales Kalshi (tenis): {len(sigs)} con ventaja >= {config.KALSHI_SIGNAL_MIN_EDGE*100:.0f}% "
          f"(banca S/{config.KALSHI_BANKROLL:.0f}, ¼ Kelly)")
    for s in sigs[:15]:
        print(f"  +{s['edge_pct']:5.1f}%  {s['pick'][:20]:20s} @ {s['odds']:.2f}  "
              f"(modelo {s['p_model']*100:.0f}% vs Kalshi {s['kalshi_prob']*100:.0f}%)  "
              f"S/{s['stake']:.2f}  vol ${s['volume']:.0f}  [{s['tournament'][:18]}]")
    print("\nHonesto: modelo vs Kalshi. Usar como paper hasta que el tracker confirme edge real.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
