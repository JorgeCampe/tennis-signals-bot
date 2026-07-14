"""Resultados rapidos de tenis desde el scoreboard publico de ESPN (sin API key).

tennis-data.co.uk (el historico con cuotas de cierre) publica con varios dias de
retraso, asi que la simulacion no podia liquidar un partido de hoy. ESPN expone
un scoreboard publico y gratuito que tiene los resultados el mismo dia.

Guarda ml/outputs/tennis_results.csv con: date, tour, tournament, winner, loser.
El historico sigue usandose para las CUOTAS DE CIERRE (CLV), que ESPN no da.
"""
import os
import sys
import time
import unicodedata
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pandas as pd
import requests

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "ml" / "outputs" / "tennis_results.csv"
BASE = "https://site.api.espn.com/apis/site/v2/sports/tennis"
TOURS = ["atp", "wta"]
DAYS_BACK = 3
COLS = ["date", "tour", "tournament", "winner", "loser", "w_games", "l_games"]

if sys.platform == "win32":
    for _s in (sys.stdout, sys.stderr):
        try:
            _s.reconfigure(encoding="utf-8")
        except (AttributeError, ValueError):
            pass


def norm_full(s):
    """'Felix Auger-Aliassime' -> 'felix auger aliassime' (para cruzar nombres completos)."""
    s = "".join(ch for ch in unicodedata.normalize("NFD", str(s)) if not unicodedata.combining(ch))
    s = s.lower().replace("'", "").replace(".", "").replace("-", " ")
    return " ".join(s.split())


def _games(comp):
    """Games totales de un jugador sumando los linescores (games por set)."""
    ls = comp.get("linescores") or []
    tot, ok = 0, False
    for st in ls:
        v = st.get("value")
        if v is None:
            continue
        try:
            tot += int(float(v))
            ok = True
        except (TypeError, ValueError):
            pass
    return tot if ok else None


def _scoreboard(tour, day):
    url = f"{BASE}/{tour}/scoreboard"
    r = requests.get(url, params={"dates": day},
                     headers={"User-Agent": "Mozilla/5.0"}, timeout=25)
    r.raise_for_status()
    return r.json()


def fetch_espn(days=DAYS_BACK):
    """Lista de partidos completados (dedupe) de los ultimos `days` dias."""
    rows, seen = [], set()
    today = datetime.now(timezone.utc).date()
    for tour in TOURS:
        for k in range(days):
            day = (today - timedelta(days=k)).strftime("%Y%m%d")
            try:
                data = _scoreboard(tour, day)
            except Exception as e:
                print(f"  ESPN {tour} {day}: {e}")
                continue
            for ev in data.get("events", []):
                tname = ev.get("name")
                for grp in ev.get("groupings", []):
                    for c in grp.get("competitions", []):
                        if not c.get("status", {}).get("type", {}).get("completed"):
                            continue
                        cs = c.get("competitors", [])
                        if len(cs) != 2:
                            continue
                        if any((x.get("athlete") or {}).get("displayName") is None for x in cs):
                            continue
                        wc = [x for x in cs if x.get("winner")]
                        lc = [x for x in cs if not x.get("winner")]
                        if len(wc) != 1 or len(lc) != 1:
                            continue
                        w = (wc[0].get("athlete") or {}).get("displayName")
                        l = (lc[0].get("athlete") or {}).get("displayName")
                        date = str(c.get("date", ""))[:10]
                        key = (date, norm_full(w), norm_full(l))
                        if not date or key in seen:
                            continue
                        seen.add(key)
                        rows.append({"date": date, "tour": tour.upper(), "tournament": tname,
                                     "winner": w, "loser": l,
                                     "w_games": _games(wc[0]), "l_games": _games(lc[0])})
    return rows


def save(rows):
    """Fusiona con lo que ya habia y guarda (dedupe por fecha+ganador+perdedor)."""
    nd = pd.DataFrame(rows, columns=COLS)
    if RESULTS.exists():
        try:
            old = pd.read_csv(RESULTS)
            nd = pd.concat([old, nd], ignore_index=True)
        except Exception:
            pass
    if nd.empty:
        return 0
    nd["_k"] = [f"{d}|{norm_full(w)}|{norm_full(l)}"
                for d, w, l in zip(nd["date"], nd["winner"], nd["loser"])]
    nd = nd.drop_duplicates(subset=["_k"], keep="last").drop(columns=["_k"])
    nd = nd.sort_values("date").reset_index(drop=True)
    RESULTS.parent.mkdir(parents=True, exist_ok=True)
    with open(RESULTS, "w", encoding="utf-8", newline="") as f:
        nd[COLS].to_csv(f, index=False)
        f.flush()
        os.fsync(f.fileno())
    return len(nd)


def load_results():
    """DataFrame con columnas normalizadas w/l, o None si no hay archivo."""
    if not RESULTS.exists():
        return None
    try:
        r = pd.read_csv(RESULTS, parse_dates=["date"])
    except Exception:
        return None
    if r.empty:
        return None
    r["w"] = r["winner"].map(norm_full)
    r["l"] = r["loser"].map(norm_full)
    return r


def refresh_if_stale(minutes=10, days=DAYS_BACK):
    """Refresca desde ESPN solo si el archivo tiene mas de `minutes` de antiguedad."""
    try:
        if RESULTS.exists() and (time.time() - RESULTS.stat().st_mtime) < minutes * 60:
            return False
        rows = fetch_espn(days)
        if rows:
            save(rows)
            return True
    except Exception:
        pass
    return False


def main():
    rows = fetch_espn()
    n = save(rows)
    print(f"Resultados ESPN: {len(rows)} partidos bajados | {n} en tennis_results.csv")
    return 0


if __name__ == "__main__":
    sys.exit(main())
