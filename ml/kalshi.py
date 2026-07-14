"""Kalshi (mercado de predicciones regulado, CFTC) como casa de line shopping.

Los datos de mercado son PUBLICOS: no hace falta auth ni firma RSA para leer
precios (eso solo se necesita para APOSTAR por API). Cubre ATP/WTA incluyendo
250 y Challengers — justo lo que The Odds API no trae.

Cada partido es un mercado binario en la serie KXATPMATCH / KXWTAMATCH. Cada
mercado ya trae los dos lados: yes_sub_title (jugador Yes) con yes_ask, y
no_sub_title (el rival) con no_ask (precios en dolares 0-1 = probabilidad).
Cuota decimal = 1 / precio. Kalshi cobra comision por operacion, asi que
devolvemos la cuota NETA: fee = tasa * p * (1-p), costo = p + fee, cuota = 1/costo.
"""
import json
import sys
import unicodedata
import urllib.request
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import config   # noqa: E402

BASE = "https://api.elections.kalshi.com/trade-api/v2"
SERIES = ["KXATPMATCH", "KXWTAMATCH"]      # ganador de partido ATP / WTA
MIN_VOLUME = float(__import__("os").getenv("KALSHI_MIN_VOLUME", "0"))

GRASS = ("wimbledon", "halle", "queen", "eastbourne", "newport", "mallorca",
         "hertogenbosch", "nottingham", "homburg", "stuttgart")
CLAY = ("bastad", "gstaad", "umag", "kitzbuhel", "hamburg", "bucharest", "cordoba",
        "buenos aires", "rio", "santiago", "houston", "marrakech", "estoril",
        "munich", "madrid", "rome", "roma", "monte", "barcelona", "roland", "french",
        "geneva", "lyon", "bogota", "iasi", "athens", "kitzb", "swiss open",
        "croatia open", "nordea", "bastad")


def norm_full(s):
    s = "".join(ch for ch in unicodedata.normalize("NFD", str(s)) if not unicodedata.combining(ch))
    return " ".join(s.lower().replace("'", "").replace(".", "").replace("-", " ").split())


def _get(url):
    req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
    with urllib.request.urlopen(req, timeout=20) as r:
        return json.loads(r.read().decode())


def net_odds(price):
    """price = precio en dolares (0-1). Cuota decimal NETA (comision descontada)."""
    try:
        p = float(price)
    except (TypeError, ValueError):
        return None
    if p <= 0 or p >= 1:
        return None
    fee = config.KALSHI_FEE_RATE * p * (1 - p)
    cost = p + fee
    return round(1.0 / cost, 3) if cost > 0 else None


def _tournament(rules):
    """Extrae 'ATP Bastad', 'Wimbledon', etc. de las reglas del mercado."""
    import re
    m = re.search(r"in the 20\d\d (.+?) (?:Singles|Men|Women|after|professional)", str(rules))
    if m:
        return m.group(1).strip()
    m = re.search(r"in the 20\d\d ([A-Za-z .'-]+?)(?: Round| Quarterfinal| Semifinal| Final)", str(rules))
    return m.group(1).strip() if m else ""


def _surface(tournament):
    t = str(tournament).lower()
    if any(w in t for w in GRASS):
        return "Grass"
    if any(w in t for w in CLAY):
        return "Clay"
    return "Hard"


def matches(min_volume=None):
    """Lista de partidos de Kalshi con cuota NETA por lado.
    [{player_a, player_b, odds_a, odds_b, volume, commence_time, tour,
      tournament, surface, event_ticker}]."""
    if not config.KALSHI_ENABLED:
        return []
    mv = MIN_VOLUME if min_volume is None else min_volume
    # 1) juntar todos los mercados por evento (partido). Cada mercado es "¿gana X?":
    #    yes_sub_title = X, yes_ask = precio de X. El rival esta en el OTRO mercado
    #    del mismo evento. Por eso agrupamos y tomamos un mercado por jugador.
    events = {}
    for series in SERIES:
        tour = "ATP" if "ATP" in series else "WTA"
        cursor = ""
        for _ in range(15):
            url = f"{BASE}/markets?series_ticker={series}&status=open&limit=200"
            if cursor:
                url += f"&cursor={cursor}"
            try:
                d = _get(url)
            except Exception as e:
                print(f"  Kalshi {series}: {e}")
                break
            for m in d.get("markets", []):
                ev = m.get("event_ticker")
                player = m.get("yes_sub_title")
                if not ev or not player:
                    continue
                e = events.setdefault(ev, {"tour": tour, "sides": {}, "meta": m})
                try:
                    vol = float(m.get("volume_fp", 0))
                except (TypeError, ValueError):
                    vol = 0.0
                # un mercado por jugador (si se repite, el de mas volumen)
                key = norm_full(player)
                prev = e["sides"].get(key)
                if prev is None or vol > prev["vol"]:
                    e["sides"][key] = {"name": player, "ask": m.get("yes_ask_dollars"),
                                       "vol": vol, "start": m.get("occurrence_datetime") or m.get("close_time"),
                                       "rules": m.get("rules_primary", "")}
            cursor = d.get("cursor") or ""
            if not cursor:
                break
    # 2) un partido = evento con exactamente 2 jugadores distintos
    out = []
    for ev, e in events.items():
        sides = list(e["sides"].values())
        if len(sides) != 2:
            continue
        a, b = sides
        oa, ob = net_odds(a["ask"]), net_odds(b["ask"])
        if not (oa and ob):
            continue
        vol = a["vol"] + b["vol"]
        if vol < mv:
            continue
        tourn = _tournament(a["rules"] or e["meta"].get("rules_primary", ""))
        out.append({
            "player_a": a["name"], "player_b": b["name"], "odds_a": oa, "odds_b": ob,
            "volume": round(vol, 2),
            "commence_time": a["start"] or b["start"],
            "tour": e["tour"], "tournament": tourn or f"{e['tour']} (Kalshi)",
            "surface": _surface(tourn), "event_ticker": ev,
        })
    return out


def by_pair():
    """{frozenset(nombres normalizados): match} para cruzar con otros feeds."""
    idx = {}
    for m in matches():
        idx[frozenset((norm_full(m["player_a"]), norm_full(m["player_b"])))] = m
    return idx


if __name__ == "__main__":
    ms = matches()
    print(f"Kalshi tenis: {len(ms)} partidos con cuota neta")
    for m in ms[:12]:
        print(f"  [{m['tour']}] {m['tournament'][:22]:22s} {m['player_a'][:16]:16s} {m['odds_a']:.2f} "
              f"vs {m['player_b'][:16]:16s} {m['odds_b']:.2f}  vol ${m['volume']:.0f}")
