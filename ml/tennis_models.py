"""Modelos de tenis: Elo v2 (mejorado) + modelo ML aparte, para comparar.

Elo v2 sobre el Elo original agrega (puntos 1-4 del plan):
  1. Margen de victoria: el update pesa mas si la victoria fue mas contundente
     (por sets ganados). Retiros/walkovers pesan menos o se excluyen.
  2. Prior por ranking: si un jugador tiene pocos partidos en el sistema, su Elo
     se mezcla con un Elo implicito por su ranking ATP/WTA (arregla el caso
     "wildcard desconocido" tipo Fery).
  3. Recencia: tras un parate largo (>120 dias) el rating revierte hacia la media
     (el Elo "olvida" mas rapido a quien no juega).
  4. Contexto: en best-of-5 se estira la confianza (el mejor jugador gana mas
     seguido); el descanso entra como feature del modelo ML.

Modelo ML (punto 7): regresion logistica calibrada sobre features as-of (sin
fuga): diferencia de Elo v2, Elo por superficie, prob del Elo, ranking, forma
reciente, descanso, best-of. Entrenado walk-forward (train < 2025, test 2025-26).

Uso:
  python ml/tennis_models.py --backtest   # compara Elo v2 vs ML vs mercado
  python ml/tennis_models.py --fit        # entrena y guarda modelo + estados
"""
import argparse
import json
import math
import sys
import unicodedata
from collections import defaultdict, deque
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA = PROJECT_ROOT / "tennis_data" / "tennis_matches.csv"
OUT_DIR = PROJECT_ROOT / "ml" / "outputs"
MODEL_OUT = OUT_DIR / "tennis_ml_model.json"
STATES_OUT = OUT_DIR / "tennis_player_states.json"
CMP_OUT = OUT_DIR / "tennis_models_backtest.json"
SERVE_SERIES = OUT_DIR / "tennis_serve_series.csv"

START_ELO = 1500.0
SURFACES = ["Hard", "Clay", "Grass"]
PRIOR_N = 20          # cuantos partidos hasta confiar plenamente en el Elo
GAP_DAYS = 120        # parate que dispara reversion a la media
GAP_REVERT = 0.25     # fraccion que revierte hacia 1500 tras el parate
TEST_FROM = pd.Timestamp("2025-01-01")
TRAIN_FROM = pd.Timestamp("2021-06-01")   # warmup de Elo antes de entrenar ML


def norm_name(s):
    s = "".join(ch for ch in unicodedata.normalize("NFD", str(s)) if not unicodedata.combining(ch))
    return " ".join(s.lower().replace(".", "").split())


def rank_to_elo(rank):
    """Elo implicito por ranking (rank 1 ~2100, 100 ~1600, 1000 ~1350)."""
    try:
        r = float(rank)
        if not np.isfinite(r) or r < 1:
            r = 500.0
    except (TypeError, ValueError):
        r = 500.0
    return 2100.0 - 250.0 * math.log10(max(r, 1.0))


def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-x))


class TennisEloV2:
    def __init__(self):
        self.elo = defaultdict(lambda: START_ELO)
        self.surf = defaultdict(lambda: START_ELO)
        self.n = defaultdict(int)
        self.n_surf = defaultdict(int)
        self.last_date = {}
        self.last_rank = {}
        self.form = defaultdict(lambda: deque(maxlen=10))  # 1 gano / 0 perdio

    @staticmethod
    def _k(n):
        return 250.0 / ((n + 5) ** 0.4)

    def _revert_if_gap(self, key, date):
        ld = self.last_date.get(key)
        if ld is not None and (date - ld).days > GAP_DAYS:
            self.elo[key] = START_ELO + (self.elo[key] - START_ELO) * (1 - GAP_REVERT)
            for s in SURFACES:
                k2 = key + (s,)
                self.surf[k2] = START_ELO + (self.surf[k2] - START_ELO) * (1 - GAP_REVERT)

    def eff_elo(self, key, rank, surface):
        """Elo efectivo (general y superficie) con prior por ranking para n bajo."""
        n = self.n[key]
        w = n / (n + PRIOR_N)
        re = rank_to_elo(rank if rank is not None else self.last_rank.get(key))
        eff_gen = w * self.elo[key] + (1 - w) * re
        if surface in SURFACES:
            ns = self.n_surf[(key[0], key[1], surface)]
            ws = ns / (ns + PRIOR_N)
            eff_surf = ws * self.surf[(key[0], key[1], surface)] + (1 - ws) * eff_gen
        else:
            eff_surf = eff_gen
        return eff_gen, eff_surf

    def prob(self, tour, a, b, surface, rank_a, rank_b, best_of):
        ka, kb = (tour, norm_name(a)), (tour, norm_name(b))
        ega, esa = self.eff_elo(ka, rank_a, surface)
        egb, esb = self.eff_elo(kb, rank_b, surface)
        ra = 0.5 * ega + 0.5 * esa
        rb = 0.5 * egb + 0.5 * esb
        logit = (ra - rb) / 400.0 * math.log(10)
        try:
            bo5 = int(float(best_of)) == 5
        except (TypeError, ValueError):
            bo5 = False
        if bo5:
            logit *= 1.15          # best-of-5: mas ventaja al mejor
        return 1.0 / (1.0 + math.exp(-logit)), (ega, esa, egb, esb)

    def update(self, tour, winner, loser, surface, date, mov_mult):
        kw, kl = (tour, norm_name(winner)), (tour, norm_name(loser))
        self._revert_if_gap(kw, date)
        self._revert_if_gap(kl, date)
        # prob previa (sin ajuste best-of, para el update crudo)
        rw = 0.5 * self.elo[kw] + 0.5 * self.surf[(kw[0], kw[1], surface)] if surface in SURFACES else self.elo[kw]
        rl = 0.5 * self.elo[kl] + 0.5 * self.surf[(kl[0], kl[1], surface)] if surface in SURFACES else self.elo[kl]
        p_w = 1.0 / (1.0 + 10 ** (-(rw - rl) / 400.0))
        kw_f = self._k(self.n[kw]) * mov_mult
        kl_f = self._k(self.n[kl]) * mov_mult
        self.elo[kw] += kw_f * (1 - p_w)
        self.elo[kl] -= kl_f * (1 - p_w)
        if surface in SURFACES:
            ksw = self._k(self.n_surf[(kw[0], kw[1], surface)]) * mov_mult
            ksl = self._k(self.n_surf[(kl[0], kl[1], surface)]) * mov_mult
            self.surf[(kw[0], kw[1], surface)] += ksw * (1 - p_w)
            self.surf[(kl[0], kl[1], surface)] -= ksl * (1 - p_w)
            self.n_surf[(kw[0], kw[1], surface)] += 1
            self.n_surf[(kl[0], kl[1], surface)] += 1
        self.n[kw] += 1
        self.n[kl] += 1
        self.form[kw].append(1)
        self.form[kl].append(0)
        self.last_date[kw] = date
        self.last_date[kl] = date

    def form_rate(self, key):
        f = self.form[key]
        return sum(f) / len(f) if len(f) >= 3 else 0.5

    def days_rest(self, key, date):
        ld = self.last_date.get(key)
        return min((date - ld).days, 60) if ld is not None else 30


def mov_from_sets(wsets, lsets, comment):
    c = str(comment).lower()
    if "walkover" in c or "awarded" in c or "disq" in c:
        return None                      # no hubo partido real
    try:
        ws, ls = float(wsets), float(lsets)
        tot = ws + ls
        dom = ws / tot if tot > 0 else 0.66
    except (TypeError, ValueError):
        dom = 0.66
    mult = float(np.clip(0.8 + (dom - 0.6) * 1.2, 0.8, 1.2))
    if "retired" in c:
        mult *= 0.5                       # victoria incompleta
    return mult


FEATURES = ["elo_diff", "surf_diff", "p_elo_logit", "rank_diff",
            "form_diff", "rest_diff", "best_of5", "n_min",
            "serve_diff", "ret_diff"]

_SERVE = {}


def _serve_index():
    """{key: (fechas ordenadas, saques, devoluciones)} para buscar el valor as-of.
    Si no existe el archivo de Sackmann, las features quedan en 0 y el modelo
    entrena exactamente como antes (sin regresion)."""
    if "idx" not in _SERVE:
        idx = {}
        if SERVE_SERIES.exists():
            try:
                sd = pd.read_csv(SERVE_SERIES, parse_dates=["date"]).sort_values("date")
                for k, g in sd.groupby("key"):
                    idx[k] = (g["date"].values, g["serve"].values, g["ret"].values)
            except Exception:
                idx = {}
        _SERVE["idx"] = idx
    return _SERVE["idx"]


def serve_asof(tour, name, date):
    """(saque, devolucion) del jugador ANTES de esa fecha, o (None, None)."""
    idx = _serve_index()
    if not idx:
        return None, None
    # las claves de Sackmann son 'TOUR|apellido i' (mismo formato que los estados)
    cand = _match_key(_serve_index(), tour, name)
    if cand is None:
        return None, None
    dates, sv, rt = idx[cand]
    i = np.searchsorted(dates, np.datetime64(pd.Timestamp(date)), side="left") - 1
    if i < 0:
        return None, None
    return float(sv[i]), float(rt[i])


def sequential_pass(m):
    """Recorre los partidos en orden, emite features as-of + prob Elo v2 + label."""
    elo = TennisEloV2()
    rows = []
    for r in m.itertuples(index=False):
        tour, surface, date = r.tour, r.Surface, r.Date
        bo = getattr(r, "BestOf", 3)
        rank_w = getattr(r, "WRank", np.nan)
        rank_l = getattr(r, "LRank", np.nan)
        mov = mov_from_sets(getattr(r, "Wsets", np.nan), getattr(r, "Lsets", np.nan),
                            getattr(r, "Comment", ""))
        if mov is None:
            continue                       # walkover: ni feature ni update
        kw, kl = (tour, norm_name(r.Winner)), (tour, norm_name(r.Loser))
        # --- features as-of, orientadas A=ganador (label 1) ---
        p_elo, (ega, esa, egb, esb) = elo.prob(tour, r.Winner, r.Loser, surface, rank_w, rank_l, bo)
        rw = 0.5 * ega + 0.5 * esa
        rl = 0.5 * egb + 0.5 * esb
        rank_diff = math.log10(max(_num(rank_l, 500), 1)) - math.log10(max(_num(rank_w, 500), 1))
        feat = {
            "elo_diff": (ega - egb) / 100.0,
            "surf_diff": (esa - esb) / 100.0,
            "p_elo_logit": math.log(min(max(p_elo, 1e-4), 1 - 1e-4) / (1 - min(max(p_elo, 1e-4), 1 - 1e-4))),
            "rank_diff": rank_diff,
            "form_diff": elo.form_rate(kw) - elo.form_rate(kl),
            "rest_diff": (elo.days_rest(kw, date) - elo.days_rest(kl, date)) / 10.0,
            "best_of5": 1.0 if int(_num(bo, 3)) == 5 else 0.0,
            "n_min": min(elo.n[kw], elo.n[kl]) / 50.0,
            "serve_diff": 0.0,
            "ret_diff": 0.0,
        }
        sw, rw_ = serve_asof(tour, r.Winner, date)
        sl, rl_ = serve_asof(tour, r.Loser, date)
        if sw is not None and sl is not None:
            feat["serve_diff"] = (sw - sl) * 10.0        # escala comparable al resto
            feat["ret_diff"] = (rw_ - rl_) * 10.0
        avg_w = getattr(r, "AvgW", np.nan)
        avg_l = getattr(r, "AvgL", np.nan)
        ps_w = getattr(r, "PSW", np.nan)
        ps_l = getattr(r, "PSL", np.nan)
        rows.append({"date": date, "tour": tour, "surface": surface,
                     "p_elo": p_elo, "n_w": elo.n[kw], "n_l": elo.n[kl],
                     "avg_w": avg_w, "avg_l": avg_l, "ps_w": ps_w, "ps_l": ps_l, **feat})
        # actualizar Elo con el resultado
        elo.update(tour, r.Winner, r.Loser, surface, date, mov)
        # guardar rank vigente
        if np.isfinite(_num(rank_w, np.nan)):
            elo.last_rank[kw] = _num(rank_w, np.nan)
        if np.isfinite(_num(rank_l, np.nan)):
            elo.last_rank[kl] = _num(rank_l, np.nan)
    return elo, pd.DataFrame(rows)


def _num(x, default):
    try:
        v = float(x)
        return v if np.isfinite(v) else default
    except (TypeError, ValueError):
        return default


def _fit_logistic(X, y, iters=300, lr=0.3, l2=1.0, features=None):
    """Regresion logistica simple (sin sklearn) con estandarizacion."""
    mu = X.mean(axis=0)
    sd = X.std(axis=0) + 1e-9
    Xs = (X - mu) / sd
    n, d = Xs.shape
    w = np.zeros(d)
    b = 0.0
    for _ in range(iters):
        z = Xs @ w + b
        p = sigmoid(z)
        g = p - y
        w -= lr * (Xs.T @ g / n + l2 / n * w)
        b -= lr * g.mean()
    return {"w": w.tolist(), "b": float(b), "mu": mu.tolist(), "sd": sd.tolist(),
            "features": features if features is not None else FEATURES}


def _predict_logistic(model, feat_dict):
    w = np.array(model["w"]); mu = np.array(model["mu"]); sd = np.array(model["sd"])
    x = np.array([feat_dict[f] for f in model["features"]])
    xs = (x - mu) / sd
    return float(sigmoid(xs @ w + model["b"]))


def _symmetric_dataset(df):
    """Cada partido -> 2 filas: ganador como A (1) y perdedor como A (0)."""
    Xs, ys = [], []
    for _, r in df.iterrows():
        f = np.array([r[c] for c in FEATURES])
        Xs.append(f); ys.append(1.0)          # A = ganador
        Xs.append(f_mirror(f)); ys.append(0.0)  # A = perdedor (features invertidas)
    return np.array(Xs), np.array(ys)


def f_mirror(f):
    """Invierte las diferencias (best_of5 y n_min no cambian de signo)."""
    fm = -f.copy()
    idx_bo = FEATURES.index("best_of5"); idx_nm = FEATURES.index("n_min")
    fm[idx_bo] = f[idx_bo]
    fm[idx_nm] = f[idx_nm]
    return fm


def _metrics(p_winner):
    p = np.clip(np.asarray(p_winner, float), 1e-4, 1 - 1e-4)
    return float((p > 0.5).mean()), float(-np.log(p).mean())


BLEND_OUT = OUT_DIR / "tennis_blend.json"
BLEND_FEATURES = ["L_elo", "L_ml", "L_mkt"]
_BLEND = {}


def _logit(pp):
    pp = min(max(float(pp), 1e-4), 1 - 1e-4)
    return math.log(pp / (1 - pp))


def _fit_blend(p_elo, p_ml, p_mkt):
    """Meta-logistica sobre los logits de las 3 probabilidades (P de A)."""
    Le = np.array([_logit(x) for x in p_elo])
    Lm = np.array([_logit(x) for x in p_ml])
    Lk = np.array([_logit(x) for x in p_mkt])
    X = np.column_stack([Le, Lm, Lk])
    Xs = np.vstack([X, -X])                    # simetrico: A=ganador (1) y A=perdedor (0)
    ys = np.concatenate([np.ones(len(X)), np.zeros(len(X))])
    return _fit_logistic(Xs, ys, features=BLEND_FEATURES)


def _blend_model():
    if "m" not in _BLEND:
        try:
            _BLEND["m"] = json.loads(BLEND_OUT.read_text(encoding="utf-8")) if BLEND_OUT.exists() else None
        except Exception:
            _BLEND["m"] = None
    return _BLEND["m"]


def blend_prob(p_elo, p_ml, p_mkt):
    """Combina las 3 probabilidades (de que gane A) en la del blend, o None."""
    if p_elo is None or p_ml is None or p_mkt is None:
        return None
    model = _blend_model()
    if model is None:
        return None
    feat = {"L_elo": _logit(p_elo), "L_ml": _logit(p_ml), "L_mkt": _logit(p_mkt)}
    return round(_predict_logistic(model, feat), 3)


ELO_CALIB_OUT = OUT_DIR / "tennis_elo_calib.json"
_ELO_T = {}


def _elo_temp():
    if "T" not in _ELO_T:
        try:
            _ELO_T["T"] = float(json.loads(ELO_CALIB_OUT.read_text(encoding="utf-8"))["T"]) if ELO_CALIB_OUT.exists() else 1.0
        except Exception:
            _ELO_T["T"] = 1.0
    return _ELO_T["T"]


def calibrate_elo(p):
    """Escala de temperatura: encoge la (sobre)confianza del Elo v2 (T>1)."""
    if p is None:
        return p
    T = _elo_temp()
    if T == 1.0:
        return p
    return 1.0 / (1.0 + math.exp(-_logit(p) / T))


def _fit_elo_temp(ft):
    """Busca la temperatura T que minimiza el log-loss del Elo (dataset simetrico)."""
    L = np.array([_logit(x) for x in ft["p_elo"].values])
    Ls = np.concatenate([L, -L])
    ys = np.concatenate([np.ones(len(L)), np.zeros(len(L))])
    best_T, best_ll = 1.0, 1e9
    for T in np.arange(1.0, 2.01, 0.05):
        pr = np.clip(1.0 / (1.0 + np.exp(-Ls / T)), 1e-4, 1 - 1e-4)
        ll = -(ys * np.log(pr) + (1 - ys) * np.log(1 - pr)).mean()
        if ll < best_ll:
            best_ll, best_T = ll, round(float(T), 2)
    return best_T


def backtest():
    m = pd.read_csv(DATA, parse_dates=["Date"])
    m = m.rename(columns={"Best of": "BestOf"})
    m = m.dropna(subset=["Winner", "Loser", "Surface", "Date"]).sort_values("Date")
    elo, feats = sequential_pass(m)
    feats = feats[(feats["n_w"] >= 10) & (feats["n_l"] >= 10)].copy()
    train = feats[(feats["date"] >= TRAIN_FROM) & (feats["date"] < TEST_FROM)]
    test = feats[feats["date"] >= TEST_FROM].copy()

    # entrenar ML
    X, y = _symmetric_dataset(train)
    model = _fit_logistic(X, y)
    test["p_ml"] = test.apply(lambda r: _predict_logistic(model, {c: r[c] for c in FEATURES}), axis=1).values
    p_ml = test["p_ml"].values

    # Elo v2 calibrado (temperatura) — prob de que gane el ganador real
    p_elo = np.array([calibrate_elo(float(x)) for x in test["p_elo"].values])
    acc_elo, ll_elo = _metrics(p_elo)
    acc_ml, ll_ml = _metrics(p_ml)

    # mercado
    both = test["avg_w"].notna() & test["avg_l"].notna() & (test["avg_w"] > 1) & (test["avg_l"] > 1)
    b = test[both]
    iw, il = 1 / b["avg_w"], 1 / b["avg_l"]
    fair_w = (iw / (iw + il)).values
    acc_mkt, ll_mkt = _metrics(fair_w)

    # blend (Elo v2 + ML + mercado) sobre las filas con mercado
    pe_calb = np.array([calibrate_elo(float(x)) for x in b["p_elo"].values])
    blend_model = _fit_blend(pe_calb, b["p_ml"].values, fair_w)
    p_blend = np.array([_predict_logistic(blend_model,
                        {"L_elo": _logit(pe), "L_ml": _logit(pm), "L_mkt": _logit(pk)})
                        for pe, pm, pk in zip(pe_calb, b["p_ml"].values, fair_w)])
    acc_bl, ll_bl = _metrics(p_blend)

    res = {
        "n_test": int(len(test)), "desde": str(TEST_FROM.date()),
        "elo_v2": {"accuracy": round(acc_elo * 100, 2), "log_loss": round(ll_elo, 4)},
        "ml": {"accuracy": round(acc_ml * 100, 2), "log_loss": round(ll_ml, 4)},
        "mercado": {"accuracy": round(acc_mkt * 100, 2), "log_loss": round(ll_mkt, 4),
                    "n": int(both.sum())},
        "blend": {"accuracy": round(acc_bl * 100, 2), "log_loss": round(ll_bl, 4)},
    }
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    with open(CMP_OUT, "w", encoding="utf-8") as f:
        json.dump(res, f, indent=2, ensure_ascii=False); f.flush()
    print(json.dumps(res, indent=2, ensure_ascii=False))
    return res




# ============================================================
# Produccion: entrenar y guardar / predecir un partido
# ============================================================
def fit():
    m = pd.read_csv(DATA, parse_dates=["Date"])
    m = m.rename(columns={"Best of": "BestOf"})
    m = m.dropna(subset=["Winner", "Loser", "Surface", "Date"]).sort_values("Date")
    elo, feats = sequential_pass(m)
    ft = feats[(feats["n_w"] >= 10) & (feats["n_l"] >= 10)]
    X, y = _symmetric_dataset(ft)
    model = _fit_logistic(X, y)
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    with open(MODEL_OUT, "w", encoding="utf-8") as f:
        json.dump(model, f); f.flush(); os.fsync(f.fileno())
    # calibracion de temperatura del Elo v2 (arregla la sobreconfianza)
    T = _fit_elo_temp(ft)
    _ELO_T.clear()
    with open(ELO_CALIB_OUT, "w", encoding="utf-8") as f:
        json.dump({"T": T}, f); f.flush(); os.fsync(f.fileno())
    print(f"Calibracion Elo v2: T = {T}")
    # blend con las cuotas del mercado disponibles
    ftm = ft[ft["avg_w"].notna() & ft["avg_l"].notna() & (ft["avg_w"] > 1) & (ft["avg_l"] > 1)].copy()
    if len(ftm) > 100:
        pml = ftm.apply(lambda r: _predict_logistic(model, {c: r[c] for c in FEATURES}), axis=1).values
        iw2, il2 = 1 / ftm["avg_w"].values, 1 / ftm["avg_l"].values
        pmkt = iw2 / (iw2 + il2)
        pe_cal = np.array([calibrate_elo(float(x)) for x in ftm["p_elo"].values])
        blend_model = _fit_blend(pe_cal, pml, pmkt)
        with open(BLEND_OUT, "w", encoding="utf-8") as f:
            json.dump(blend_model, f); f.flush(); os.fsync(f.fileno())
    states = {}
    for (tour, name), rating in elo.elo.items():
        if elo.n[(tour, name)] < 5:
            continue
        ld = elo.last_date.get((tour, name))
        states[f"{tour}|{name}"] = {
            "elo": round(rating, 1),
            "surf": {sf: round(elo.surf[(tour, name, sf)], 1) for sf in SURFACES},
            "n": elo.n[(tour, name)],
            "n_surf": {sf: elo.n_surf[(tour, name, sf)] for sf in SURFACES},
            "last_rank": elo.last_rank.get((tour, name)),
            "form": round(elo.form_rate((tour, name)), 3),
            "last_date": str(pd.Timestamp(ld).date()) if ld is not None else None,
        }
    with open(STATES_OUT, "w", encoding="utf-8") as f:
        json.dump(states, f); f.flush(); os.fsync(f.fileno())
    print(f"Guardado: modelo ML + {len(states)} estados de jugadores")
    backtest()


import os  # para fsync en fit


_CACHE = {}


def _load():
    if not _CACHE:
        _CACHE["model"] = json.loads(MODEL_OUT.read_text(encoding="utf-8")) if MODEL_OUT.exists() else None
        _CACHE["states"] = json.loads(STATES_OUT.read_text(encoding="utf-8")) if STATES_OUT.exists() else {}
    return _CACHE["model"], _CACHE["states"]


def _eff_from_state(st, surface, prior_gen=None, prior_surf=None):
    """Elo efectivo. prior_gen/prior_surf (Elo ampliado Sackmann) reemplazan al
    prior por ranking cuando el jugador tiene pocos partidos (arregla desconocidos)."""
    elo = st["elo"]; n = st["n"]
    w = n / (n + PRIOR_N)
    re = prior_gen if prior_gen is not None else rank_to_elo(st.get("last_rank"))
    eff_gen = w * elo + (1 - w) * re
    if surface in SURFACES:
        ns = st["n_surf"].get(surface, 0)
        ws = ns / (ns + PRIOR_N)
        base_surf = st["surf"].get(surface, elo)
        sp = prior_surf if prior_surf is not None else eff_gen
        eff_surf = ws * base_surf + (1 - ws) * sp
    else:
        eff_surf = eff_gen
    return eff_gen, eff_surf


SACK_OUT = OUT_DIR / "tennis_sackmann_states.json"
_SACK = {}


def _sack_states():
    if "s" not in _SACK:
        try:
            _SACK["s"] = json.loads(SACK_OUT.read_text(encoding="utf-8")) if SACK_OUT.exists() else {}
        except Exception:
            _SACK["s"] = {}
    return _SACK["s"]


def _sack_prior(sk, surface):
    """(Elo general, Elo superficie) del rating ampliado, o (None, None)."""
    if not sk:
        return None, None
    ps = sk.get("surf", {}).get(surface) if surface in SURFACES else None
    return sk.get("elo"), ps


def _state_from_sack(sk):
    """Pseudo-estado primario a partir del Sackmann (jugador ausente del principal)."""
    return {"elo": sk.get("elo", START_ELO), "surf": sk.get("surf", {}),
            "n": sk.get("n", 0), "n_surf": sk.get("n_surf", {}),
            "last_rank": None, "form": sk.get("form", 0.5)}


def _match_key(states, tour, full_name):
    """'Jannik Sinner' -> 'ATP|sinner j' buscando en los estados guardados."""
    toks = norm_name(full_name).split()
    if not toks:
        return None
    fi = toks[0][0]
    for start in range(1, len(toks)):
        cand = f"{tour}|" + " ".join(toks[start:]) + " " + fi
        if cand in states:
            return cand
    cand = f"{tour}|" + " ".join(toks[:-1]) + " " + toks[-1][0]
    if cand in states:
        return cand
    direct = f"{tour}|{norm_name(full_name)}"
    return direct if direct in states else None


def predict_pair(tour, name_a, name_b, surface, best_of=3):
    """Devuelve (p_elo_v2, p_ml) de que gane A. None si falta el modelo."""
    model, states = _load()
    sack = _sack_states()
    ka = _match_key(states, tour, name_a)
    kb = _match_key(states, tour, name_b)
    sa = states.get(ka) if ka else None
    sb = states.get(kb) if kb else None
    ska = _match_key(sack, tour, name_a)
    skb = _match_key(sack, tour, name_b)
    sacka = sack.get(ska) if ska else None
    sackb = sack.get(skb) if skb else None
    if sa is None and sacka:
        sa = _state_from_sack(sacka)          # ausente del principal -> uso el ampliado
    if sb is None and sackb:
        sb = _state_from_sack(sackb)
    if sa is None or sb is None:
        return None, None
    pga, psa = _sack_prior(sacka, surface)
    pgb, psb = _sack_prior(sackb, surface)
    ega, esa = _eff_from_state(sa, surface, prior_gen=pga, prior_surf=psa)
    egb, esb = _eff_from_state(sb, surface, prior_gen=pgb, prior_surf=psb)
    ra = 0.5 * ega + 0.5 * esa
    rb = 0.5 * egb + 0.5 * esb
    logit = (ra - rb) / 400.0 * math.log(10)
    try:
        if int(float(best_of)) == 5:
            logit *= 1.15
    except (TypeError, ValueError):
        pass
    p_elo = 1.0 / (1.0 + math.exp(-logit))
    p_ml = None
    if model is not None:
        ranka = _num(sa.get("last_rank"), 500); rankb = _num(sb.get("last_rank"), 500)
        p_clamp = min(max(p_elo, 1e-4), 1 - 1e-4)
        feat = {
            "elo_diff": (ega - egb) / 100.0,
            "surf_diff": (esa - esb) / 100.0,
            "p_elo_logit": math.log(p_clamp / (1 - p_clamp)),
            "rank_diff": math.log10(max(rankb, 1)) - math.log10(max(ranka, 1)),
            "form_diff": sa.get("form", 0.5) - sb.get("form", 0.5),
            "rest_diff": 0.0,
            "best_of5": 1.0 if str(best_of) in ("5", "5.0") else 0.0,
            "n_min": min(sa["n"], sb["n"]) / 50.0,
            "serve_diff": 0.0,
            "ret_diff": 0.0,
        }
        _sw, _rw = serve_asof(tour, name_a, pd.Timestamp.now())
        _sl, _rl = serve_asof(tour, name_b, pd.Timestamp.now())
        if _sw is not None and _sl is not None:
            feat["serve_diff"] = (_sw - _sl) * 10.0
            feat["ret_diff"] = (_rw - _rl) * 10.0
        p_ml = _predict_logistic(model, feat)
    p_elo_out = calibrate_elo(p_elo)   # reportado/apostable = calibrado; feature ML = crudo
    return round(p_elo_out, 3), (round(p_ml, 3) if p_ml is not None else None)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--backtest", action="store_true")
    ap.add_argument("--fit", action="store_true")
    a = ap.parse_args()
    if a.backtest:
        backtest()
    else:
        fit()
