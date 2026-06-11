"""Goal-expectation model: Elo difference -> expected goals -> Dixon-Coles
score matrix -> market probabilities.

Pipeline:
  1. Poisson GLM (fit by IRLS) maps the Elo rating difference to each team's
     expected goals.  Features: intercept, scaled rating diff, and a
     "competitive match" flag (friendlies are lower-scoring / lower-effort).
     Training rows are weighted with a 10-year half-life time decay.
  2. The Dixon-Coles low-score correction (rho) is fit by maximum likelihood
     on recent matches — it fixes the independent-Poisson model's known bias
     on 0-0 / 1-1 type scorelines.
  3. The joint score matrix gives every market: 1X2, totals, BTTS,
     correct scores, and knockout advancement (ET + penalties).
"""
import math

import numpy as np
import pandas as pd
from scipy.optimize import minimize_scalar
from scipy.stats import poisson

from .elo import HOME_ADV, BASE_RATING

MAX_GOALS = 10
DECAY_HALF_LIFE_YEARS = 10.0
FIT_WINDOW_YEARS = 40
RHO_WINDOW_YEARS = 15


# ---------------------------------------------------------------- GLM fit

def _build_training_rows(history, fit_date):
    """Two rows per match (each team's perspective). Returns X, y, weights."""
    rows = []
    min_date = fit_date - pd.Timedelta(days=365 * FIT_WINDOW_YEARS)
    for m in history:
        if m["date"] < min_date or m["date"] >= fit_date:
            continue
        years_ago = (fit_date - m["date"]).days / 365.25
        w = 0.5 ** (years_ago / DECAY_HALF_LIFE_YEARS)
        comp = 0.0 if m["tournament"].lower() == "friendly" else 1.0
        rh_adj = m["elo_home"] + (0.0 if m["neutral"] else HOME_ADV)
        ra = m["elo_away"]
        x = (rh_adj - ra) / 400.0
        rows.append((x, comp, m["home_score"], w))
        rows.append((-x, comp, m["away_score"], w))
    arr = np.array(rows, dtype=float)
    X = np.column_stack([np.ones(len(arr)), arr[:, 0], arr[:, 1]])
    return X, arr[:, 2], arr[:, 3]


def _fit_poisson_glm(X, y, w, iters=100, tol=1e-9):
    """Poisson regression via iteratively reweighted least squares."""
    beta = np.zeros(X.shape[1])
    beta[0] = math.log(max(np.average(y, weights=w), 1e-6))
    for _ in range(iters):
        eta = np.clip(X @ beta, -8.0, 4.0)
        mu = np.exp(eta)
        W = w * mu
        z = eta + (y - mu) / mu
        XtW = X.T * W
        beta_new = np.linalg.solve(XtW @ X, XtW @ z)
        if np.max(np.abs(beta_new - beta)) < tol:
            beta = beta_new
            break
        beta = beta_new
    return beta


# ------------------------------------------------------- Dixon-Coles rho

def _dc_tau(i, j, lam, mu, rho):
    if i == 0 and j == 0:
        return 1.0 - lam * mu * rho
    if i == 0 and j == 1:
        return 1.0 + lam * rho
    if i == 1 and j == 0:
        return 1.0 + mu * rho
    if i == 1 and j == 1:
        return 1.0 - rho
    return 1.0


def _fit_rho(history, beta, fit_date):
    """Profile-likelihood fit of the DC correlation parameter."""
    min_date = fit_date - pd.Timedelta(days=365 * RHO_WINDOW_YEARS)
    lams, mus, hs, as_ = [], [], [], []
    for m in history:
        if m["date"] < min_date or m["date"] >= fit_date:
            continue
        comp = 0.0 if m["tournament"].lower() == "friendly" else 1.0
        rh_adj = m["elo_home"] + (0.0 if m["neutral"] else HOME_ADV)
        x = (rh_adj - m["elo_away"]) / 400.0
        lams.append(math.exp(beta[0] + beta[1] * x + beta[2] * comp))
        mus.append(math.exp(beta[0] - beta[1] * x + beta[2] * comp))
        hs.append(m["home_score"])
        as_.append(m["away_score"])
    lams, mus = np.array(lams), np.array(mus)
    hs, as_ = np.array(hs), np.array(as_)

    base_ll = (poisson.logpmf(hs, lams) + poisson.logpmf(as_, mus))

    def neg_ll(rho):
        tau = np.ones(len(hs))
        m00 = (hs == 0) & (as_ == 0)
        m01 = (hs == 0) & (as_ == 1)
        m10 = (hs == 1) & (as_ == 0)
        m11 = (hs == 1) & (as_ == 1)
        tau[m00] = 1.0 - lams[m00] * mus[m00] * rho
        tau[m01] = 1.0 + lams[m01] * rho
        tau[m10] = 1.0 + mus[m10] * rho
        tau[m11] = 1.0 - rho
        tau = np.clip(tau, 1e-10, None)
        return -np.sum(base_ll + np.log(tau))

    res = minimize_scalar(neg_ll, bounds=(-0.25, 0.25), method="bounded")
    return float(res.x)


# ----------------------------------------------------------- score matrix

def score_matrix(lam, mu, rho, max_goals=MAX_GOALS):
    goals = np.arange(max_goals + 1)
    ph = poisson.pmf(goals, lam)
    pa = poisson.pmf(goals, mu)
    M = np.outer(ph, pa)
    for i in (0, 1):
        for j in (0, 1):
            M[i, j] *= max(_dc_tau(i, j, lam, mu, rho), 0.0)
    return M / M.sum()


def matrix_markets(M):
    """Extract market probabilities from a score matrix."""
    home = np.tril(M, -1).sum()
    away = np.triu(M, 1).sum()
    draw = np.trace(M)
    n = M.shape[0]
    totals = {}
    idx = np.add.outer(np.arange(n), np.arange(n))
    for line in (1.5, 2.5, 3.5):
        totals[line] = M[idx > line].sum()
    btts = M[1:, 1:].sum()
    scores = [
        ((i, j), M[i, j])
        for i in range(min(6, n))
        for j in range(min(6, n))
    ]
    scores.sort(key=lambda kv: -kv[1])
    return {
        "home": float(home), "draw": float(draw), "away": float(away),
        "over": {k: float(v) for k, v in totals.items()},
        "btts": float(btts),
        "top_scores": scores[:8],
    }


# ----------------------------------------------------------- match model

class MatchModel:
    """Bundles fitted Elo ratings + goal GLM + DC rho into one predictor."""

    def __init__(self, elo, beta, rho, fit_date):
        self.elo = elo
        self.beta = beta
        self.rho = rho
        self.fit_date = fit_date

    @classmethod
    def fit(cls, played, elo=None, fit_date=None):
        from .elo import EloRatings
        if elo is None:
            elo = EloRatings().fit(played)
        if fit_date is None:
            fit_date = played["date"].max() + pd.Timedelta(days=1)
        X, y, w = _build_training_rows(elo.history, fit_date)
        beta = _fit_poisson_glm(X, y, w)
        rho = _fit_rho(elo.history, beta, fit_date)
        return cls(elo, beta, rho, fit_date)

    # venue: "neutral", "a_home", or "b_home"
    def expected_goals(self, team_a, team_b, venue="neutral", competitive=True):
        ra, rb = self.elo.get(team_a), self.elo.get(team_b)
        if venue == "a_home":
            ra += HOME_ADV
        elif venue == "b_home":
            rb += HOME_ADV
        x = (ra - rb) / 400.0
        comp = 1.0 if competitive else 0.0
        lam = math.exp(self.beta[0] + self.beta[1] * x + self.beta[2] * comp)
        mu = math.exp(self.beta[0] - self.beta[1] * x + self.beta[2] * comp)
        return lam, mu

    def predict(self, team_a, team_b, venue="neutral", knockout=False,
                competitive=True):
        lam, mu = self.expected_goals(team_a, team_b, venue, competitive)
        M = score_matrix(lam, mu, self.rho)
        out = matrix_markets(M)
        out.update(
            team_a=team_a, team_b=team_b,
            elo_a=self.elo.get(team_a), elo_b=self.elo.get(team_b),
            xg_a=lam, xg_b=mu, matrix=M,
        )
        if knockout:
            # Extra time: same scoring rates over 30 minutes.
            M_et = score_matrix(lam / 3.0, mu / 3.0, self.rho, max_goals=6)
            et = matrix_markets(M_et)
            # Penalties treated as a coin flip (historically near 50/50).
            p_pens = 0.5
            adv_a = out["home"] + out["draw"] * (et["home"] + et["draw"] * p_pens)
            out["advance_a"] = float(adv_a)
            out["advance_b"] = 1.0 - float(adv_a)
        return out
