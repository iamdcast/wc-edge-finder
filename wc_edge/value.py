"""Value detection: devig bookmaker odds, compute EV and Kelly stakes.

The core idea: a bet is worth making not when it's likely to win, but when
the bookmaker's implied probability is LOWER than the true probability.
EV per $1 staked = p_model * odds - 1.
"""
import numpy as np


def american_to_decimal(a):
    """American odds (-150, +200) -> decimal odds. |a| must be >= 100."""
    return 1.0 + (a / 100.0 if a > 0 else 100.0 / abs(a))


def decimal_to_american(d):
    """Decimal odds -> American odds. d=2.0 -> +100 by convention."""
    if d >= 2.0:
        return (d - 1.0) * 100.0
    return -100.0 / (d - 1.0)


def implied_probs(odds):
    """Decimal odds -> raw implied probabilities (contains the vig)."""
    return np.array([1.0 / o for o in odds])


def overround(odds):
    """Total implied probability; the excess over 1.0 is the bookmaker margin."""
    return float(implied_probs(odds).sum())


def devig_proportional(odds):
    """Scale implied probabilities to sum to 1 (multiplicative method)."""
    p = implied_probs(odds)
    return p / p.sum()


def devig_shin(odds, tol=1e-10, max_iter=100):
    """Shin's method — accounts for the favorite-longshot bias by modelling
    a fraction z of insider bettors. Better fair-prob estimates on lopsided
    markets than the proportional method."""
    pi = implied_probs(odds)
    s = pi.sum()
    lo, hi = 0.0, 0.4
    for _ in range(max_iter):
        z = (lo + hi) / 2.0
        p = (np.sqrt(z * z + 4.0 * (1.0 - z) * pi * pi / s) - z) / (2.0 * (1.0 - z))
        total = p.sum()
        if abs(total - 1.0) < tol:
            break
        if total > 1.0:
            hi = z
        else:
            lo = z
    return p / p.sum()


def devig(odds, method="shin"):
    if method == "shin":
        return devig_shin(odds)
    return devig_proportional(odds)


def expected_value(p_model, odds):
    """EV per unit staked."""
    return p_model * odds - 1.0


def kelly_fraction(p_model, odds):
    """Full-Kelly fraction of bankroll. 0 if no edge."""
    b = odds - 1.0
    if b <= 0:
        return 0.0
    return max(0.0, (p_model * odds - 1.0) / b)


def evaluate_market(label, p_model, odds, fair_p, bankroll=1000.0,
                    kelly_mult=0.25, min_edge=0.02, max_stake_pct=0.10):
    """Build one row of the value table for a single selection."""
    ev = expected_value(p_model, odds)
    edge = p_model - fair_p
    kelly = kelly_fraction(p_model, odds) * kelly_mult
    stake = min(kelly, max_stake_pct) * bankroll if ev >= min_edge else 0.0
    if ev >= min_edge and edge > 0.15:
        verdict = "CHECK INPUTS"  # too good to be true, usually is
    elif ev >= min_edge:
        verdict = "BET"
    elif ev >= 0:
        verdict = "marginal"
    else:
        verdict = "no value"
    return {
        "Market": label,
        "Model %": p_model,
        "Fair odds": (1.0 / p_model) if p_model > 0 else float("inf"),
        "Book odds": odds,
        "Book implied %": 1.0 / odds,
        "Devig %": fair_p,
        "EV %": ev,
        "Stake $": stake,
        "Verdict": verdict,
    }
