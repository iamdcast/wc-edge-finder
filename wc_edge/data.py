"""Load the international results dataset (martj42/international_results)."""
import urllib.request
from pathlib import Path

import pandas as pd

DATA_DIR = Path(__file__).resolve().parent.parent / "data"
RESULTS_PATH = DATA_DIR / "results.csv"
RESULTS_URL = (
    "https://raw.githubusercontent.com/martj42/international_results/master/results.csv"
)


def refresh_results():
    """Re-download the results dataset (new matches are added daily during tournaments)."""
    DATA_DIR.mkdir(exist_ok=True)
    urllib.request.urlretrieve(RESULTS_URL, RESULTS_PATH)


def load_results():
    """Return (played, upcoming) DataFrames.

    played   — all matches with a final score, sorted by date.
    upcoming — scheduled matches with no score yet (e.g. WC 2026 fixtures).
    """
    if not RESULTS_PATH.exists():
        refresh_results()
    df = pd.read_csv(RESULTS_PATH, parse_dates=["date"])
    df["neutral"] = df["neutral"].astype(bool)
    df = df.sort_values("date").reset_index(drop=True)

    played = df.dropna(subset=["home_score", "away_score"]).copy()
    played["home_score"] = played["home_score"].astype(int)
    played["away_score"] = played["away_score"].astype(int)

    upcoming = df[df["home_score"].isna()].copy()
    return played, upcoming
