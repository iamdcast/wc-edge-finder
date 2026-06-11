"""Elo rating engine for national teams.

Follows the World Football Elo Ratings methodology (eloratings.net):
  - K factor scaled by match importance (World Cup 60 ... friendly 20)
  - K multiplied by a goal-margin factor
  - +100 rating points of home advantage for non-neutral venues
"""
from collections import defaultdict

BASE_RATING = 1500.0
HOME_ADV = 100.0  # Elo points

_CONTINENTAL_FINALS = (
    "uefa euro",
    "copa américa",
    "copa america",
    "african cup of nations",
    "africa cup of nations",
    "afc asian cup",
    "gold cup",
    "concacaf championship",
    "oceania nations cup",
    "ofc nations cup",
    "confederations cup",
)


def k_factor(tournament):
    t = tournament.lower()
    if t == "friendly":
        return 20.0
    if "qualification" in t or "nations league" in t:
        return 40.0
    if t == "fifa world cup":
        return 60.0
    if any(name in t for name in _CONTINENTAL_FINALS):
        return 50.0
    return 30.0


def margin_multiplier(goal_diff):
    gd = abs(goal_diff)
    if gd <= 1:
        return 1.0
    if gd == 2:
        return 1.5
    return (11.0 + gd) / 8.0


class EloRatings:
    """Chronological Elo fit over a played-matches DataFrame.

    Also records the pre-match ratings of every game (training data for the
    goal model) and per-team rating timelines (for trend displays).
    """

    def __init__(self):
        self.ratings = {}
        self.history = []  # one dict per match, pre-match ratings included
        self.timelines = defaultdict(list)  # team -> [(date, rating_after)]

    def get(self, team):
        return self.ratings.get(team, BASE_RATING)

    def expected(self, rating_home_adj, rating_away):
        dr = rating_home_adj - rating_away
        return 1.0 / (1.0 + 10.0 ** (-dr / 400.0))

    def fit(self, played, record_history=True):
        """Replay matches chronologically, updating ratings."""
        for row in played.itertuples(index=False):
            self.update(
                row.date,
                row.home_team,
                row.away_team,
                row.home_score,
                row.away_score,
                row.tournament,
                row.neutral,
                record_history=record_history,
            )
        return self

    def update(self, date, home, away, hs, as_, tournament, neutral,
               record_history=True):
        rh, ra = self.get(home), self.get(away)
        rh_adj = rh + (0.0 if neutral else HOME_ADV)
        we = self.expected(rh_adj, ra)

        if record_history:
            self.history.append(
                dict(date=date, home_team=home, away_team=away,
                     home_score=hs, away_score=as_, tournament=tournament,
                     neutral=neutral, elo_home=rh, elo_away=ra)
            )

        w = 1.0 if hs > as_ else (0.5 if hs == as_ else 0.0)
        delta = k_factor(tournament) * margin_multiplier(hs - as_) * (w - we)
        self.ratings[home] = rh + delta
        self.ratings[away] = ra - delta
        self.timelines[home].append((date, self.ratings[home]))
        self.timelines[away].append((date, self.ratings[away]))

    def rating_at(self, team, date):
        """Rating a team held just before `date` (BASE_RATING if no history)."""
        rating = BASE_RATING
        for d, r in self.timelines.get(team, []):
            if d >= date:
                break
            rating = r
        return rating

    def top(self, n=50):
        return sorted(self.ratings.items(), key=lambda kv: -kv[1])[:n]
