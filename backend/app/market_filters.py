from __future__ import annotations

from typing import Iterable


_SPORTS_TOKENS = {
    "nba",
    "nfl",
    "mlb",
    "nhl",
    "wnba",
    "ncaa",
    "ncaab",
    "ncaaf",
    "soccer",
    "football",
    "basketball",
    "baseball",
    "hockey",
    "tennis",
    "atp",
    "wta",
    "golf",
    "pga",
    "ufc",
    "mma",
    "boxing",
    "f1",
    "formula",
    "nascar",
    "motogp",
    "cricket",
    "rugby",
    "mls",
    "epl",
    "uefa",
    "fifa",
    "olympics",
    "olympic",
    "lol",
    "cs2",
    "dota",
    "valorant",
    "overwatch",
    "esports",
    "lcs",
    "lec",
    "lpl",
    "lck",
    "vct",
    "cod",
    "apex",
    "fortnite",
    "rlcs",
}

_SPORTS_SUBSTRINGS = (
    "super-bowl",
    "world-cup",
    "champions-league",
    "europa-league",
    "premier-league",
    "rocket-league",
    "call-of-duty",
    "rainbow-six",
    "league-of-legends",
    "la-liga",
    "bundesliga",
    "serie-a",
    "ligue-1",
    "uefa-euro",
    "march-madness",
)


def _tokenize_slug(value: str) -> list[str]:
    return [t for t in value.lower().split("-") if t]


def _has_sports_token(tokens: Iterable[str]) -> bool:
    for token in tokens:
        if token in _SPORTS_TOKENS:
            return True
    return False


def is_sports_market(
    *,
    event_slug: str | None = None,
    slug: str | None = None,
    title: str | None = None,
) -> bool:
    """
    Heuristic filter to exclude sports markets.
    Uses event_slug/slug tokens first, then title fallback.
    """
    for raw in (event_slug, slug):
        if not raw:
            continue
        lowered = raw.lower()
        if any(s in lowered for s in _SPORTS_SUBSTRINGS):
            return True
        if _has_sports_token(_tokenize_slug(lowered)):
            return True

    if title:
        lowered = title.lower()
        if any(s in lowered for s in _SPORTS_SUBSTRINGS):
            return True
        tokens = [t.strip(".,:;!?()[]{}") for t in lowered.split()]
        if _has_sports_token(tokens):
            return True

    return False
