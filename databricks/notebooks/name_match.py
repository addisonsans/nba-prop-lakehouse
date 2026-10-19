"""Match sportsbook player names to NBA.com player IDs.

Used by the silver notebook (`from name_match import ...`; Databricks puts the notebook's folder
on sys.path) and unit-tested in tests/test_name_match.py.
"""

from __future__ import annotations

import difflib
import re
import unicodedata

# Sportsbook names that no rule can resolve (two Wagners play for Orlando).
ALIASES = {
    "moe wagner": "moritz wagner",            # Franz is on the same team
    "carlton carrington": "bub carrington",
    "nahshon hyland": "bones hyland",
    "edrice femi adebayo": "bam adebayo",
    "edrice adebayo": "bam adebayo",
    "aleksandar vezenkov": "sasha vezenkov",
    "gregory jackson": "gg jackson",
    "ishmael larry smith": "ish smith",
    "ishmael smith": "ish smith",
    "kenyon martin": "kj martin",
    "martin kenyon": "kj martin",
    "trayce jackson": "trayce jackson davis",
    "christian james mccollum": "cj mccollum",
}


def norm(name: str) -> str:
    n = unicodedata.normalize("NFKD", name).encode("ascii", "ignore").decode().lower()
    n = re.sub(r"\(.*?\)", " ", n)  # "Jalen (2001) Johnson"
    if "," in n:  # "Portis, Bobby"
        last, _, first = n.partition(",")
        n = f"{first} {last}"
    n = re.sub(r"[.'’`]", "", n).replace("-", " ")
    n = re.sub(r"\b(jr|sr|ii|iii|iv|v)\b", "", n)
    n = re.sub(r"\s+", " ", n).strip()
    return ALIASES.get(n, n)


# Stands in for the player_id when two different players share a normalized name
# (Chris Johnson x2 in 2012-13, Tony Mitchell x2 in 2013-14).
AMBIGUOUS = -1


def name_index(pairs) -> dict[str, int]:
    """(name, player_id) pairs -> normalized name -> player_id, or AMBIGUOUS when a name is shared.

    A plain dict(zip(...)) would silently keep whichever same-name player came last.
    """
    index: dict[str, int] = {}
    for name, pid in pairs:
        if not name:
            continue
        key = norm(name)
        index[key] = pid if index.get(key, pid) == pid else AMBIGUOUS
    return index


def merge(*indexes: dict[str, int]) -> dict[str, int]:
    merged: dict[str, int] = {}
    for index in indexes:
        for key, pid in index.items():
            merged[key] = pid if merged.get(key, pid) == pid else AMBIGUOUS
    return merged


def strict_match(target: str, candidates: dict[str, int]) -> tuple[int | None, str | None]:
    """candidates: normalized name -> player_id."""
    if target in candidates:
        return candidates[target], "exact"
    first, _, last = target.partition(" ")
    nick = {pid for c, pid in candidates.items()
            if c.split(" ", 1)[-1] == last and (c.split(" ")[0].startswith(first) or first.startswith(c.split(" ")[0]))}
    if len(nick) == 1:
        return nick.pop(), "nickname"
    return None, None


def loose_match(target: str, candidates: dict[str, int]) -> tuple[int | None, str | None]:
    tokens = set(target.split())
    # Legal or reordered names: every word of the NBA.com name appears ("Brook Robert Lopez").
    subset = {pid for c, pid in candidates.items() if set(c.split()) <= tokens}
    if len(subset) == 1:
        return subset.pop(), "all_words"
    # An initial ("J. Green") only ever matches via the nickname rule: string similarity
    # would happily turn "j green" into "aj green".
    initial_only = len(target.split()[0]) == 1
    scored = sorted(((difflib.SequenceMatcher(None, target, c).ratio(), pid) for c, pid in candidates.items()), reverse=True)
    if not initial_only and scored and scored[0][0] >= 0.85 and (len(scored) == 1 or scored[1][0] < scored[0][0] - 0.05):
        return scored[0][1], "fuzzy"
    # One candidate's surname appears, and a first-name word shares its first initial
    # ("Alfred Joel Horford Reynoso" -> Al Horford, "Obadiah Toppin" -> Obi Toppin).
    hits = []
    for c, pid in candidates.items():
        first, _, last = c.partition(" ")
        if last and set(last.split()) <= tokens:
            given = tokens - set(last.split())
            if any(t[0] == first[0] for t in given):
                hits.append(pid)
    if len(set(hits)) == 1:
        return hits[0], "surname"
    return None, None


def match_player(name: str, box: dict[str, int], roster: dict[str, int]) -> tuple[int | None, str | None]:
    """Match one sportsbook name for one game.

    box: normalized name -> player_id for that game's box score.
    roster: normalized name -> player_id for both teams' season rosters.
    Strict rules run on the box score, then the roster (a player who sat out is only on the
    roster); loose rules run last on both rosters, so they can't pick a teammate of a player who
    simply sat out.
    """
    target = norm(name)
    pid, method = strict_match(target, box)
    if pid is None:
        pid, method = strict_match(target, roster)
        method = method and f"{method}_season_roster"
    if pid is None:
        pid, method = loose_match(target, merge(roster, box))
    if pid == AMBIGUOUS:
        return None, "ambiguous_same_name"
    return pid, method
