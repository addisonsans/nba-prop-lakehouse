import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "databricks" / "notebooks"))

from name_match import match_player, norm  # noqa: E402


def ids(names, offset=0):
    return {norm(n): offset + i for i, n in enumerate(names)}


def resolve(name, box, roster):
    box_ids, roster_ids = ids(box), ids(roster, offset=100)
    by_id = {**{v: k for k, v in roster_ids.items()}, **{v: k for k, v in box_ids.items()}}
    pid, method = match_player(name, box_ids, roster_ids)
    return by_id.get(pid), method


HOU = ["Jalen Green", "Fred VanVleet", "Alperen Sengun", "Jeff Green", "Dillon Brooks"]


@pytest.mark.parametrize("name, box, roster, expected", [
    # A player who sat out is only on the roster; never fall back to a same-surname teammate.
    ("Jalen Green", ["Fred VanVleet", "Jeff Green"], HOU, "jalen green"),
    ("Dorian Finney-Smith", ["Dennis Smith Jr.", "Dru Smith"], ["Dennis Smith Jr.", "Dru Smith", "Dorian Finney-Smith"], "dorian finney smith"),
    ("G.G. Jackson", ["Jaren Jackson Jr."], ["Jaren Jackson Jr.", "GG Jackson"], "gg jackson"),
    ("Keyonte George", ["Kyshawn George"], ["Kyshawn George", "Keyonte George"], "keyonte george"),
    # Formatting differences
    ("Josh Green", ["Kyrie Irving", "Josh Green"], ["Kyrie Irving", "Josh Green"], "josh green"),
    ("Portis, Bobby", ["Bobby Portis"], ["Bobby Portis"], "bobby portis"),
    ("Wembanyama Victor", ["Victor Wembanyama"], ["Victor Wembanyama"], "victor wembanyama"),
    ("Brook Robert Lopez", ["Brook Lopez", "Robin Lopez"], ["Brook Lopez", "Robin Lopez"], "brook lopez"),
    ("Alfred Joel Horford Reynoso", ["Al Horford", "Jayson Tatum"], ["Al Horford", "Jayson Tatum"], "al horford"),
    ("Obadiah Toppin", ["Obi Toppin"], ["Obi Toppin"], "obi toppin"),
    # Aliases
    ("Carlton Carrington", ["Bub Carrington"], ["Bub Carrington"], "bub carrington"),
    ("Moe Wagner", ["Franz Wagner", "Moritz Wagner"], ["Franz Wagner", "Moritz Wagner"], "moritz wagner"),
    ("Gregory Jackson II", ["Jaren Jackson Jr."], ["Jaren Jackson Jr.", "GG Jackson"], "gg jackson"),
    ("Christian James McCollum", ["CJ McCollum"], ["CJ McCollum"], "cj mccollum"),
    # Ambiguous or not a player: leave unmatched rather than guess
    ("J. Green", ["Jalen Green", "Jeff Green"], HOU, None),
    ("J. Green", ["AJ Green", "Giannis Antetokounmpo"], ["AJ Green", "Giannis Antetokounmpo"], None),
    ("D. Robinson", ["Duncan Robinson", "Bam Adebayo"], ["Duncan Robinson", "Bam Adebayo"], "duncan robinson"),
    ("Bkn Nets Alternate Total", ["Mikal Bridges"], ["Mikal Bridges"], None),
])
def test_match_player(name, box, roster, expected):
    assert resolve(name, box, roster)[0] == expected


def test_norm_strips_accents_suffixes_and_punctuation():
    assert norm("Luka Dončić") == "luka doncic"
    assert norm("Jaren Jackson Jr.") == "jaren jackson"
    assert norm("De'Aaron Fox") == "deaaron fox"
    assert norm("Jalen (2001) Johnson") == "jalen johnson"
