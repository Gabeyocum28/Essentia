"""The licence register must cover everything we ship, and stay sellable.

Two failures this catches, both of which are expensive to find any later
way:

  1. a dependency added to pyproject.toml with no row in
     docs/THIRD_PARTY.md -- i.e. nobody checked its licence;
  2. a non-commercial component creeping back in. The whole clean-room
     exercise exists because Discogs-EffNet is CC BY-NC-SA; re-acquiring
     that problem by accident would invalidate the work.

The one NC row still in the register (the v1 EffNet model) is named here
explicitly. That is the point: it is a tracked, scheduled removal rather
than something the check quietly tolerates, and when the cutover deletes it
this list goes empty and stays that way.
"""
from __future__ import annotations

import re
import tomllib
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PYPROJECT = ROOT / "pyproject.toml"
REGISTER = ROOT / "docs" / "THIRD_PARTY.md"

# Rows allowed to carry a non-commercial licence, with the reason. Each is a
# component we have committed to removing; nothing may be added here without
# the same commitment.
NC_ALLOWED = {
    "discogs-effnet": "v1 embedding + feel heads; deleted at the cutover",
}

NC_PATTERN = re.compile(r"\bNC\b|NonCommercial|Non-Commercial", re.IGNORECASE)


def _rows() -> list[tuple[str, str]]:
    """(name cell, licence cell) for every table row in the register."""
    rows = []
    for line in REGISTER.read_text().splitlines():
        line = line.strip()
        if not line.startswith("|") or line.startswith("|---"):
            continue
        cells = [c.strip() for c in line.strip("|").split("|")]
        if len(cells) < 2 or cells[0] in ("Package", "Model", "Component"):
            continue
        rows.append((cells[0], cells[1]))
    return rows


def _declared_dependencies() -> set[str]:
    """Every distribution named in pyproject.toml, extras included.

    Normalized the way pip does (PEP 503: lowercase, - and _ and . all the
    same character), so `beat_this` in the extra matches a `beat-this` row.
    """
    data = tomllib.loads(PYPROJECT.read_text())
    project = data["project"]
    specs = list(project.get("dependencies", []))
    for extra in project.get("optional-dependencies", {}).values():
        specs += list(extra)
    names = set()
    for spec in specs:
        # "beat_this @ git+https://..." / "torch>=2.2" / "numpy>=1.26"
        head = spec.split("@")[0]
        name = re.split(r"[<>=!~\[; ]", head.strip())[0]
        if name:
            names.add(_normalize(name))
    return names


def _normalize(name: str) -> str:
    return re.sub(r"[-_.]+", "-", name).strip().lower()


def test_every_declared_dependency_has_a_row():
    registered = {_normalize(_strip_markup(name)) for name, _lic in _rows()}
    missing = sorted(_declared_dependencies() - registered)
    assert not missing, (
        f"add a row to docs/THIRD_PARTY.md for: {', '.join(missing)}"
    )


def _strip_markup(cell: str) -> str:
    """`**torch**` / `torch` (link) -> torch."""
    cell = re.sub(r"\[([^\]]+)\]\([^)]*\)", r"\1", cell)
    return cell.replace("*", "").replace("`", "").strip()


def test_every_row_states_a_licence():
    blank = [name for name, lic in _rows() if not _strip_markup(lic)]
    assert not blank, f"rows with no licence: {blank}"


def test_no_unscheduled_non_commercial_component():
    offenders = []
    for name, licence in _rows():
        if not NC_PATTERN.search(licence):
            continue
        plain = _strip_markup(name).lower()
        if not any(key in plain for key in NC_ALLOWED):
            offenders.append(f"{plain}: {licence}")
    assert not offenders, (
        "non-commercial components are what this project is removing; "
        f"found: {offenders}"
    )


def test_the_only_tolerated_nc_component_is_the_v1_model():
    """Shrinking NC_ALLOWED is the deliverable. If the cutover has landed and
    this still lists EffNet, the register was not updated with the code."""
    assert set(NC_ALLOWED) == {"discogs-effnet"}
    nc_rows = [_strip_markup(n).lower() for n, lic in _rows()
               if NC_PATTERN.search(lic)]
    assert len(nc_rows) == 1, nc_rows
    assert "discogs-effnet" in nc_rows[0]


def test_the_analysis_extra_is_the_v2_stack():
    """A guard on the thing the register is about: CLAP and Beat This! are
    installed, and TensorFlow is present only as the tracked leftover."""
    data = tomllib.loads(PYPROJECT.read_text())
    extra = {
        _normalize(re.split(r"[<>=!~\[; @]", s.strip())[0])
        for s in data["project"]["optional-dependencies"]["analysis"]
    }
    assert {"torch", "msclap", "librosa", "pyloudnorm", "beat-this"} <= extra
