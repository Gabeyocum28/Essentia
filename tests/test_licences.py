"""The licence register must cover everything we ship, and stay sellable.

Two failures this catches, both of which are expensive to find any later
way:

  1. a dependency added to pyproject.toml with no row in
     docs/THIRD_PARTY.md -- i.e. nobody checked its licence;
  2. a non-commercial component creeping back in. The whole clean-room
     exercise exists because Discogs-EffNet is CC BY-NC-SA; re-acquiring
     that problem by accident would invalidate the work.

NC_ALLOWED is now EMPTY, and that is the deliverable: the last
non-commercial component (Discogs-EffNet) went at the cutover. Anything
added back to that set is a promise to remove it again, not a way to pass
this test.
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
NC_ALLOWED: dict[str, str] = {}

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


def test_nothing_non_commercial_is_tolerated_any_more():
    """Emptying NC_ALLOWED was the deliverable. The register now has no
    non-commercial row at all, and nothing is exempted from the check."""
    assert NC_ALLOWED == {}
    nc_rows = [_strip_markup(n) for n, lic in _rows()
               if NC_PATTERN.search(lic)]
    assert nc_rows == []


def test_the_analysis_extra_is_exactly_the_clean_room_stack():
    """A guard on the thing the register is about. TensorFlow in particular
    must stay out: it existed only to run the non-commercial model, and it is
    ~600 MB of image for a graph nothing loads any more."""
    data = tomllib.loads(PYPROJECT.read_text())
    extra = {
        _normalize(re.split(r"[<>=!~\[; @]", s.strip())[0])
        for s in data["project"]["optional-dependencies"]["analysis"]
    }
    assert extra == {"torch", "msclap", "librosa", "pyloudnorm", "soxr",
                     "beat-this"}


def test_tensorflow_is_gone_from_the_project():
    """Not only from the extra: from the lockfile and the Dockerfile too, so
    nothing quietly reinstalls it."""
    assert "tensorflow" not in PYPROJECT.read_text().lower()
    assert "tensorflow" not in (ROOT / "uv.lock").read_text().lower()
    assert "tensorflow" not in (ROOT / "deploy" / "Dockerfile").read_text().lower()
