"""dedupe.py: the normalized title+artist key two editions of one recording share."""
from music_recommendations.server.dedupe import (
    dedupe_key, normalize_artist, normalize_title,
)


def _key(title, artist="Miles Davis"):
    return dedupe_key(title, artist)


# ---- normalize_title ----

def test_feature_credit_is_dropped():
    assert normalize_title("So What (feat. John Coltrane & Bill Evans)") == "so what"


def test_remaster_with_year_matches_the_plain_title():
    assert _key("Mood Indigo (1990 Remaster)") == _key("Mood Indigo")


def test_trailing_dash_live_segment_matches_the_plain_title():
    assert _key("Take Five - Live at Carnegie Hall") == _key("Take Five")


def test_with_credit_is_dropped():
    assert _key("Miracle (with Ellie Goulding)") == _key("Miracle")


def test_radio_edit_matches_the_plain_title():
    assert _key("Around the World (Radio Edit)") == _key("Around the World")


def test_take_number_and_case_do_not_matter():
    assert _key("Blue in Green") == _key("Blue In Green (Take 2)")


def test_trailing_remastered_segment_is_dropped():
    assert _key("Freddie Freeloader - Remastered") == _key("Freddie Freeloader")


def test_several_trailing_segments_are_dropped():
    assert _key("So What - Live - 2001 Remaster") == _key("So What")


def test_a_meaningful_parenthetical_survives():
    assert normalize_title("Wish You Were Here (Reprise)") == "wish you were here reprise"


def test_punctuation_and_accents_collapse():
    assert normalize_title("Chega de Saudade") == "chega de saudade"
    assert normalize_title("Björk's  Song!") == "bjork s song"


# ---- normalize_artist ----

def test_primary_artist_only():
    assert normalize_artist("Miles Davis feat. John Coltrane") == "miles davis"
    assert normalize_artist("Simon & Garfunkel") == "simon"
    assert normalize_artist("Duke Ellington, Count Basie") == "duke ellington"
    assert normalize_artist("Calvin Harris featuring Rihanna") == "calvin harris"


# ---- dedupe_key ----

def test_same_title_by_different_artists_is_a_different_key():
    assert dedupe_key("So What", "Miles Davis") != dedupe_key("So What", "Ron Carter")


def test_a_cover_is_not_collapsed_into_the_original():
    original = dedupe_key("Autumn Leaves", "Bill Evans")
    cover = dedupe_key("Autumn Leaves (Live)", "Cannonball Adderley")
    assert original != cover


def test_key_joins_artist_and_title_with_a_pipe():
    assert dedupe_key("So What", "Miles Davis") == "miles davis|so what"


def test_missing_values_do_not_raise():
    assert dedupe_key(None, None) == "|"
