"""Tests for apply_tmt_itraq_from_search_data helpers."""

from __future__ import annotations

from scripts.verification.apply_tmt_itraq_from_search_data import (
    TagKind,
    modifications_match_itraq_4plex,
    parse_spec,
)


def test_modifications_match_itraq_4plex_requires_literal_4() -> None:
    """Only iTRAQ 4-plex text matches; 8-plex does not."""
    assert modifications_match_itraq_4plex("iTRAQ 4-plex") is True
    assert modifications_match_itraq_4plex("iTRAQ 8-plex") is False


def test_parse_spec_project_and_kind() -> None:
    """PROJECT:TAG_KIND specs parse into project id and TagKind."""
    assert parse_spec("PXD123:TMT_6_8_10") == ("PXD123", TagKind.TMT_6_8_10)
