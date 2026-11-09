from __future__ import annotations

from instanovo_fm.data.metadata_builder import _VOCAB_LOOKUPS, FRAG_TYPES, _encode_categorical, get_vocabulary_sizes


def test_existing_frag_type_indices_are_unchanged() -> None:
    """New fragmentation types are appended, so data encoded before keeps its indices."""
    assert FRAG_TYPES[:4] == ["HCD", "HCID", "CID", "ETD"]


def test_ethcd_and_etcid_have_their_own_index() -> None:
    """EThcD and ETciD encode as themselves, not as the unknown index 0 (HCD)."""
    encoded = _encode_categorical(["EThcD", "ETciD", "ethcd", "HCD", "nonsense", None], _VOCAB_LOOKUPS["FRAG_TYPES"]).tolist()
    assert encoded == [4, 5, 4, 0, 0, 0]
    assert get_vocabulary_sizes()["n_frag"] == 6
