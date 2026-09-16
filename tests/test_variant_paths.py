"""PlantVillage variant path resolution (used by the ``bgremoval`` regime).

``bgremoval`` trains on ``segmented/`` while the splits are defined against
``color/``. Filenames are not identical between the two variants, so the mapping
has to be by normalised key rather than by name. Getting this wrong silently
shrinks the training set, which would make bgremoval incomparable to the
regimes it is supposed to be measured against - the comparison would look valid
and be wrong.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from train.train import build_variant_index, resolve_paths, variant_key


@pytest.mark.parametrize(
    ("name", "expected"),
    [
        # plain colour name
        ("0055dd26-23a7-4415-ac61-e0b44ebfaf80___RS_HL 5672.JPG", "rs_hl 5672"),
        # segmented counterpart: _final_masked suffix, lower-case extension
        ("0055dd26-23a7-4415-ac61-e0b44ebfaf80___RS_HL 5672_final_masked.jpg", "rs_hl 5672"),
        # ~2% of colour files carry no UUID prefix at all
        ("RS_Rust 1818.JPG", "rs_rust 1818"),
        # ...while their segmented counterpart does
        ("17d09699-bebb-42c2-88be-a5696ae6aade___RS_Rust 1818_final_masked.jpg", "rs_rust 1818"),
    ],
)
def test_variant_key_normalisation(name, expected):
    assert variant_key(name) == expected


def test_uuid_prefixed_and_bare_names_collide_by_design():
    """The bare and UUID-prefixed forms of one capture must map to one key."""
    bare = "RS_Rust 1818.JPG"
    prefixed = "17d09699-bebb-42c2-88be-a5696ae6aade___RS_Rust 1818_final_masked.jpg"
    assert variant_key(bare) == variant_key(prefixed)


def test_build_variant_index(tmp_path):
    d = tmp_path / "segmented" / "Corn___rust"
    d.mkdir(parents=True)
    (d / "abc___RS_Rust 1818_final_masked.jpg").write_bytes(b"x")
    (d / "def___RS_Rust 1933_final_masked.jpg").write_bytes(b"x")

    index = build_variant_index(d)
    assert set(index) == {"rs_rust 1818", "rs_rust 1933"}


def test_resolve_paths_maps_across_variants(tmp_path):
    color = tmp_path / "color" / "Corn___rust"
    seg = tmp_path / "segmented" / "Corn___rust"
    color.mkdir(parents=True)
    seg.mkdir(parents=True)
    (color / "RS_Rust 1818.JPG").write_bytes(b"x")
    (seg / "17d09699___RS_Rust 1818_final_masked.jpg").write_bytes(b"x")

    out = resolve_paths([color / "RS_Rust 1818.JPG"], {"image_variant": "segmented"})
    assert out == [seg / "17d09699___RS_Rust 1818_final_masked.jpg"]


def test_resolve_paths_is_identity_for_color(tmp_path):
    paths = [Path("data/raw/plantvillage/color/A/x.JPG")]
    assert resolve_paths(paths, {"image_variant": "color"}) == paths
    assert resolve_paths(paths, {}) == paths


def test_resolve_paths_raises_rather_than_dropping(tmp_path):
    """A missing counterpart must fail the run, not silently shrink the set."""
    color = tmp_path / "color" / "Corn___rust"
    (tmp_path / "segmented" / "Corn___rust").mkdir(parents=True)
    color.mkdir(parents=True)
    (color / "RS_Rust 9999.JPG").write_bytes(b"x")

    with pytest.raises(FileNotFoundError, match="have no counterpart"):
        resolve_paths([color / "RS_Rust 9999.JPG"], {"image_variant": "segmented"})
