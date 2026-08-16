from pathlib import Path

from ocean.osrs.scripts import osrs_asset_manifest


def synthetic_manifest(*groups: tuple[str, tuple[str, ...]]) -> osrs_asset_manifest.AssetManifest:
    return osrs_asset_manifest.AssetManifest(
        format=osrs_asset_manifest.EXPECTED_FORMAT,
        asset_version="test",
        archive=osrs_asset_manifest.AssetArchive(
            name="test.tar.gz",
            url="https://example.com/test.tar.gz",
            sha256="0" * 64,
            strip_components=0,
        ),
        required_groups=tuple(
            osrs_asset_manifest.RequiredGroup(name=name, files=files)
            for name, files in groups
        ),
    )


def test_selected_groups_emit_ordered_unique_full_vfs_paths() -> None:
    manifest = synthetic_manifest(
        ("core", ("shared.bin", "core.bin")),
        ("inferno", ("shared.bin", "inferno.bin")),
    )

    assert osrs_asset_manifest.emcc_preload_args(manifest, ["core", "inferno"]) == [
        "--preload-file ocean/osrs/data/shared.bin@ocean/osrs/data/shared.bin",
        "--preload-file ocean/osrs/data/core.bin@ocean/osrs/data/core.bin",
        "--preload-file ocean/osrs/data/inferno.bin@ocean/osrs/data/inferno.bin",
    ]


def test_group_argument_order_does_not_change_manifest_order() -> None:
    manifest = synthetic_manifest(
        ("core", ("core.bin",)),
        ("inferno", ("inferno.bin",)),
    )

    forward = osrs_asset_manifest.emcc_preload_args(manifest, ["core", "inferno"])
    reverse = osrs_asset_manifest.emcc_preload_args(manifest, ["inferno", "core"])

    assert reverse == forward
    assert forward == [
        "--preload-file ocean/osrs/data/core.bin@ocean/osrs/data/core.bin",
        "--preload-file ocean/osrs/data/inferno.bin@ocean/osrs/data/inferno.bin",
    ]


def test_empty_group_selection_emits_every_asset_once() -> None:
    manifest = synthetic_manifest(
        ("core", ("shared.bin", "core.bin")),
        ("inferno", ("shared.bin", "inferno.bin")),
        ("gui", ("gui.bin",)),
    )

    assert osrs_asset_manifest.emcc_preload_args(manifest, []) == [
        "--preload-file ocean/osrs/data/shared.bin@ocean/osrs/data/shared.bin",
        "--preload-file ocean/osrs/data/core.bin@ocean/osrs/data/core.bin",
        "--preload-file ocean/osrs/data/inferno.bin@ocean/osrs/data/inferno.bin",
        "--preload-file ocean/osrs/data/gui.bin@ocean/osrs/data/gui.bin",
    ]


def test_inferno_web_groups_emit_43_unique_matching_vfs_paths() -> None:
    root = Path(__file__).resolve().parents[1]
    manifest = osrs_asset_manifest.load_manifest(root / "ocean/osrs/asset_manifest.json")
    lines = osrs_asset_manifest.emcc_preload_args(
        manifest,
        ["core", "inferno", "combat_visuals", "gui", "items"],
    )

    assert len(lines) == 43
    assert len(set(lines)) == 43
    for line in lines:
        source, destination = line.removeprefix("--preload-file ").split("@", 1)
        assert source == destination
        assert source.startswith("ocean/osrs/data/")
