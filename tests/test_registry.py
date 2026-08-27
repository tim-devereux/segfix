import json

import pytest

from segfix import registry


def test_add_entry_creates_file_and_records_type(tmp_path):
    target = tmp_path / "registry.json"
    a_file = tmp_path / "a.ply"
    a_file.touch()

    registry.add_entry(str(a_file), kind="file", registry_file=target)

    entries = registry.load_registry(target)
    assert len(entries) == 1
    assert entries[0]["path"] == str(a_file.resolve())
    assert entries[0]["type"] == "file"


def test_add_entry_rejects_unknown_kind(tmp_path):
    with pytest.raises(ValueError):
        registry.add_entry(
            str(tmp_path / "a.ply"), kind="bogus", registry_file=tmp_path / "r.json"
        )


def test_add_entry_bumps_existing_to_front(tmp_path):
    target = tmp_path / "registry.json"
    a = tmp_path / "a.ply"
    b = tmp_path / "b.ply"
    a.touch()
    b.touch()

    registry.add_entry(str(a), "file", registry_file=target)
    registry.add_entry(str(b), "file", registry_file=target)
    registry.add_entry(str(a), "file", registry_file=target)  # re-open a

    entries = registry.load_registry(target)
    assert [e["path"] for e in entries] == [str(a.resolve()), str(b.resolve())]


def test_load_registry_prunes_missing_paths(tmp_path):
    target = tmp_path / "registry.json"
    gone = tmp_path / "gone.ply"
    still_here = tmp_path / "here.ply"
    still_here.touch()

    # gone.ply is recorded but then deleted before the next load.
    registry.add_entry(str(gone), "file", registry_file=target)
    gone_resolved = str(gone.resolve())
    registry.add_entry(str(still_here), "file", registry_file=target)

    entries = registry.load_registry(target)
    assert gone_resolved not in {e["path"] for e in entries}
    assert str(still_here.resolve()) in {e["path"] for e in entries}

    # The pruned list should also have been persisted back to disk.
    on_disk = json.loads(target.read_text())["entries"]
    assert len(on_disk) == 1


def test_max_entries_trims_oldest(tmp_path, monkeypatch):
    target = tmp_path / "registry.json"
    monkeypatch.setattr(registry, "MAX_ENTRIES", 3)
    paths = []
    for i in range(5):
        p = tmp_path / f"{i}.ply"
        p.touch()
        paths.append(p)
        registry.add_entry(str(p), "file", registry_file=target)

    entries = registry.load_registry(target)
    assert len(entries) == 3
    # Most recent 3 (indices 4, 3, 2) survive, oldest (0, 1) are trimmed.
    assert [e["path"] for e in entries] == [
        str(paths[4].resolve()), str(paths[3].resolve()), str(paths[2].resolve()),
    ]


def test_corrupt_registry_file_returns_empty(tmp_path):
    target = tmp_path / "registry.json"
    target.write_text("not valid json{{{")
    assert registry.load_registry(target) == []


def test_workspace_kind_records_folder_not_data_file(tmp_path):
    target = tmp_path / "registry.json"
    workspace_dir = tmp_path / "MyProject"
    workspace_dir.mkdir()

    registry.add_entry(str(workspace_dir), kind="workspace", registry_file=target)

    entries = registry.load_registry(target)
    assert entries[0]["type"] == "workspace"
    assert entries[0]["path"] == str(workspace_dir.resolve())
