import pytest

from segfix import workspace


def test_create_workspace_copies_file_and_writes_manifest(tmp_path):
    source = tmp_path / "source" / "cloud.ply"
    source.parent.mkdir()
    source.write_bytes(b"fake ply data")

    ws_dir = tmp_path / "MyProject"
    data_path = workspace.create_workspace(source, ws_dir)

    assert data_path == ws_dir / "cloud.ply"
    assert data_path.read_bytes() == b"fake ply data"
    assert workspace.is_workspace(ws_dir)
    assert workspace.data_file(ws_dir) == data_path
    assert workspace.source_file(ws_dir) == str(source.resolve())


def test_create_workspace_is_independent_copy(tmp_path):
    source = tmp_path / "cloud.ply"
    source.write_bytes(b"original")
    ws_dir = tmp_path / "proj"

    data_path = workspace.create_workspace(source, ws_dir)
    data_path.write_bytes(b"edited")

    # The copy changed; the source did not.
    assert source.read_bytes() == b"original"
    assert data_path.read_bytes() == b"edited"


def test_create_workspace_refuses_nonempty_existing_dir(tmp_path):
    source = tmp_path / "cloud.ply"
    source.write_bytes(b"data")
    ws_dir = tmp_path / "proj"
    ws_dir.mkdir()
    (ws_dir / "something.txt").write_text("already here")

    with pytest.raises(FileExistsError):
        workspace.create_workspace(source, ws_dir)


def test_is_workspace_false_for_plain_directory(tmp_path):
    plain = tmp_path / "just_a_dir"
    plain.mkdir()
    assert not workspace.is_workspace(plain)
