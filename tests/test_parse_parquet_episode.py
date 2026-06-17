from pathlib import Path

import pytest

from app.parse_parquet_episode import load_episode_parquet, summarize_parquet


class FakeColumn:
    def __init__(self, values, column_type):
        self._values = values
        self.type = column_type

    def combine_chunks(self):
        return self

    def to_pylist(self):
        return self._values

    def to_numpy(self, zero_copy_only=False):
        import numpy as np

        return np.asarray(self._values)


class FakeTable:
    column_names = ["frame", "root_pos", "joint_pos"]
    num_rows = 2
    num_columns = 3

    @property
    def schema(self):
        return "frame: int64\nroot_pos: list<float>\njoint_pos: list<float>"

    def slice(self, offset, length):
        assert offset == 0
        assert length == 1
        return self

    def to_pylist(self):
        return [{"frame": 0, "root_pos": [0.0, 0.0, 0.5], "joint_pos": [0.1, 0.2]}]

    def __getitem__(self, column_name):
        columns = {
            "frame": FakeColumn([0, 1], "int64"),
            "root_pos": FakeColumn([[0.0, 0.0, 0.5], [0.1, 0.0, 0.5]], "fixed_size_list<element: float>[3]"),
            "joint_pos": FakeColumn([[0.1, 0.2], [0.3, 0.4]], "fixed_size_list<element: float>[2]"),
        }
        return columns[column_name]


class FakeParquetFile:
    def __init__(self, path):
        self.path = Path(path)
        self.num_row_groups = 1

    def read(self):
        return FakeTable()


class FakeParquetModule:
    ParquetFile = FakeParquetFile

    @staticmethod
    def read_table(path):
        return FakeTable()


def test_summarize_parquet_reports_schema_shape_and_preview(tmp_path: Path):
    path = tmp_path / "episode.parquet"
    path.write_bytes(b"PAR1")

    summary = summarize_parquet(path, row_limit=1, parquet=FakeParquetModule)

    assert summary.path == path
    assert summary.num_rows == 2
    assert summary.num_columns == 3
    assert summary.num_row_groups == 1
    assert summary.columns == ["frame", "root_pos", "joint_pos"]
    assert summary.preview_rows == [{"frame": 0, "root_pos": [0.0, 0.0, 0.5], "joint_pos": [0.1, 0.2]}]
    assert "root_pos" in summary.schema


def test_summarize_parquet_rejects_missing_files(tmp_path: Path):
    with pytest.raises(FileNotFoundError):
        summarize_parquet(tmp_path / "missing.parquet", parquet=FakeParquetModule)


def test_load_episode_parquet_returns_numpy_arrays(tmp_path: Path):
    path = tmp_path / "episode.parquet"
    path.write_bytes(b"PAR1")

    episode = load_episode_parquet(path, parquet=FakeParquetModule)

    assert episode["frame"].dtype.name == "int64"
    assert episode["root_pos"].dtype.name == "float32"
    assert episode["root_pos"].shape == (2, 3)
    assert episode["joint_pos"].shape == (2, 2)
    assert episode["num_frames"].item() == 2
