import argparse
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np


@dataclass(frozen=True)
class ParquetSummary:
    path: Path
    num_rows: int
    num_columns: int
    num_row_groups: int
    columns: list[str]
    schema: str
    preview_rows: list[dict[str, Any]]


def _load_pyarrow_parquet():
    try:
        import pyarrow.parquet as parquet
    except ModuleNotFoundError as exc:
        raise ModuleNotFoundError(
            "Reading parquet files requires pyarrow. Install it in the soma-retargeter uv "
            "environment with: uv add --project soma-retargeter pyarrow"
        ) from exc
    return parquet


def summarize_parquet(
    path: Path | str,
    *,
    row_limit: int = 5,
    parquet=None,
) -> ParquetSummary:
    parquet_path = Path(path)
    if not parquet_path.exists():
        raise FileNotFoundError(parquet_path)
    if row_limit < 0:
        raise ValueError("row_limit must be non-negative")

    parquet_reader = parquet if parquet is not None else _load_pyarrow_parquet()
    parquet_file = parquet_reader.ParquetFile(parquet_path)
    table = parquet_file.read()
    preview_rows = table.slice(0, row_limit).to_pylist() if row_limit else []

    return ParquetSummary(
        path=parquet_path,
        num_rows=table.num_rows,
        num_columns=table.num_columns,
        num_row_groups=parquet_file.num_row_groups,
        columns=list(table.column_names),
        schema=str(table.schema),
        preview_rows=preview_rows,
    )


def _table_to_numpy_dict(table) -> dict[str, np.ndarray]:
    episode: dict[str, np.ndarray] = {}
    for column_name in table.column_names:
        column = table[column_name].combine_chunks()
        if "list" in str(column.type):
            values = column.to_pylist()
            episode[column_name] = np.asarray(values, dtype=np.float32)
        elif "float" in str(column.type):
            episode[column_name] = column.to_numpy(zero_copy_only=False).astype(np.float32)
        elif "int" in str(column.type):
            episode[column_name] = column.to_numpy(zero_copy_only=False).astype(np.int64)
        else:
            episode[column_name] = np.asarray(column.to_pylist())
    return episode


def _estimate_fps(timestamps: np.ndarray) -> np.ndarray | None:
    if timestamps.shape[0] < 2:
        return None
    diffs = np.diff(timestamps.astype(np.float64))
    positive_diffs = diffs[diffs > 0]
    if positive_diffs.size == 0:
        return None
    return np.asarray(1.0 / np.median(positive_diffs), dtype=np.float32)


def load_episode_parquet(
    path: Path | str,
    *,
    parquet=None,
) -> dict[str, np.ndarray]:
    parquet_path = Path(path)
    if not parquet_path.exists():
        raise FileNotFoundError(parquet_path)

    parquet_reader = parquet if parquet is not None else _load_pyarrow_parquet()
    table = parquet_reader.read_table(parquet_path)
    episode = _table_to_numpy_dict(table)
    episode["num_frames"] = np.asarray(table.num_rows, dtype=np.int64)

    timestamps = episode.get("timestamp")
    if timestamps is not None:
        fps = _estimate_fps(timestamps)
        if fps is not None:
            episode["fps"] = fps

    return episode


def save_episode_npz(path: Path | str, output_path: Path | str) -> dict[str, np.ndarray]:
    episode = load_episode_parquet(path)
    np.savez(output_path, **episode)
    return episode


def _json_default(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if hasattr(value, "as_py"):
        return value.as_py()
    return str(value)


def print_summary(summary: ParquetSummary, *, as_json_output: bool) -> None:
    if as_json_output:
        print(json.dumps(asdict(summary), ensure_ascii=False, indent=2, default=_json_default))
        return

    print(f"File: {summary.path}")
    print(f"Rows: {summary.num_rows}")
    print(f"Columns: {summary.num_columns}")
    print(f"Row groups: {summary.num_row_groups}")
    print("Column names:")
    for column in summary.columns:
        print(f"  - {column}")
    print("Schema:")
    print(summary.schema)
    if summary.preview_rows:
        print("Preview rows:")
        for idx, row in enumerate(summary.preview_rows):
            print(f"  [{idx}] {row}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Inspect an episode parquet file.")
    parser.add_argument(
        "parquet_path",
        nargs="?",
        type=Path,
        default=Path("../episode_000000.parquet"),
        help="Path to the parquet file. Defaults to ../episode_000000.parquet from soma-retargeter.",
    )
    parser.add_argument("--rows", type=int, default=5, help="Number of preview rows to print.")
    parser.add_argument("--json", action="store_true", help="Print machine-readable JSON.")
    parser.add_argument(
        "--output-npz",
        type=Path,
        default=None,
        help="Optional path for saving the parsed episode as a NumPy .npz file.",
    )
    args = parser.parse_args()

    try:
        summary = summarize_parquet(args.parquet_path, row_limit=args.rows)
        if args.output_npz is not None:
            episode = save_episode_npz(args.parquet_path, args.output_npz)
    except (FileNotFoundError, ModuleNotFoundError, ValueError) as exc:
        parser.exit(1, f"error: {exc}\n")
    print_summary(summary, as_json_output=args.json)
    if args.output_npz is not None:
        print(f"Saved NPZ: {args.output_npz}")
        print("NPZ arrays:")
        for key, value in episode.items():
            print(f"  - {key}: shape={value.shape}, dtype={value.dtype}")


if __name__ == "__main__":
    main()
