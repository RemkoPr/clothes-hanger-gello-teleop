#!/usr/bin/env python3
"""Plot creation chronology for parquet files in selected evaluation datasets.

The script scans each dataset's `data/chunk-000` directory and records the
filesystem creation timestamp (birth time) for each parquet file when available.
If birth time is unavailable, it falls back to the file modification time.
"""

from __future__ import annotations

import argparse
import subprocess
from collections import defaultdict
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

import matplotlib.pyplot as plt


DEFAULT_DATASETS = [
    "b-n200-INSTR0-300k-1enc-EVAL",
    "b-n200-INSTR0-300k-1enc-EVAL2",
    "b-n200-INSTR0-300k-1enc-EVAL3",
    "b-n200-INSTR1-300k-1enc-EVAL",
    "b-n200-INSTR1-300k-1enc-EVAL2",
    "b-n200-INSTR1-300k-1enc-EVAL3",
]
'''
DEFAULT_DATASETS = [
    "b-n150-INSTR0-200k-EVAL",
    "b-n150-INSTR0-250k-EVAL",
]
'''

@dataclass
class ParquetCreationEvent:
    dataset: str
    label: str
    parquet_file: Path
    created_at: datetime
    source: str


def infer_instr_label(dataset_name: str) -> str:
    if "INSTR0" in dataset_name:
        return "INSTR0"
    if "INSTR1" in dataset_name:
        return "INSTR1"
    return "OTHER"


def get_creation_time(path: Path) -> tuple[datetime, str]:
    """Return best creation timestamp and source name for a filesystem entry."""
    stat_result = path.stat()

    # Native birth time support (macOS, some Linux builds, BSD).
    if hasattr(stat_result, "st_birthtime"):
        return datetime.fromtimestamp(stat_result.st_birthtime), "birthtime"

    # Linux fallback: parse GNU stat birth-time placeholder (%w).
    # If unavailable, stat prints '-' and we fall back to mtime.
    try:
        proc = subprocess.run(
            ["stat", "-c", "%w", str(path)],
            check=False,
            capture_output=True,
            text=True,
        )
        birth_raw = proc.stdout.strip()
        if proc.returncode == 0 and birth_raw and birth_raw != "-":
            # Example: 2026-03-18 23:04:57.786097750 +0100
            fmt = "%Y-%m-%d %H:%M:%S.%f %z"
            parsed = datetime.strptime(birth_raw, fmt)
            return parsed.replace(tzinfo=None), "stat_birth"
    except Exception:
        pass

    return datetime.fromtimestamp(stat_result.st_mtime), "mtime"


def collect_events(datasets_root: Path, dataset_names: list[str]) -> list[ParquetCreationEvent]:
    events: list[ParquetCreationEvent] = []
    for dataset_name in dataset_names:
        chunk_dir = datasets_root / dataset_name / "data" / "chunk-000"
        if not chunk_dir.exists():
            print(f"[WARN] Missing directory: {chunk_dir}")
            continue

        parquet_files = sorted(chunk_dir.glob("*.parquet"))
        if not parquet_files:
            print(f"[WARN] No parquet files found: {chunk_dir}")
            continue

        label = infer_instr_label(dataset_name)
        for parquet_path in parquet_files:
            created_at, source = get_creation_time(parquet_path)
            events.append(
                ParquetCreationEvent(
                    dataset=dataset_name,
                    label=label,
                    parquet_file=parquet_path,
                    created_at=created_at,
                    source=source,
                )
            )

    events.sort(key=lambda e: e.created_at)
    return events


def print_summary(events: list[ParquetCreationEvent]) -> None:
    print("\nChronological parquet creation events:")
    for idx, event in enumerate(events, start=1):
        print(
            f"{idx:03d} | {event.created_at} | {event.label:6s} | "
            f"{event.dataset:35s} | {event.parquet_file.name:15s} | {event.source}"
        )


def build_daily_timeline_plots(
    events: list[ParquetCreationEvent], output_path: Path, show_plot: bool
) -> None:
    events_by_day: dict[str, list[ParquetCreationEvent]] = defaultdict(list)
    for event in events:
        day_key = event.created_at.strftime("%Y-%m-%d")
        events_by_day[day_key].append(event)

    y_map = {"INSTR0": 0, "INSTR1": 1}
    colors = {"INSTR0": "#136f63", "INSTR1": "#d95d39"}

    for day_key in sorted(events_by_day.keys()):
        day_events = sorted(events_by_day[day_key], key=lambda e: e.created_at)
        instr0_events = [e for e in day_events if e.label == "INSTR0"]
        instr1_events = [e for e in day_events if e.label == "INSTR1"]

        fig, ax = plt.subplots(figsize=(13, 4.8))

        for label, group_events in (("INSTR0", instr0_events), ("INSTR1", instr1_events)):
            if not group_events:
                continue
            xs = [e.created_at for e in group_events]
            ys = [y_map[label]] * len(group_events)
            ax.scatter(
                xs,
                ys,
                s=28,
                alpha=0.85,
                c=colors[label],
                edgecolors="black",
                linewidths=0.25,
                label=f"{label} ({len(group_events)})",
            )

        ax.set_yticks([0, 1], labels=["INSTR0", "INSTR1"])
        ax.set_xlabel("Parquet creation time")
        ax.set_ylabel("Dataset family")
        ax.set_title(f"Parquet creation chronology on {day_key}: INSTR0 vs INSTR1")
        ax.grid(True, linestyle="--", linewidth=0.5, alpha=0.35)
        ax.legend(loc="upper left")

        fig.autofmt_xdate(rotation=25)
        fig.tight_layout()

        daily_output = output_path.with_name(f"{output_path.stem}_{day_key}{output_path.suffix}")
        fig.savefig(daily_output, dpi=170)
        print(f"\nSaved daily plot to: {daily_output}")

        if show_plot:
            plt.show()
        plt.close(fig)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Plot chronological parquet file creation times for selected datasets."
    )
    parser.add_argument(
        "--datasets-root",
        type=Path,
        default=Path("datasets"),
        help="Path containing dataset directories.",
    )
    parser.add_argument(
        "--datasets",
        nargs="+",
        default=DEFAULT_DATASETS,
        help="Dataset directory names to scan.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("outputs") / "parquet_creation_timeline_instr0_vs_instr1.png",
        help="Base output PNG path. The script appends _YYYY-MM-DD per daily plot.",
    )
    parser.add_argument(
        "--show",
        action="store_true",
        help="Display the plot in a window.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    args.output.parent.mkdir(parents=True, exist_ok=True)

    events = collect_events(args.datasets_root, args.datasets)
    if not events:
        print("No parquet events found. Check dataset paths.")
        return 1

    print_summary(events)
    build_daily_timeline_plots(events, args.output, args.show)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
