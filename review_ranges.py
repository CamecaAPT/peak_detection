"""Interactive range review tool.

Displays APT spectra in navigable chunks with RRNG range overlays so a user
can visually verify that each range spans exactly one peak.

Usage
-----
    py review_ranges.py <data_directory> <csv_file> [--max_window_size 10] [--buffer 0.5] [--show_all]
"""

import argparse
import os
import sys

import matplotlib
matplotlib.use("QtAgg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from peak_detection.data_io import load_apt_from_file, parse_rrng
from peak_detection.models import PeakRange


# ---------------------------------------------------------------------------
# Chunk computation
# ---------------------------------------------------------------------------

def compute_range_chunks(ranges: list[PeakRange], max_window_size: float,
                         buffer: float) -> list[dict]:
    """Greedy forward sweep that groups sorted ranges into non-overlapping chunks.

    Returns a list of dicts with keys:
        ranges  – list[PeakRange] in this chunk
        view_start, view_end – display bounds (with buffer)
    """
    if not ranges:
        return []

    sorted_ranges = sorted(ranges, key=lambda r: r.start)
    chunks: list[dict] = []
    i = 0

    while i < len(sorted_ranges):
        anchor = sorted_ranges[i].start
        chunk_end = anchor + max_window_size

        # Expand until stable
        while True:
            collected = []
            for r in sorted_ranges[i:]:
                if r.start < chunk_end:
                    collected.append(r)
                    if r.end > chunk_end:
                        chunk_end = r.end
                else:
                    break
            # Re-sweep to check stability
            new_end = chunk_end
            for r in sorted_ranges[i:]:
                if r.start < new_end:
                    if r.end > new_end:
                        new_end = r.end
                else:
                    break
            if new_end == chunk_end:
                break
            chunk_end = new_end

        chunks.append({
            "ranges": collected,
            "view_start": collected[0].start - buffer,
            "view_end": collected[-1].end + buffer,
        })
        i += len(collected)

    return chunks


def compute_full_chunks(ranges: list[PeakRange], max_window_size: float,
                        buffer: float, x_max: float) -> list[dict]:
    """Generate fixed-width chunks covering 0..x_max, never cutting a range."""
    sorted_ranges = sorted(ranges, key=lambda r: r.start)
    range_idx = 0
    chunks: list[dict] = []
    pos = 0.0

    while pos < x_max:
        chunk_start = pos
        chunk_end = pos + max_window_size

        # Expand to avoid cutting any range
        while True:
            expanded = False
            while range_idx + len([
                r for r in sorted_ranges[range_idx:]
                if r.start < chunk_end and r.end > chunk_end
            ]) > 0:
                for r in sorted_ranges[range_idx:]:
                    if r.start < chunk_end and r.end > chunk_end:
                        chunk_end = r.end
                        expanded = True
                if not expanded:
                    break
            break

        # Collect ranges in this chunk
        collected = []
        temp_idx = range_idx
        while temp_idx < len(sorted_ranges) and sorted_ranges[temp_idx].start < chunk_end:
            collected.append(sorted_ranges[temp_idx])
            temp_idx += 1

        view_start = chunk_start - buffer
        view_end = chunk_end + buffer

        chunks.append({
            "ranges": collected,
            "view_start": max(view_start, 0),
            "view_end": view_end,
        })

        range_idx = temp_idx
        pos = chunk_end

    return chunks


# ---------------------------------------------------------------------------
# File loading
# ---------------------------------------------------------------------------

def load_file_pair(data_directory: str, row) -> tuple:
    """Load an APT/RRNG pair. Returns (x, spectrum_log, ranges, label) or None."""
    apt_col = row.iloc[0]
    rrng_col = row.iloc[1]

    apt_path = os.path.join(data_directory, apt_col)
    rrng_path = os.path.join(data_directory, rrng_col)

    # Fallback to apt_path / rrng_path columns if they exist
    if not os.path.isfile(apt_path) and "apt_path" in row.index:
        apt_path = row["apt_path"]
    if not os.path.isfile(rrng_path) and "rrng_path" in row.index:
        rrng_path = row["rrng_path"]

    if not os.path.isfile(apt_path):
        print(f"  WARNING: APT not found: {apt_path}")
        return None
    if not os.path.isfile(rrng_path):
        print(f"  WARNING: RRNG not found: {rrng_path}")
        return None

    x, _spectrum, spectrum_log = load_apt_from_file(apt_path)
    # spectrum_log is a torch tensor; convert to numpy
    y = np.array(spectrum_log, dtype=np.float32)
    ranges = parse_rrng(rrng_path)
    label = os.path.basename(apt_col)
    return x, y, ranges, label


# ---------------------------------------------------------------------------
# Viewer
# ---------------------------------------------------------------------------

class RangeReviewer:
    def __init__(self, data_directory: str, csv_path: str,
                 max_window_size: float, buffer: float, show_all: bool):
        self.data_directory = data_directory
        self.max_window_size = max_window_size
        self.buffer = buffer
        self.show_all = show_all

        self.df = pd.read_csv(csv_path)
        self.file_idx = -1
        self.chunk_idx = 0

        # Current file state
        self.x = None
        self.y = None
        self.ranges = None
        self.chunks = None
        self.file_label = ""

        self.fig, self.ax = plt.subplots(figsize=(14, 5))
        self.fig.canvas.mpl_connect("key_press_event", self._on_key)

        # Load first file and draw
        if not self._next_file():
            print("No valid file pairs found.")
            sys.exit(1)
        self._draw()
        plt.show()

    # -- file navigation ---------------------------------------------------

    def _load_file(self, idx: int) -> bool:
        if idx < 0 or idx >= len(self.df):
            return False
        row = self.df.iloc[idx]
        result = load_file_pair(self.data_directory, row)
        if result is None:
            return False
        self.x, self.y, self.ranges, self.file_label = result

        if self.show_all:
            self.chunks = compute_full_chunks(
                self.ranges, self.max_window_size, self.buffer, float(self.x[-1]))
        else:
            self.chunks = compute_range_chunks(
                self.ranges, self.max_window_size, self.buffer)

        if not self.chunks:
            # File with no ranges – create a single chunk of the whole spectrum
            self.chunks = [{"ranges": [], "view_start": 0,
                            "view_end": float(self.x[-1])}]
        self.file_idx = idx
        return True

    def _next_file(self) -> bool:
        idx = self.file_idx + 1
        while idx < len(self.df):
            if self._load_file(idx):
                self.chunk_idx = 0
                return True
            idx += 1
        return False

    def _prev_file(self) -> bool:
        idx = self.file_idx - 1
        while idx >= 0:
            if self._load_file(idx):
                self.chunk_idx = len(self.chunks) - 1
                return True
            idx -= 1
        return False

    # -- drawing -----------------------------------------------------------

    def _draw(self):
        ax = self.ax
        ax.clear()

        chunk = self.chunks[self.chunk_idx]
        vs, ve = chunk["view_start"], chunk["view_end"]

        # Spectrum line
        mask = (self.x >= vs) & (self.x <= ve)
        ax.plot(self.x[mask], self.y[mask], color="black", alpha=0.3)

        # Range overlays
        for r in chunk["ranges"]:
            ax.axvspan(r.start, r.end, color="blue", alpha=0.15)
            if r.label:
                center = (r.start + r.end) / 2
                ax.text(center, 0.85, r.label, color="blue", fontsize=6,
                        ha="center", va="bottom", rotation=90, alpha=0.7)

        ax.set_xlim(vs, ve)
        ax.set_ylim(0, 1.05)
        ax.grid(alpha=0.2)

        n_ranges = len(chunk["ranges"])
        title = (f"{self.file_label}  |  chunk {self.chunk_idx + 1}/"
                 f"{len(self.chunks)}  |  {n_ranges} ranges  |  "
                 f"{vs:.2f}–{ve:.2f} Da  |  "
                 f"file {self.file_idx + 1}/{len(self.df)}")
        ax.set_title(title, fontsize=9)
        ax.set_xlabel("Mass-to-charge (Da)")
        ax.set_ylabel("Intensity (log-mapped)")

        self.fig.canvas.draw_idle()

    # -- key handling ------------------------------------------------------

    def _on_key(self, event):
        if event.key == "right":
            if self.chunk_idx < len(self.chunks) - 1:
                self.chunk_idx += 1
                self._draw()
            else:
                if self._next_file():
                    self._draw()
                else:
                    print("Reached last chunk of last file.")
        elif event.key == "left":
            if self.chunk_idx > 0:
                self.chunk_idx -= 1
                self._draw()
            else:
                if self._prev_file():
                    self._draw()
                else:
                    print("Already at first chunk of first file.")
        elif event.key in ("q", "escape"):
            plt.close(self.fig)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Interactive range review: arrow-key through APT spectra "
                    "with RRNG range overlays.")
    parser.add_argument("data_directory", help="Folder with APT + RRNG files")
    parser.add_argument("csv_file", help="CSV with apt_file and rrng_file columns")
    parser.add_argument("--max_window_size", type=float, default=10,
                        help="Max chunk width in Da (default 10)")
    parser.add_argument("--buffer", type=float, default=0.5,
                        help="Padding in Da on each side (default 0.5)")
    parser.add_argument("--show_all", action="store_true",
                        help="Step through the entire spectrum, not just regions with ranges")
    args = parser.parse_args()

    RangeReviewer(
        data_directory=args.data_directory,
        csv_path=args.csv_file,
        max_window_size=args.max_window_size,
        buffer=args.buffer,
        show_all=args.show_all,
    )


if __name__ == "__main__":
    main()
