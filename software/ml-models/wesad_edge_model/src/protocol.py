"""
Protocol facts from the WESAD label vectors (Step C): condition segments, stress onset, stress/amusement order.

Run from software/ml-models/wesad_edge_model:
    python -m src.protocol --data "C:\\datasets\\WESAD"
Writes results/metrics/stepc_label_segments.csv and results/metrics/stepc_condition_order.csv.
Labels only; no signal is used. These files feed alert latency (stress onset) and the order analysis (C ii).
"""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from src import preprocessing as pp

ROOT = Path(__file__).resolve().parents[1]
METRICS = ROOT / "results" / "metrics"
SEGMENTS_CSV = METRICS / "stepc_label_segments.csv"
ORDER_CSV = METRICS / "stepc_condition_order.csv"


def segments(label: np.ndarray) -> pd.DataFrame:
    """Runs of constant label: label, start_s, end_s (end exclusive)."""
    change = np.flatnonzero(np.diff(label)) + 1
    starts = np.r_[0, change]
    ends = np.r_[change, len(label)]
    return pd.DataFrame({"label": label[starts], "start_s": starts / pp.FS_LABEL, "end_s": ends / pp.FS_LABEL})


def build(data_root: Path) -> tuple:
    seg_rows, order_rows = [], []
    for name, pkl in pp.find_subjects(data_root):
        lab = pp.load_subject(pkl)["label"]
        seg = segments(lab).assign(subject=name)
        seg["duration_s"] = seg.end_s - seg.start_s
        seg_rows.append(seg)
        study = seg[seg.label.isin([1, 2, 3, 4])]
        first = {k: float(study[study.label == k].start_s.min()) for k in (1, 2, 3)}
        n_stress = int((seg.label == 2).sum())
        order_rows.append({
            "subject": name, "sequence_labels_1_to_4": "-".join(str(int(x)) for x in study.label),
            "baseline_onset_s": first[1], "stress_onset_s": first[2], "amusement_onset_s": first[3],
            "stress_end_s": float(seg[seg.label == 2].end_s.max()), "n_stress_segments": n_stress,
            "order": "stress_first" if first[2] < first[3] else "amusement_first",
        })
        print(f"  {name}: {order_rows[-1]['sequence_labels_1_to_4']}  ({order_rows[-1]['order']})", flush=True)
    seg = pd.concat(seg_rows, ignore_index=True)[["subject", "label", "start_s", "end_s", "duration_s"]]
    order = pd.DataFrame(order_rows)
    assert (order.n_stress_segments == 1).all(), "expected exactly one stress segment per subject"
    return seg, order


def load_order() -> pd.DataFrame:
    assert ORDER_CSV.exists(), f"run `python -m src.protocol --data ...` first ({ORDER_CSV} missing)"
    return pd.read_csv(ORDER_CSV)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data", required=True)
    args = ap.parse_args()
    seg, order = build(Path(args.data))
    seg.to_csv(SEGMENTS_CSV, index=False)
    order.to_csv(ORDER_CSV, index=False)
    print(order.to_string(index=False))
    print(order.order.value_counts().to_string())


if __name__ == "__main__":
    main()
