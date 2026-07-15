#!/usr/bin/env python3
"""
Compute EER for w2v2-AASIST-AIED-CSD scores against ASVspoof2021 LA keys.

Usage:
    python compute_eer.py \
        --scores ./eval_output/w2v2_AASIST_AIED_CSD.txt \
        --keys   ./datasets/keys/CM/trial_metadata.txt \
        --subset eval
"""
import argparse
import sys
import numpy as np

from eval_metric_LA import compute_eer  # your attached file


def load_scores(scores_path):
    """
    Parses lines like:
        LA_E_2042719 -2.9710092544555664
    Returns dict: filename -> score (float)
    """
    scores = {}
    with open(scores_path, "r") as fh:
        for lineno, line in enumerate(fh, 1):
            line = line.strip()
            if not line:
                continue
            parts = line.split()
            if len(parts) < 2:
                print(f"  ⚠ line {lineno}: unexpected format, skipping: {line}",
                      file=sys.stderr)
                continue
            fname, score = parts[0], parts[1]
            try:
                scores[fname] = float(score)
            except ValueError:
                print(f"  ⚠ line {lineno}: bad score value, skipping: {line}",
                      file=sys.stderr)
    return scores


def load_keys(keys_path, filename_col, label_col, subset_col=None, subset_filter=None):
    labels = {}
    with open(keys_path, "r") as fh:
        for lineno, line in enumerate(fh, 1):
            line = line.strip()
            if not line:
                continue
            parts = line.split()
            if len(parts) <= max(filename_col, label_col):
                print(f"  ⚠ line {lineno}: too few columns, skipping: {line}",
                      file=sys.stderr)
                continue

            if subset_col is not None and subset_filter is not None:
                if len(parts) <= subset_col:
                    continue
                if parts[subset_col] != subset_filter:
                    continue

            fname = parts[filename_col]
            label_raw = parts[label_col].lower()
            labels[fname] = 1 if label_raw == "bonafide" else 0
    return labels


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--scores", required=True,
                         help="Score file produced by produce_evaluation_file_2021 "
                              "(lines: '<filename> <score>')")
    parser.add_argument("--keys", required=True,
                         help="ASVspoof2021 LA key/protocol file with ground-truth labels")
    parser.add_argument("--filename_col", type=int, default=1,
                         help="0-indexed column in --keys holding the filename "
                              "(default: 1, matching 'speaker filename ...' layout)")
    parser.add_argument("--label_col", type=int, default=5,
                         help="0-indexed column in --keys holding 'bonafide'/'spoof' "
                              "(default: 5 — VERIFY against your actual file with --preview)")
    parser.add_argument("--subset_col", type=int, default=None,
                         help="0-indexed column holding partition name (e.g. 'eval', "
                              "'progress', 'hidden_track'), if present")
    parser.add_argument("--subset", type=str, default=None,
                         help="Only keep rows where --subset_col equals this value "
                              "(e.g. 'eval' — required for official ASVspoof2021 scoring "
                              "if your key file mixes partitions)")
    parser.add_argument("--preview", action="store_true",
                         help="Print first 5 parsed key rows and exit, to verify column indices")
    args = parser.parse_args()

    if args.preview:
        n = 0
        with open(args.keys, "r") as fh:
            for line in fh:
                parts = line.strip().split()
                print(parts)
                n += 1
                if n >= 5:
                    break
        print("\nCheck: does the value at --filename_col look like a filename "
              "(e.g. LA_E_XXXXXXX), and --label_col look like 'bonafide'/'spoof'? "
              "Adjust --filename_col / --label_col / --subset_col accordingly.")
        return

    print(f"Loading scores: {args.scores}")
    scores = load_scores(args.scores)
    print(f"  → {len(scores)} scored files")

    print(f"Loading keys: {args.keys}")
    labels = load_keys(
        args.keys,
        filename_col=args.filename_col,
        label_col=args.label_col,
        subset_col=args.subset_col,
        subset_filter=args.subset,
    )
    print(f"  → {len(labels)} labeled trials"
          + (f" (filtered to subset='{args.subset}')" if args.subset else ""))

    bonafide_scores = []
    spoof_scores = []
    missing = 0

    for fname, score in scores.items():
        if fname not in labels:
            missing += 1
            continue
        if labels[fname] == 1:
            bonafide_scores.append(score)
        else:
            spoof_scores.append(score)

    if missing:
        print(f"  ⚠ {missing} scored file(s) had no matching key entry — skipped")

    bonafide_scores = np.array(bonafide_scores)
    spoof_scores = np.array(spoof_scores)

    print(f"\nBonafide trials: {len(bonafide_scores)}")
    print(f"Spoof trials   : {len(spoof_scores)}")

    if len(bonafide_scores) == 0 or len(spoof_scores) == 0:
        print("Need at least one bonafide AND one spoof score to compute EER. "
              "Check --filename_col / --label_col / --subset match your key file.")
        sys.exit(1)

    eer, threshold = compute_eer(bonafide_scores, spoof_scores)

    print("\n" + "=" * 50)
    print(f"  EER        : {eer * 100:.2f}%")
    print(f"  Threshold  : {threshold:.4f}")
    print("=" * 50)
    print("\nNote: this threshold is on the raw score scale your model "
          "produces (which looks like unbounded logits, not [0,1] "
          "probabilities) — use THIS value, not 0.5, as your bonafide/spoof "
          "decision boundary in inference scripts.")


if __name__ == "__main__":
    main()