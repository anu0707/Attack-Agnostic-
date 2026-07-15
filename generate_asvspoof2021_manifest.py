#!/usr/bin/env python3
"""
Convert an ASVspoof 2021 LA CM trial metadata file into the CSV manifest
format consumed by src/eval.py.

ASVspoof 2021 LA directory layout (after download):
  <root>/
    LA/
      ASVspoof2021_LA_eval/
        flac/
          <utterance_id>.flac   <-- audio files
      keys/
        LA/
          CM/
            trial_metadata.txt  <-- CM ground-truth labels

The CM trial_metadata.txt columns (space-separated, 0-indexed):
  0: speaker_id
  1: utterance_id          <- used as audio filename stem
  2: '-'
  3: dataset tag
  4: attack_type           <- stored as attack_id
  5: label                 <- 'bonafide' or 'spoof'
  6: '-'
  7: phase                 <- 'progress', 'eval', or 'hidden_track'

Usage:
  python generate_asvspoof2021_manifest.py \\
      --keys-dir   /data/asvspoof2021/keys/LA/CM \\
      --audio-dir  /data/asvspoof2021/LA/ASVspoof2021_LA_eval/flac \\
      --phase      eval \\
      --output     data/manifests/asvspoof2021_la_eval.csv
"""

from __future__ import annotations

import argparse
import csv
import os
from pathlib import Path


PHASE_CHOICES = ("progress", "eval", "hidden_track")
AUDIO_EXTENSIONS = (".flac", ".wav", ".mp3")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build an eval manifest from ASVspoof 2021 LA CM trial metadata."
    )
    parser.add_argument(
        "--keys-dir",
        required=True,
        help="Path to the CM keys directory containing trial_metadata.txt. "
             "Download from https://www.asvspoof.org/asvspoof2021/keys.tar.gz "
             "and point here to the extracted keys/LA/CM folder.",
    )
    parser.add_argument(
        "--audio-dir",
        required=True,
        help="Path to the directory containing the evaluation audio files "
             "(ASVspoof2021_LA_eval/flac/).",
    )
    parser.add_argument(
        "--phase",
        default="eval",
        choices=PHASE_CHOICES,
        help="Evaluation phase to extract. Default: eval",
    )
    parser.add_argument(
        "--output",
        default="data/manifests/asvspoof2021_la_eval.csv",
        help="Output CSV manifest path.",
    )
    parser.add_argument(
        "--corpus",
        default="asvspoof2021_la",
        help="Corpus tag written into the manifest (default: asvspoof2021_la).",
    )
    return parser.parse_args()


def find_audio_file(audio_dir: Path, stem: str) -> str | None:
    """Return the first matching audio file path for the given stem."""
    for ext in AUDIO_EXTENSIONS:
        candidate = audio_dir / f"{stem}{ext}"
        if candidate.exists():
            return str(candidate)
    return None


def load_cm_metadata(keys_dir: Path, phase: str) -> list[dict]:
    """Parse trial_metadata.txt and return rows matching the requested phase."""
    meta_path = keys_dir / "trial_metadata.txt"
    if not meta_path.exists():
        raise FileNotFoundError(
            f"CM trial_metadata.txt not found at {meta_path}. "
            "Download the official ASVspoof 2021 keys archive."
        )

    rows = []
    with meta_path.open("r", encoding="utf-8") as fh:
        for line in fh:
            parts = line.strip().split()
            if len(parts) < 8:
                continue
            row_phase = parts[7]
            if row_phase != phase:
                continue
            rows.append(
                {
                    "speaker_id": parts[0],
                    "utterance_id": parts[1],
                    "attack_id": parts[4],
                    "label": parts[5],   # 'bonafide' or 'spoof'
                    "phase": row_phase,
                }
            )
    return rows


def main() -> int:
    args = parse_args()

    keys_dir = Path(args.keys_dir)
    audio_dir = Path(args.audio_dir)
    output_path = Path(args.output)

    if not audio_dir.is_dir():
        raise NotADirectoryError(f"Audio directory not found: {audio_dir}")

    print(f"Loading CM metadata from {keys_dir} (phase={args.phase}) ...")
    cm_rows = load_cm_metadata(keys_dir, args.phase)
    print(f"  Found {len(cm_rows)} trials for phase '{args.phase}'")

    output_path.parent.mkdir(parents=True, exist_ok=True)

    missing = 0
    written = 0

    with output_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(
            f, fieldnames=["filepath", "label", "attack_id", "corpus", "split"]
        )
        writer.writeheader()

        for row in cm_rows:
            filepath = find_audio_file(audio_dir, row["utterance_id"])
            if filepath is None:
                missing += 1
                if missing <= 5:
                    print(f"  [WARN] Audio not found for: {row['utterance_id']}")
                continue

            writer.writerow(
                {
                    "filepath": filepath,
                    "label": row["label"],          # bonafide / spoof
                    "attack_id": row["attack_id"],
                    "corpus": args.corpus,
                    "split": args.phase,
                }
            )
            written += 1

    if missing > 5:
        print(f"  [WARN] ... and {missing - 5} more missing audio files.")

    print(f"\nDone. Written {written} rows -> {output_path}")
    if missing:
        print(f"  Skipped {missing} trials (audio file not found in {audio_dir})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
