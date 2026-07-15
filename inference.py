#!/usr/bin/env python3
"""
Inference for AttackAgnostic CSD+AIED model.
Supports:
- single audio file
- directory (recursive) batch inference

Label mapping (from repo):
- 1 -> bonafide (real)
- 0 -> spoof (fake)
"""

import argparse
import csv
from pathlib import Path
from types import SimpleNamespace
from typing import List

import librosa
import numpy as np
import torch
import torch.nn.functional as F
from transformers import Wav2Vec2Model

from model_OnlyAASIST import Model as AASISTModel
from autoencoder_model import Autoencoder
from model_OnlyCSD import SimpleDNN


AUDIO_EXTS = {".wav", ".flac", ".mp3", ".m4a", ".ogg"}


def pad_or_trim(x: np.ndarray, max_len: int = 64600) -> np.ndarray:
    x_len = x.shape[0]
    if x_len >= max_len:
        return x[:max_len]
    num_repeats = int(max_len / x_len) + 1
    return np.tile(x, (num_repeats,))[:max_len]


def collect_audio_paths(inp: Path, recursive: bool = True) -> List[Path]:
    if inp.is_file():
        return [inp]
    if inp.is_dir():
        if recursive:
            return sorted([p for p in inp.rglob("*") if p.suffix.lower() in AUDIO_EXTS])
        return sorted([p for p in inp.glob("*") if p.suffix.lower() in AUDIO_EXTS])
    raise FileNotFoundError(f"Input path not found: {inp}")


def build_dummy_args():
    # AASIST constructor expects args, but inference doesn't need augmentation params.
    return SimpleNamespace()


def load_models(checkpoint_path: Path, device: torch.device, hf_model: str):
    ckpt = torch.load(checkpoint_path, map_location=device)

    args = build_dummy_args()

    w2v2 = Wav2Vec2Model.from_pretrained(hf_model).to(device).eval()
    model1 = AASISTModel(args, str(device)).to(device).eval()

    # IMPORTANT: this must match your edited autoencoder that returns encoded output.
    model2 = Autoencoder(input_dim=160, hidden_dim1=128).to(device).eval()
    model3 = SimpleDNN(input_size=128, hidden_size=80, num_classes=2, domain_size=7, K=4).to(device).eval()

    # checkpoint expected from main_w2v2_AASIST_AIED_CSD_hugface.py
    if "model1_state_dict" in ckpt:
        model1.load_state_dict(ckpt["model1_state_dict"], strict=True)
        model2.load_state_dict(ckpt["model2_state_dict"], strict=True)
        model3.load_state_dict(ckpt["model3_state_dict"], strict=True)
    else:
        raise KeyError(
            "Checkpoint does not contain model{1,2,3}_state_dict keys. "
            "Please pass the CSD+AIED training checkpoint."
        )

    return w2v2, model1, model2, model3


@torch.no_grad()
def infer_batch(
    audio_paths: List[Path],
    w2v2,
    model1,
    model2,
    model3,
    device: torch.device,
    sample_rate: int,
    max_len: int,
    batch_size: int,
    threshold_real: float,
    domain_id: int,
):
    results = []

    # process in mini-batches
    for i in range(0, len(audio_paths), batch_size):
        batch_paths = audio_paths[i : i + batch_size]

        wavs = []
        for p in batch_paths:
            x, _ = librosa.load(str(p), sr=sample_rate, mono=True)
            if x.size == 0:
                x = np.zeros(max_len, dtype=np.float32)
            x = pad_or_trim(x.astype(np.float32), max_len=max_len)
            wavs.append(x)

        x = torch.tensor(np.stack(wavs, axis=0), dtype=torch.float32, device=device)

        # wav2vec2 layer-5 features
        w2v2_out = w2v2(x, output_hidden_states=True)
        layer5 = w2v2_out.hidden_states[4]  # [B, T, 1024]

        # AASIST -> Autoencoder(encoded) -> CSD head
        aasist_out = model1(layer5)  # expected shape [B, 160]
        encoded = model2(aasist_out)  # your modified autoencoder returns encoded [B, 128]

        # model3 forward requires labels/domain; logits_common is what training uses for prediction
        dummy_labels = torch.zeros(encoded.size(0), dtype=torch.long, device=device)
        dummy_domains = torch.full((encoded.size(0),), int(domain_id), dtype=torch.long, device=device)

        _, logits_common, _, _, _ = model3(encoded, dummy_labels, dummy_domains)  # [B,2]

        probs = F.softmax(logits_common, dim=1)  # index 0 spoof, index 1 bonafide
        p_spoof = probs[:, 0].detach().cpu().numpy()
        p_real = probs[:, 1].detach().cpu().numpy()
        conf = probs.max(dim=1).values.detach().cpu().numpy()
        pred_idx = probs.argmax(dim=1).detach().cpu().numpy()

        for j, p in enumerate(batch_paths):
            pred_label = "real" if p_real[j] >= threshold_real else "fake"
            # Also keep argmax class from model
            argmax_label = "real" if int(pred_idx[j]) == 1 else "fake"

            results.append(
                {
                    "audio_path": str(p),
                    "pred_label_threshold": pred_label,   # based on p_real threshold
                    "pred_label_argmax": argmax_label,    # based on max prob class
                    "confidence": float(conf[j]),         # confidence of predicted argmax class
                    "p_real_bonafide": float(p_real[j]),  # class-1 probability
                    "p_fake_spoof": float(p_spoof[j]),    # class-0 probability
                    "score_class1_logit": float(logits_common[j, 1].item()),  # same style as repo eval
                }
            )

    return results


def main():
    parser = argparse.ArgumentParser("AttackAgnostic CSD+AIED inference")
    parser.add_argument("--input", required=True, help="Audio file or directory")
    parser.add_argument("--checkpoint", required=True, help="Path to best_model.pth from CSD+AIED training")
    parser.add_argument("--output_csv", default="inference_results.csv", help="Output CSV path")
    parser.add_argument("--hf_model", default="facebook/wav2vec2-xls-r-300m", help="HuggingFace wav2vec2 model")
    parser.add_argument("--sample_rate", type=int, default=16000)
    parser.add_argument("--max_len", type=int, default=64600)
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--threshold_real", type=float, default=0.5, help="Threshold on p_real for real/fake decision")
    parser.add_argument("--domain_id", type=int, default=0, help="Domain id for CSD forward (0 is safe default)")
    parser.add_argument("--recursive", action="store_true", help="Recursively scan input directory")
    parser.add_argument("--device", default="cuda", choices=["cuda", "cpu"])
    args = parser.parse_args()

    device = torch.device("cuda" if args.device == "cuda" and torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    inp = Path(args.input)
    audio_paths = collect_audio_paths(inp, recursive=args.recursive)
    if not audio_paths:
        raise SystemExit(f"No audio files found in: {inp}")

    print(f"Found {len(audio_paths)} audio files")

    w2v2, model1, model2, model3 = load_models(Path(args.checkpoint), device, args.hf_model)

    results = infer_batch(
        audio_paths=audio_paths,
        w2v2=w2v2,
        model1=model1,
        model2=model2,
        model3=model3,
        device=device,
        sample_rate=args.sample_rate,
        max_len=args.max_len,
        batch_size=args.batch_size,
        threshold_real=args.threshold_real,
        domain_id=args.domain_id,
    )

    out_csv = Path(args.output_csv)
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    with out_csv.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(results[0].keys()))
        writer.writeheader()
        writer.writerows(results)

    print(f"Saved: {out_csv}")
    print("Sample predictions:")
    for row in results[:5]:
        print(
            f"{row['audio_path']} -> {row['pred_label_threshold']} "
            f"(p_real={row['p_real_bonafide']:.4f}, conf={row['confidence']:.4f})"
        )


if __name__ == "__main__":
    main()