"""
infer_single.py — check the prediction for one audio file, with timing.
"""
import argparse
import time
import librosa
import numpy as np
import torch
from torch import Tensor
from transformers import Wav2Vec2Model
from model_OnlyAASIST import Model
from autoencoder_model import Autoencoder
from model_OnlyCSD import SimpleDNN


def pad(x, max_len=64600):
    x_len = x.shape[0]
    if x_len >= max_len:
        return x[:max_len]
    num_repeats = int(max_len / x_len) + 1
    return np.tile(x, num_repeats)[:max_len]


def load_models(ckpt_path, device):
    w2v2 = Wav2Vec2Model.from_pretrained("facebook/wav2vec2-xls-r-300m").to(device)
    w2v2.eval()

    class Args:
        pass

    model1 = Model(Args(), device).to(device)
    model2 = Autoencoder(input_dim=160, hidden_dim1=128).to(device)
    model3 = SimpleDNN(input_size=128, hidden_size=80, num_classes=2,
                        domain_size=7, K=4).to(device)
    checkpoint = torch.load(ckpt_path, map_location=device)
    model1.load_state_dict(checkpoint["model1_state_dict"])
    model2.load_state_dict(checkpoint["model2_state_dict"])
    model3.load_state_dict(checkpoint["model3_state_dict"])
    model1.eval(); model2.eval(); model3.eval()
    return w2v2, model1, model2, model3


def run_inference(x_inp, dummy_domain, w2v2, model1, model2, model3, device, time_it=False):
    """
    One forward pass through the full pipeline. When time_it=True, returns
    (score, label_logits, elapsed_seconds) with proper CUDA synchronization
    so the timing reflects actual GPU compute, not just kernel-launch time.
    """
    if time_it and device.type == "cuda":
        torch.cuda.synchronize()
        t0 = time.perf_counter()
    elif time_it:
        t0 = time.perf_counter()

    with torch.no_grad():
        outputs_x = w2v2(x_inp, output_hidden_states=True)
        fifth_layer_embeddings = outputs_x.hidden_states[4]
        x = model1(fifth_layer_embeddings)
        x = model2(x)
        _, logits, _, _, _ = model3(x, dummy_domain, dummy_domain)

    if time_it:
        if device.type == "cuda":
            torch.cuda.synchronize()   # wait for all queued GPU work to actually finish
        elapsed = time.perf_counter() - t0
        return logits, elapsed

    return logits, None


def predict_file(file_path, ckpt_path, device, threshold=0.0,
                  warmup_runs=3, timed_runs=10):
    # ── Model loading (timed separately — this is a one-time cost per
    #    session, not part of per-utterance inference latency) ──────────
    t_load0 = time.perf_counter()
    w2v2, model1, model2, model3 = load_models(ckpt_path, device)
    load_time = time.perf_counter() - t_load0
    print(f"Model load time: {load_time:.3f}s")

    # ── Audio loading + preprocessing (also excluded from GPU inference
    #    timing, since it's CPU-bound I/O + resampling, not model compute) ──
    X, _ = librosa.load(file_path, sr=16000)
    x_inp = Tensor(pad(X, 64600)).unsqueeze(0).to(device)
    dummy_domain = torch.zeros(1, dtype=torch.long).to(device)

    # ── Warmup: absorb one-time CUDA kernel compilation / cuDNN
    #    autotuning cost so it doesn't pollute the timed measurements ──────
    for _ in range(warmup_runs):
        run_inference(x_inp, dummy_domain, w2v2, model1, model2, model3, device, time_it=False)

    # ── Timed runs ────────────────────────────────────────────────────
    times = []
    logits = None
    for _ in range(timed_runs):
        logits, elapsed = run_inference(x_inp, dummy_domain, w2v2, model1, model2, model3, device, time_it=True)
        times.append(elapsed)

    times = np.array(times)
    mean_t, std_t, min_t, max_t = times.mean(), times.std(), times.min(), times.max()

    score = logits[0, 1].item()
    probs = torch.softmax(logits, dim=1)[0]
    label = "REAL (bonafide)" if score > threshold else "FAKE (spoof)"

    print(f"File: {file_path}")
    print(f"Raw score (class-1 logit): {score:.4f}")
    print(f"Softmax probs -> fake: {probs[0].item():.4f}, real: {probs[1].item():.4f}")
    print(f"Decision (threshold={threshold}): {label}")
    print()
    print(f"Inference timing over {timed_runs} runs (after {warmup_runs} warmup runs):")
    print(f"  mean : {mean_t*1000:.2f} ms")
    print(f"  std  : {std_t*1000:.2f} ms")
    print(f"  min  : {min_t*1000:.2f} ms")
    print(f"  max  : {max_t*1000:.2f} ms")
    print(f"  RTF  : {mean_t / (len(X) / 16000):.4f}  (real-time factor; <1.0 = faster than real-time)")

    return score, label, mean_t


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--file_path", type=str, required=True)
    parser.add_argument("--ckpt_path", type=str, required=True)
    parser.add_argument("--threshold", type=float, default=0.0)
    parser.add_argument("--warmup_runs", type=int, default=3,
                         help="Number of untimed warmup passes before measuring")
    parser.add_argument("--timed_runs", type=int, default=10,
                         help="Number of timed passes to average over")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type == "cpu":
        raise RuntimeError("This model requires a CUDA GPU (CSD layer params are hardcoded to cuda:0).")

    predict_file(args.file_path, args.ckpt_path, device, args.threshold,
                 args.warmup_runs, args.timed_runs)