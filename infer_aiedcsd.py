"""
run_benchmark.py — score ASVspoof2021 LA (official) and/or your own
audio folder, using the same w2v2 -> AASIST -> AIED -> CSD checkpoint.

Usage:
  # ASVspoof2021 LA official benchmark (EER + min tDCF)
  python run_benchmark.py --mode asv2021_la \
      --ckpt_path ./w2v2_AASIST_AIED_CSD_DNN/best_model.pth \
      --la_base_dir /base_path/Data/ASVspoof2021_LA_eval/ \
      --la_protocol /base_path/Data/ASVspoof2021_LA_eval/ASVspoof2021.LA.cm.eval.trl.txt \
      --keys_dir ./keys \
      --phase eval

  # Your own audio folder (raw scores; EER too if you pass --labels_csv)
  python run_benchmark.py --mode custom \
      --ckpt_path ./w2v2_AASIST_AIED_CSD_DNN/best_model.pth \
      --custom_folder /path/to/your/wavs \
      --labels_csv /path/to/labels.csv   # optional: filename,label(bonafide/spoof)

  # Both in one run
  python run_benchmark.py --mode both --ckpt_path ... --la_base_dir ... \
      --la_protocol ... --keys_dir ./keys --phase eval --custom_folder ...
"""
import argparse
import csv
import os

import librosa
import numpy as np
import torch
from torch import Tensor
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
from transformers import Wav2Vec2Model

from model_OnlyAASIST import Model              # model1
from autoencoder_model import Autoencoder        # model2
from model_OnlyCSD import SimpleDNN              # model3
from data_utils_CSD import genSpoof_list, Dataset_ASVspoof2021_eval
import eval_metric_LA as em                      # your repo's own EER/tDCF code
import evaluate_2021_LA as ev21                  # your repo's official LA scorer

AUDIO_EXTS = (".wav", ".flac", ".mp3", ".m4a", ".ogg")


# ---------- shared model loading ----------
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
    print("All models loaded.")
    return w2v2, model1, model2, model3


def combined_model(x, labels, domain, model1, model2, model3):
    x = model1(x)
    x = model2(x)
    x = model3(x, labels, domain)
    return x


def run_scoring(data_loader, w2v2, model1, model2, model3, device, out_path):
    results = []
    for batch_x, utt_ids, batch_domain in tqdm(data_loader):
        batch_x = batch_x.to(device)
        batch_domain = torch.as_tensor(batch_domain).to(device)
        with torch.no_grad():
            outputs_x = w2v2(batch_x, output_hidden_states=True)
            fifth_layer_embeddings = outputs_x.hidden_states[4]
            _, outputs, _, _, _ = combined_model(
                fifth_layer_embeddings, batch_domain, batch_domain,
                model1, model2, model3,
            )
        batch_score = outputs[:, 1].data.cpu().numpy().ravel()  # higher = bonafide
        results.extend(zip(utt_ids, batch_score.tolist()))

    with open(out_path, "w") as fh:
        for utt_id, score in results:
            fh.write(f"{utt_id} {score}\n")
    print(f"Scores saved to {out_path}")
    return results


# ---------- ASVspoof2021 LA official benchmark ----------
def bench_asv2021_la(args, w2v2, model1, model2, model3, device):
    file_eval = genSpoof_list(dir_meta=args.la_protocol, is_train=False, is_eval=False)
    print(f"ASVspoof2021 LA eval trials: {len(file_eval)}")
    eval_set = Dataset_ASVspoof2021_eval(list_IDs=file_eval, base_dir=args.la_base_dir)
    loader = DataLoader(eval_set, batch_size=args.batch_size, num_workers=8,
                         shuffle=False, drop_last=False)

    score_path = os.path.join(args.out_dir, "la2021_scores.txt")
    run_scoring(loader, w2v2, model1, model2, model3, device, score_path)

    # official EER + min tDCF using your repo's evaluate_2021_LA.py logic
    cm_key_file = os.path.join(args.keys_dir, "CM", "trial_metadata.txt")
    ev21.submit_file = score_path          # these are read as module-level
    ev21.truth_dir = args.keys_dir         # globals inside evaluate_2021_LA.py
    ev21.phase = args.phase
    ev21.asv_key_file = os.path.join(args.keys_dir, "ASV", "trial_metadata.txt")
    ev21.asv_scr_file = os.path.join(args.keys_dir, "ASV", "ASVTorch_Kaldi", "score.txt")
    ev21.cm_key_file = cm_key_file
    min_tDCF = ev21.eval_to_score_file(score_path, cm_key_file)
    print(f"[ASVspoof2021 LA] min-tDCF: {min_tDCF:.4f}")


# ---------- your own audio folder ----------
def pad(x, max_len=64600):
    x_len = x.shape[0]
    if x_len >= max_len:
        return x[:max_len]
    num_repeats = int(max_len / x_len) + 1
    return np.tile(x, num_repeats)[:max_len]


class CustomAudioDataset(Dataset):
    def __init__(self, file_paths, cut=64600):
        self.file_paths = file_paths
        self.cut = cut

    def __len__(self):
        return len(self.file_paths)

    def __getitem__(self, index):
        path = self.file_paths[index]
        X, _ = librosa.load(path, sr=16000)
        x_inp = Tensor(pad(X, self.cut))
        return x_inp, os.path.basename(path), 0


def bench_custom(args, w2v2, model1, model2, model3, device):
    file_paths = [
        os.path.join(args.custom_folder, f)
        for f in sorted(os.listdir(args.custom_folder))
        if f.lower().endswith(AUDIO_EXTS)
    ]
    print(f"Found {len(file_paths)} custom audio files.")
    dataset = CustomAudioDataset(file_paths)
    loader = DataLoader(dataset, batch_size=args.batch_size, num_workers=4, shuffle=False)

    score_path = os.path.join(args.out_dir, "custom_scores.txt")
    results = run_scoring(loader, w2v2, model1, model2, model3, device, score_path)

    # decision at a threshold (default 0.0, since these are raw logits;
    # re-tune this against a couple of files you know the label of)
    print(f"\n=== Custom audio decisions (threshold={args.threshold}) ===")
    for utt_id, score in results:
        label = "REAL (bonafide)" if score > args.threshold else "FAKE (spoof)"
        print(f"{utt_id}: score={score:.4f} -> {label}")

    # optional EER if the user supplies ground-truth labels
    if args.labels_csv:
        labels = {}
        with open(args.labels_csv) as fh:
            for row in csv.reader(fh):
                fname, lab = row[0], row[1].strip().lower()
                labels[fname] = lab
        bona, spoof = [], []
        for utt_id, score in results:
            lab = labels.get(utt_id)
            if lab == "bonafide":
                bona.append(score)
            elif lab == "spoof":
                spoof.append(score)
        if bona and spoof:
            eer, threshold = em.compute_eer(np.array(bona), np.array(spoof))
            print(f"\n[Custom audio] EER: {eer*100:.2f}% @ threshold {threshold:.4f}")
        else:
            print("labels_csv provided but no matching bonafide/spoof entries found.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=["asv2021_la", "custom", "both"], required=True)
    parser.add_argument("--ckpt_path", type=str, required=True)
    parser.add_argument("--out_dir", type=str, default="./benchmark_out")
    parser.add_argument("--batch_size", type=int, default=10)

    # ASVspoof2021 LA args
    parser.add_argument("--la_base_dir", type=str, help="e.g. .../ASVspoof2021_LA_eval/")
    parser.add_argument("--la_protocol", type=str, help="...cm.eval.trl.txt")
    parser.add_argument("--keys_dir", type=str, default="./keys")
    parser.add_argument("--phase", type=str, default="eval",
                         choices=["progress", "eval", "hidden_track"])

    # custom audio args
    parser.add_argument("--custom_folder", type=str)
    parser.add_argument("--labels_csv", type=str, default=None,
                         help="Optional CSV: filename,bonafide|spoof")
    parser.add_argument("--threshold", type=float, default=0.0)

    args = parser.parse_args()
    os.makedirs(args.out_dir, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type == "cpu":
        raise RuntimeError("This model requires a CUDA GPU (CSD layer params are hardcoded to cuda:0).")

    w2v2, model1, model2, model3 = load_models(args.ckpt_path, device)

    if args.mode in ("asv2021_la", "both"):
        assert args.la_base_dir and args.la_protocol, "Need --la_base_dir and --la_protocol"
        bench_asv2021_la(args, w2v2, model1, model2, model3, device)

    if args.mode in ("custom", "both"):
        assert args.custom_folder, "Need --custom_folder"
        bench_custom(args, w2v2, model1, model2, model3, device)