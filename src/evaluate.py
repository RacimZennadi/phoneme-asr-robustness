"""
evaluate.py
Stage 5 of the pipeline.
Reads all prediction manifests (pred_clean.jsonl + pred_snr*.jsonl) for one
language and computes the mean PER at each noise level.
Writes a metrics.json file (consumed by dvc metrics show) and a per-language
PER vs SNR curve as a PNG figure.
"""
import argparse
import json
import os
from pathlib import Path

import editdistance
import matplotlib.pyplot as plt


def normalize(phon: str) -> str:
    # remove spaces so "aɪ kæn" and "aɪkæn" are treated identically
    # this is needed because espeak groups by word but wav2vec2 outputs
    # one token per phoneme — comparing at character level avoids the mismatch
    return phon.replace(" ", "").strip()


def compute_per(ref: str, hyp: str) -> float:
    ref_chars = list(normalize(ref))
    hyp_chars = list(normalize(hyp))
    if not ref_chars:
        return 0.0
    # edit distance at character level = PER when each char is one phoneme
    return editdistance.eval(ref_chars, hyp_chars) / len(ref_chars)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest_dir", required=True, help="folder with pred_*.jsonl files")
    parser.add_argument("--lang",         required=True)
    parser.add_argument("--metrics",      required=True, help="output metrics.json path")
    parser.add_argument("--figure",       required=True, help="output PNG path")
    args = parser.parse_args()

    manifest_dir = Path(args.manifest_dir).resolve()
    metrics_path = Path(args.metrics).resolve()
    figure_path  = Path(args.figure).resolve()
    metrics_path.parent.mkdir(parents=True, exist_ok=True)
    figure_path.parent.mkdir(parents=True, exist_ok=True)

    # collect prediction files: clean first, then noisy in filename order
    pred_files = sorted(manifest_dir.glob("pred_snr*.jsonl"))
    clean_pred = manifest_dir / "pred_clean.jsonl"
    if clean_pred.exists():
        pred_files = [clean_pred] + list(pred_files)

    # compute mean PER across all utterances for each SNR level
    snr_to_per = {}
    for path in pred_files:
        records  = [json.loads(l) for l in path.read_text(encoding="utf-8").splitlines()]
        snr      = records[0]["snr_db"]   # None for clean, float for noisy
        mean_per = sum(compute_per(r["ref_phon"], r["hyp_phon"]) for r in records) / len(records)
        snr_to_per[snr] = round(mean_per, 4)
        label = f"SNR={snr:+.0f}" if snr is not None else "clean"
        print(f"  {label:>12}  PER = {mean_per:.4f}")

    # sort by SNR numerically, with clean appended at the end
    snrs   = sorted(snr_to_per, key=lambda x: x if x is not None else float("inf"))
    pers   = [snr_to_per[s] for s in snrs]
    labels = [f"{int(s):+d}" if s is not None else "clean" for s in snrs]

    # write metrics.json atomically
    metrics_out = {args.lang: {f"snr_{l}": p for l, p in zip(labels, pers)}}
    tmp = metrics_path.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(metrics_out, indent=2), encoding="utf-8")
    os.replace(tmp, metrics_path)
    print(f"Metrics → {metrics_path}")

    # plot PER vs SNR for this language
    plt.figure(figsize=(8, 5))
    plt.plot(range(len(pers)), pers, marker="o", label=args.lang)
    plt.xticks(range(len(labels)), labels)
    plt.xlabel("SNR (dB)")
    plt.ylabel("PER")
    plt.title("Phoneme Error Rate vs Noise Level")
    plt.legend()
    plt.tight_layout()
    plt.savefig(str(figure_path))
    print(f"Figure  → {figure_path}")


if __name__ == "__main__":
    main()
