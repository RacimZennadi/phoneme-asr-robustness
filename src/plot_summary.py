"""
plot_summary.py
Final stage of the pipeline.
Reads all per-language metrics files from the metrics/ folder and plots them
together on a single figure, including a cross-language mean curve.
This stage only runs after all language-specific evaluate stages are done.
"""
import argparse
import json
import os
import statistics
from pathlib import Path

import matplotlib.pyplot as plt


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--metrics_dir", required=True, help="folder with en.json, fr.json, etc.")
    parser.add_argument("--figure",      required=True, help="output PNG path")
    args = parser.parse_args()

    metrics_dir = Path(args.metrics_dir).resolve()
    figure_path = Path(args.figure).resolve()
    figure_path.parent.mkdir(parents=True, exist_ok=True)

    # load one dict per language: {"snr_-5": 0.9, "snr_+0": 0.7, ...}
    all_data = {}
    for path in sorted(metrics_dir.glob("*.json")):
        lang = path.stem
        data = json.loads(path.read_text(encoding="utf-8"))
        all_data[lang] = data[lang]   # unwrap the outer {"en": {...}} wrapper

    if not all_data:
        print("No metrics found.")
        return

    # sort SNR keys numerically, with "snr_clean" always last
    def snr_sort_key(k):
        return float("inf") if k == "snr_clean" else float(k.replace("snr_", ""))

    sample     = next(iter(all_data.values()))
    snr_keys   = sorted(sample.keys(), key=snr_sort_key)
    snr_labels = [k.replace("snr_", "") for k in snr_keys]   # e.g. "-5", "+10", "clean"

    plt.figure(figsize=(10, 6))

    # one curve per language
    per_matrix = []
    for lang, values in all_data.items():
        pers = [values[k] for k in snr_keys]
        per_matrix.append(pers)
        plt.plot(range(len(pers)), pers, marker="o", label=lang)

    # add cross-language mean curve (dashed black) if more than one language
    if len(per_matrix) > 1:
        mean_pers = [
            statistics.mean(per_matrix[i][j] for i in range(len(per_matrix)))
            for j in range(len(snr_keys))
        ]
        plt.plot(
            range(len(mean_pers)), mean_pers,
            marker="s", linestyle="--", linewidth=2,
            color="black", label="mean"
        )

    plt.xticks(range(len(snr_labels)), snr_labels)
    plt.xlabel("SNR (dB)")
    plt.ylabel("PER")
    plt.title("Phoneme Error Rate vs Noise Level")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(str(figure_path))
    print(f"Summary figure → {figure_path}")


if __name__ == "__main__":
    main()
