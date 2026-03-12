# src/plot_summary.py
import argparse, json
from pathlib import Path
import matplotlib.pyplot as plt
import yaml

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--metrics_dir", required=True)   # folder with en.json, fr.json etc
    parser.add_argument("--figure",      required=True)   # output path
    args = parser.parse_args()

    metrics_dir = Path(args.metrics_dir).resolve()
    figure_path = Path(args.figure).resolve()
    figure_path.parent.mkdir(parents=True, exist_ok=True)

    # load all per-language metrics
    all_data = {}
    for path in sorted(metrics_dir.glob("*.json")):
        lang = path.stem
        data = json.loads(path.read_text(encoding="utf-8"))
        all_data[lang] = data[lang]

    if not all_data:
        print("No metrics found.")
        return

    # collect all SNR labels in sorted order
    sample_lang = next(iter(all_data.values()))
    def snr_sort_key(k):
        if k == "snr_clean":
            return float("inf")
        return float(k.replace("snr_", ""))

    snr_keys = sorted(sample_lang.keys(), key=snr_sort_key)
    snr_labels = [k.replace("snr_", "") for k in snr_keys]

    plt.figure(figsize=(10, 6))

    # plot each language
    per_matrix = []
    for lang, values in all_data.items():
        pers = [values[k] for k in snr_keys]
        per_matrix.append(pers)
        plt.plot(range(len(pers)), pers, marker="o", label=lang)

    # compute and plot cross-language mean
    if len(per_matrix) > 1:
        import statistics
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
