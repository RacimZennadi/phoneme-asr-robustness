# src/evaluate.py
import argparse, json
from pathlib import Path
import editdistance
import matplotlib.pyplot as plt
import yaml


def normalize(phon: str) -> str:
    """Strip spaces and lowercase — compare at character level."""
    return phon.replace(" ", "").strip()

def compute_per(ref: str, hyp: str) -> float:
    ref_chars = list(normalize(ref))   # each IPA char is one unit
    hyp_chars = list(normalize(hyp))
    if not ref_chars:
        return 0.0
    return editdistance.eval(ref_chars, hyp_chars) / len(ref_chars)



def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest_dir", required=True)
    parser.add_argument("--lang",         required=True)
    parser.add_argument("--metrics",      required=True)
    parser.add_argument("--figure",       required=True)
    args = parser.parse_args()

    root         = Path(__file__).resolve().parent.parent
    manifest_dir = Path(args.manifest_dir).resolve()
    metrics_path = Path(args.metrics).resolve()
    figure_path  = Path(args.figure).resolve()
    metrics_path.parent.mkdir(parents=True, exist_ok=True)
    figure_path.parent.mkdir(parents=True, exist_ok=True)

    # collect: pred_clean.jsonl + pred_snr*.jsonl
    pred_files = sorted(manifest_dir.glob("pred_snr*.jsonl"))
    clean_pred = manifest_dir / "pred_clean.jsonl"
    if clean_pred.exists():
        pred_files = [clean_pred] + list(pred_files)

    snr_to_per = {}
    for path in pred_files:
        records  = [json.loads(l) for l in path.read_text(encoding="utf-8").splitlines()]
        snr      = records[0]["snr_db"]
        mean_per = sum(compute_per(r["ref_phon"], r["hyp_phon"]) for r in records) / len(records)
        snr_to_per[snr] = round(mean_per, 4)
        label = f"SNR={snr:+.0f}" if snr is not None else "clean"
        print(f"  {label:>12}  PER = {mean_per:.4f}")

    # sort numerically, clean at the far right
    snrs   = sorted(snr_to_per, key=lambda x: x if x is not None else float("inf"))
    pers   = [snr_to_per[s] for s in snrs]
    labels = [f"{int(s):+d}" if s is not None else "clean" for s in snrs]

    # write metrics.json
    metrics_out = {args.lang: {f"snr_{l}": p for l, p in zip(labels, pers)}}
    tmp = metrics_path.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(metrics_out, indent=2), encoding="utf-8")
    tmp.rename(metrics_path)
    print(f"Metrics → {metrics_path}")

    # write figure
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
