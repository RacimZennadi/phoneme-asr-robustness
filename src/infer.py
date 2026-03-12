# src/infer.py
import argparse, json, platform
from pathlib import Path
import soundfile as sf
import torch
from transformers import Wav2Vec2Processor, Wav2Vec2ForCTC
import yaml

if platform.system() == "Windows":
    from phonemizer.backend.espeak.wrapper import EspeakWrapper


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest_dir", required=True)
    parser.add_argument("--output_dir",   required=True)
    args = parser.parse_args()

    root   = Path(__file__).resolve().parent.parent
    params = yaml.safe_load((root / "params.yaml").read_text())

    if platform.system() == "Windows":
        EspeakWrapper.set_library(params["espeak"]["windows_dll"])

    model_name   = params["model"]["name"]
    target_sr    = params["model"]["sample_rate"]
    manifest_dir = Path(args.manifest_dir).resolve()
    output_dir   = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"Loading {model_name} ...")
    processor = Wav2Vec2Processor.from_pretrained(model_name)
    model     = Wav2Vec2ForCTC.from_pretrained(model_name)
    model.eval()
    print("Model loaded.")

    manifests = [manifest_dir / "clean_phon.jsonl"] + \
                sorted(manifest_dir.glob("noisy_snr*.jsonl"))

    for manifest_path in manifests:
        if not manifest_path.exists():
            print(f"  Skipping missing: {manifest_path.name}")
            continue

        records     = [json.loads(l) for l in manifest_path.read_text(encoding="utf-8").splitlines()]
        out_records = []

        for r in records:
            wav = Path(r["wav_path"]).resolve()
            signal, sr = sf.read(str(wav))

            if sr != target_sr:
                raise ValueError(f"Expected {target_sr}Hz, got {sr} in {wav}")
            if signal.ndim != 1:
                raise ValueError(f"Expected mono audio in {wav}")

            inputs = processor(
                signal, sampling_rate=target_sr, return_tensors="pt"
            ).input_values
            with torch.no_grad():
                logits = model(inputs).logits
            pred_ids = torch.argmax(logits, dim=-1)
            hyp_phon = processor.batch_decode(pred_ids)[0]
            out_records.append({**r, "hyp_phon": hyp_phon})

        stem     = manifest_path.stem \
                       .replace("clean_phon", "pred_clean") \
                       .replace("noisy_",     "pred_")
        out_path = output_dir / f"{stem}.jsonl"

        tmp = out_path.with_suffix(".jsonl.tmp")
        with tmp.open("w", encoding="utf-8") as f:
            for rec in out_records:
                f.write(json.dumps(rec, ensure_ascii=False) + "\n")
        tmp.rename(out_path)
        print(f"  {manifest_path.name} → {out_path.name}")

    print(f"Done → {output_dir}")


if __name__ == "__main__":
    main()
