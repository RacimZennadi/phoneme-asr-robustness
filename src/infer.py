""" 
infer.py
Stage 4 of the pipeline.
Runs the wav2vec2 phoneme recognizer on every manifest in manifest_dir
(clean + all noisy variants) and adds a hyp_phon field to each utterance.

The model (facebook/wav2vec2-lv-60-espeak-cv-ft) takes raw 16kHz mono audio
and outputs a sequence of IPA phoneme tokens — the same format as ref_phon.
"""
import argparse
import json
import os
import platform
from pathlib import Path

import soundfile as sf
import torch
from transformers import Wav2Vec2Processor, Wav2Vec2ForCTC
import yaml

# the model's tokenizer internally uses espeak, so the DLL fix is needed here 
if platform.system() == "Windows":
    from phonemizer.backend.espeak.wrapper import EspeakWrapper


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest_dir", required=True, help="folder with clean_phon.jsonl + noisy_snr*.jsonl")
    parser.add_argument("--output_dir",   required=True, help="where to write pred_*.jsonl files")
    args = parser.parse_args()

    root   = Path(__file__).resolve().parent.parent
    params = yaml.safe_load((root / "params.yaml").read_text())

    if platform.system() == "Windows":
        EspeakWrapper.set_library(params["espeak"]["windows_dll"])

    model_name   = params["model"]["name"]
    target_sr    = params["model"]["sample_rate"]   # must be 16000 for this model
    manifest_dir = Path(args.manifest_dir).resolve()
    output_dir   = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    # load the model once — reused for every manifest
    print(f"Loading {model_name} ...")
    processor = Wav2Vec2Processor.from_pretrained(model_name)
    model     = Wav2Vec2ForCTC.from_pretrained(model_name)
    model.eval()
    print("Model loaded.")

    # process clean first, then noisy variants in SNR order
    manifests = [manifest_dir / "clean_phon.jsonl"] + \
                sorted(manifest_dir.glob("noisy_snr*.jsonl"))

    for manifest_path in manifests:
        if not manifest_path.exists():
            print(f"  Skipping missing: {manifest_path.name}")
            continue

        records     = [json.loads(l) for l in manifest_path.read_text(encoding="utf-8").splitlines()]
        out_records = []

        for r in records:
            wav        = Path(r["wav_path"]).resolve()
            signal, sr = sf.read(str(wav))

            # the model strictly requires 16kHz mono — fail loudly if not
            if sr != target_sr:
                raise ValueError(f"Expected {target_sr}Hz, got {sr} in {wav}")
            if signal.ndim != 1:
                raise ValueError(f"Expected mono audio in {wav}")

            # run the model: audio → logits → token ids → phoneme string
            inputs   = processor(signal, sampling_rate=target_sr, return_tensors="pt").input_values
            with torch.no_grad():
                logits = model(inputs).logits
            pred_ids = torch.argmax(logits, dim=-1)   # greedy decoding
            hyp_phon = processor.batch_decode(pred_ids)[0]

            # copy all existing fields and add the model's prediction
            out_records.append({**r, "hyp_phon": hyp_phon})

        # derive output filename from input:
        # clean_phon.jsonl → pred_clean.jsonl
        # noisy_snr+10.jsonl → pred_snr+10.jsonl
        stem     = manifest_path.stem \
                       .replace("clean_phon", "pred_clean") \
                       .replace("noisy_",     "pred_")
        out_path = output_dir / f"{stem}.jsonl"

        tmp = out_path.with_suffix(".jsonl.tmp")
        with tmp.open("w", encoding="utf-8") as f:
            for rec in out_records:
                f.write(json.dumps(rec, ensure_ascii=False) + "\n")
        os.replace(tmp, out_path)
        print(f"  {manifest_path.name} → {out_path.name}")

    print(f"Done → {output_dir}")


if __name__ == "__main__":
    main()
