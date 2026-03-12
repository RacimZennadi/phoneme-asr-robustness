"""
add_noise.py
Stage 3 of the pipeline.
Takes the phonemized manifest (clean_phon.jsonl) and produces one noisy
manifest per SNR level defined in params.yaml.
For each SNR level:
 - a noisy copy of each wav is written to data/noisy/{lang}/snrXX/
 - a new manifest is written with updated wav_path and snr_db fields

Everything else in the manifest (utt_id, ref_text, ref_phon, ...) is unchanged, 
so downstream stages can still match utterances by utt_id.
"""
import argparse
import hashlib
import json
import os
import platform
import sys
from pathlib import Path

import yaml

# espeak-ng on Windows needs an explicit path to the DLL
if platform.system() == "Windows":
    from phonemizer.backend.espeak.wrapper import EspeakWrapper

# add src/ to path so noise_utils can be imported regardless of working dir
sys.path.insert(0, str(Path(__file__).resolve().parent))
from noise_utils import add_noise_to_file


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--lang",       required=True, help="e.g. 'en' or 'fr'")
    parser.add_argument("--manifest",   required=True, help="path to clean_phon.jsonl")
    parser.add_argument("--output_dir", required=True, help="where to write noisy_snrXX.jsonl files")
    args = parser.parse_args()

    # resolve project root from this file's location (works regardless of cwd)
    root   = Path(__file__).resolve().parent.parent
    params = yaml.safe_load((root / "params.yaml").read_text())

    if platform.system() == "Windows":
        EspeakWrapper.set_library(params["espeak"]["windows_dll"])

    snr_levels     = params["snr_levels"]          # e.g. [-5, 0, 5, 10, 20, 30, 40]
    manifest_path  = Path(args.manifest).resolve()
    output_dir     = Path(args.output_dir).resolve()
    noisy_wav_base = (root / params["data"][args.lang]["noisy_dir"]).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    # load all utterances from the input manifest into memory
    records = [json.loads(l) for l in manifest_path.read_text(encoding="utf-8").splitlines()]

    for snr in snr_levels:
        snr_tag  = f"snr{int(snr):+d}"           # e.g. "snr+10", "snr-5"
        wav_dir  = noisy_wav_base / snr_tag       # folder for noisy wavs at this SNR
        wav_dir.mkdir(parents=True, exist_ok=True)
        out_path = output_dir / f"noisy_{snr_tag}.jsonl"

        out_records = []
        for r in records:
            # derive a deterministic seed from utt_id so the same utterance
            # always gets the same noise, even if processing order changes
            seed       = int(hashlib.md5(r["utt_id"].encode()).hexdigest(), 16) % (2**32)
            noisy_path = (wav_dir / Path(r["wav_path"]).name).resolve()

            # write the noisy wav using the professor's function
            add_noise_to_file(r["wav_path"], str(noisy_path), snr_db=snr, seed=seed)

            # copy all fields from the clean record, only update wav_path and snr_db
            out_records.append({**r, "wav_path": noisy_path.as_posix(), "snr_db": snr})

        # atomic write: only replace the real file once everything is written
        tmp = out_path.with_suffix(out_path.suffix + ".tmp")
        with open(tmp, "w", encoding="utf-8") as f:
            for r in out_records:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")
        os.replace(tmp, out_path)
        print(f"  SNR={snr:+.0f} → {out_path.name}")

    print(f"Done: {len(snr_levels)} noisy manifests → {output_dir}")


if __name__ == "__main__":
    main()
