"""
build_manifest.py : Stage 1 of the pipeline. Reads raw audio files 
and their transcripts, computes metadata for each utterance (duration, sample rate, md5), 
and writes the base manifest.
"""

import argparse
import hashlib
import json
import os
from pathlib import Path

import soundfile as sf
import yaml


def md5(path):
    # chunked read to avoid loading large files into memory
    hash_md5 = hashlib.md5()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(4096), b""):
            hash_md5.update(chunk)
    return hash_md5.hexdigest()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--lang", required=True, help="language code, e.g. 'en'")
    args = parser.parse_args()

    params = yaml.safe_load(open("params.yaml"))
    cfg    = params["data"][args.lang]

    raw_dir         = Path(cfg["raw_dir"])
    manifest_dir    = Path(cfg["manifest_dir"])
    transcript_file = Path(cfg["transcript_file"])

    records = []
    for line in transcript_file.read_text(encoding="utf-8").splitlines():
        stem, ref_text = line.strip().split("\t", 1)
        wav_path = raw_dir / f"{stem}.wav"

        if not wav_path.exists():
            print(f"Warning: missing {wav_path}, skipping")
            continue

        info = sf.info(str(wav_path))
        records.append({
            "utt_id":     f"{args.lang}_{stem}",
            "lang":       args.lang,
            "wav_path":   wav_path.as_posix(),  # forward slashes on all OS
            "ref_text":   ref_text,
            "ref_phon":   None,                 # filled in by phonemize.py
            "sr":         info.samplerate,
            "duration_s": round(info.duration, 3),
            "snr_db":     None,                 # null = clean audio
            "audio_md5":  md5(wav_path),
        })

    out_path = manifest_dir / "clean.jsonl"
    out_path.parent.mkdir(parents=True, exist_ok=True)

    # write atomically: only rename once everything is written
    tmp = str(out_path) + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        for r in records:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    os.replace(tmp, out_path)

    print(f"Wrote {len(records)} records → {out_path}")


if __name__ == "__main__":
    main()
