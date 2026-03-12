import argparse
import hashlib
import json
import os
from pathlib import Path

import soundfile as sf
import yaml


def md5(path):
    """Compute MD5 without loading full file in memory."""
    hash_md5 = hashlib.md5()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(4096), b""):
            hash_md5.update(chunk)
    return hash_md5.hexdigest()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--lang", required=True)
    args = parser.parse_args()

    with open("params.yaml") as f:
        params = yaml.safe_load(f)

    cfg = params["data"][args.lang]

    raw_dir = Path(cfg["raw_dir"])
    manifest_dir = Path(cfg["manifest_dir"])
    transcript_file = Path(cfg["transcript_file"])

    records = []

    with open(transcript_file, encoding="utf-8") as f:
        for line in f:
            stem, ref_text = line.strip().split("\t", 1)

            wav_path = raw_dir / f"{stem}.wav"

            if not wav_path.exists():
                print(f"Warning: missing {wav_path}")
                continue

            info = sf.info(str(wav_path))

            records.append({
                "utt_id": f"{args.lang}_{stem}",
                "lang": args.lang,
                "wav_path": wav_path.as_posix(),   # portable path
                "ref_text": ref_text,
                "ref_phon": None,
                "sr": info.samplerate,
                "duration_s": round(info.duration, 3),
                "snr_db": None,
                "audio_md5": md5(wav_path),
            })

    out_path = manifest_dir / "clean.jsonl"
    os.makedirs(out_path.parent, exist_ok=True)

    tmp = str(out_path) + ".tmp"

    with open(tmp, "w", encoding="utf-8") as f:
        for r in records:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    os.replace(tmp, out_path)

    print(f"Wrote {len(records)} records to {out_path}")


if __name__ == "__main__":
    main()