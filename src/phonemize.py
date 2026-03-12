import argparse
import json
import os
from pathlib import Path

from phonemizer.backend.espeak.wrapper import EspeakWrapper

EspeakWrapper.set_library(
    r"C:\Program Files\eSpeak NG\libespeak-ng.dll"
)

from phonemizer.backend import EspeakBackend
import yaml

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--lang", required=True)
    args = parser.parse_args()

    with open("params.yaml", encoding="utf-8") as f:
        params = yaml.safe_load(f)

    espeak_lang = params["data"][args.lang]["espeak_lang"]

    manifest_path = Path(args.manifest)
    output_path = Path(args.output)

    records = []
    texts = []

    with open(manifest_path, encoding="utf-8") as f:
        for line in f:
            r = json.loads(line)
            records.append(r)
            texts.append(r["ref_text"])

    backend = EspeakBackend(
        language=espeak_lang,
        preserve_punctuation=False,
        with_stress=False
    )

    phonemes = backend.phonemize(texts)

    for r, phon in zip(records, phonemes):
        r["ref_phon"] = phon.strip()

    os.makedirs(output_path.parent, exist_ok=True)

    tmp = str(output_path) + ".tmp"

    with open(tmp, "w", encoding="utf-8") as f:
        for r in records:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    os.replace(tmp, output_path)

    print(f"Phonemized {len(records)} records → {output_path}")


if __name__ == "__main__":
    main()