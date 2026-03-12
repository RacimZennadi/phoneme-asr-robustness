"""
phonemize.py :  Stage 2 of the pipeline.
Reads the base manifest (clean.jsonl) and adds the ref_phon field to each
utterance by converting ref_text to IPA phonemes using espeak-ng.
Output is a new manifest (clean_phon.jsonl) with ref_phon populated.
"""
import argparse
import json
import os
import platform
from pathlib import Path

import yaml
from phonemizer.backend import EspeakBackend

# espeak-ng on Windows needs an explicit path to the DLL
if platform.system() == "Windows":
    from phonemizer.backend.espeak.wrapper import EspeakWrapper
    _params = yaml.safe_load(open("params.yaml"))
    EspeakWrapper.set_library(_params["espeak"]["windows_dll"])


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--output",   required=True)
    parser.add_argument("--lang",     required=True)
    args = parser.parse_args()

    params      = yaml.safe_load(open("params.yaml"))
    espeak_lang = params["data"][args.lang]["espeak_lang"]

    manifest_path = Path(args.manifest)
    output_path   = Path(args.output)

    records = [json.loads(l) for l in manifest_path.read_text(encoding="utf-8").splitlines()]
    texts   = [r["ref_text"] for r in records]

    # with_stress=False keeps the phoneme set consistent with wav2vec2 output
    backend  = EspeakBackend(espeak_lang, preserve_punctuation=False, with_stress=False)
    phonemes = backend.phonemize(texts)

    for r, phon in zip(records, phonemes):
        r["ref_phon"] = phon.strip()

    output_path.parent.mkdir(parents=True, exist_ok=True)

    tmp = str(output_path) + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        for r in records:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    os.replace(tmp, output_path)

    print(f"Phonemized {len(records)} records → {output_path}")


if __name__ == "__main__":
    main()
