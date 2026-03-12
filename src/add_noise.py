# src/add_noise.py
import argparse, hashlib, json, platform, sys
from pathlib import Path
import yaml

if platform.system() == "Windows":
    from phonemizer.backend.espeak.wrapper import EspeakWrapper

sys.path.insert(0, str(Path(__file__).resolve().parent))
from noise_utils import add_noise_to_file


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--lang",       required=True)
    parser.add_argument("--manifest",   required=True)
    parser.add_argument("--output_dir", required=True)
    args = parser.parse_args()

    root   = Path(__file__).resolve().parent.parent
    params = yaml.safe_load((root / "params.yaml").read_text())

    if platform.system() == "Windows":
        EspeakWrapper.set_library(params["espeak"]["windows_dll"])

    snr_levels    = params["snr_levels"]
    manifest_path = Path(args.manifest).resolve()
    output_dir    = Path(args.output_dir).resolve()
    noisy_wav_base = (root / params["data"][args.lang]["noisy_dir"]).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    records = [json.loads(l) for l in manifest_path.read_text(encoding="utf-8").splitlines()]

    for snr in snr_levels:
        snr_tag     = f"snr{int(snr):+d}"
        wav_dir     = noisy_wav_base / snr_tag
        wav_dir.mkdir(parents=True, exist_ok=True)
        out_path    = output_dir / f"noisy_{snr_tag}.jsonl"
        out_records = []

        for r in records:
            seed       = int(hashlib.md5(r["utt_id"].encode()).hexdigest(), 16) % (2**32)
            noisy_path = (wav_dir / Path(r["wav_path"]).name).resolve()

            add_noise_to_file(
                input_wav  = r["wav_path"],
                output_wav = str(noisy_path),
                snr_db     = snr,
                seed       = seed,
            )
            out_records.append({**r,
                "wav_path": noisy_path.as_posix(),
                "snr_db":   snr,
            })

        # write output safely
        tmp = out_path.with_suffix(out_path.suffix + ".tmp")
        with open(tmp, "w", encoding="utf-8") as f:
            for r in out_records:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")

        # Windows-safe replacement
        tmp.replace(out_path)   # overwrites if out_path exists
        print(f"  SNR={snr:+.0f} → {out_path}")

    print(f"Done: {len(snr_levels)} noisy manifests → {output_dir}")


if __name__ == "__main__":
    main()
