"""
noise_utils.py
Utility functions for adding Gaussian white noise to audio signals.
- add_noise: pure function, no file I/O, easy to unit test
- add_noise_to_file: handles reading/writing, calls add_noise internally
"""
import numpy as np
import soundfile as sf


def add_noise(
    signal: np.ndarray,
    snr_db: float,
    rng: np.random.Generator,
) -> np.ndarray:
    # SNR (dB) = 10 * log10(signal_power / noise_power)
    # so noise_power = signal_power / 10^(snr_db/10)
    signal_power = np.mean(signal ** 2)
    snr_linear   = 10 ** (snr_db / 10)
    noise_power  = signal_power / snr_linear

    # Gaussian white noise with the computed power
    noise = rng.normal(
        loc=0.0,
        scale=np.sqrt(noise_power),  # std dev = sqrt(power)
        size=signal.shape,
    )
    return signal + noise


def add_noise_to_file(
    input_wav: str,
    output_wav: str,
    snr_db: float,
    seed: int | None = None,
) -> None:
    signal, sr = sf.read(input_wav)

    if signal.ndim != 1:
        raise ValueError("Only mono audio is supported")

    # fixing the seed means the same input + SNR always gives the same output
    rng          = np.random.default_rng(seed)
    noisy_signal = add_noise(signal, snr_db, rng)

    sf.write(output_wav, noisy_signal, sr)
