"""Functional FFT helpers backed by Grilly compute kernels."""

import numpy as np


def fft(input: np.ndarray) -> np.ndarray:
    """
    Fast Fourier Transform
    Uses: fft-bitrev.glsl, fft-butterfly.glsl

    Args:
        input: Input signal (real or complex)

    Returns:
        FFT output (complex)
    """
    # CPU fallback
    return np.fft.fft(input)


def ifft(input: np.ndarray) -> np.ndarray:
    """
    Inverse Fast Fourier Transform
    Uses: fft-bitrev.glsl, fft-butterfly.glsl

    Args:
        input: FFT output (complex)

    Returns:
        Reconstructed signal
    """
    # CPU fallback
    return np.fft.ifft(input)


def fft_magnitude(input: np.ndarray) -> np.ndarray:
    """
    FFT magnitude spectrum
    Uses: fft-magnitude.glsl

    Args:
        input: FFT output (complex)

    Returns:
        Magnitude spectrum
    """
    # CPU fallback
    return np.abs(input)


def fft_power_spectrum(input: np.ndarray) -> np.ndarray:
    """
    FFT power spectrum
    Uses: fft-power-spectrum.glsl

    Args:
        input: FFT output (complex)

    Returns:
        Power spectrum
    """
    # CPU fallback
    return np.abs(input) ** 2
