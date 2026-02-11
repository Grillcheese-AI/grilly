"""Functional FFT helpers backed by Grilly compute kernels."""

import numpy as np


def _get_backend():
    """Get compute backend"""
    from grilly import Compute

    return Compute()


def fft(input: np.ndarray) -> np.ndarray:
    """
    Fast Fourier Transform
    Uses: fft-bitrev.glsl, fft-butterfly.glsl

    Args:
        input: Input signal (real or complex)

    Returns:
        FFT output (complex)
    """
    _get_backend()
    # Note: May need to implement FFT in backend if not already exposed
    # CPU fallback for now
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
    _get_backend()
    # Note: May need to implement IFFT in backend if not already exposed
    # CPU fallback for now
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
    _get_backend()
    # Note: May need to implement in backend if not already exposed
    # CPU fallback for now
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
    _get_backend()
    # Note: May need to implement in backend if not already exposed
    # CPU fallback for now
    return np.abs(input) ** 2
