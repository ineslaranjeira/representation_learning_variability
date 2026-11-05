"""
Wheel states over learning -- shared code for the notebooks in this folder.

Input: the 200 Hz wheel design matrices built by ../wheel_design_matrices.py
(paper-individuality/data/training/training_wheel_200Hz/, one per training session).

Features, per 5 ms sample: log Morlet-wavelet amplitude of wheel velocity in FREQS bands.
No per-session normalisation: amplitudes stay in rad/s, so a mouse that moves more or faster
than in its last session shows it.
Direction: a wavelet amplitude is blind to sign (a left and a right turn give the same
spectrum). With RECTIFY the transform runs separately on the rightward part max(v, 0) and the
leftward part max(-v, 0), so the features carry direction. Direction is in wheel coordinates;
stimulus/choice-relative coding is left for the analysis.

Glitches: the raw encoder occasionally jumps by radians within one sample (a physical
impossibility, e.g. 2.9 rad in 5 ms). Samples within GLITCH_PAD_S of a jump are masked.
"""
import pathlib
import numpy as np
import pandas as pd
from scipy.fft import fft, ifft, fftshift

HERE = pathlib.Path(__file__).resolve().parent
DM_DIR = HERE.parent.parent / 'paper-individuality' / 'data' / 'training' / 'training_wheel_200Hz'

FS = 200                                    # Hz, design-matrix rate
FREQS = 0.5 * 2 ** (np.arange(11) / 2)      # 0.5 .. 16 Hz, half-octave steps (the paper used 0.5-8, octaves)
OMEGA0 = 5                                  # Morlet parameter, as the paper pipeline
LOG_FLOOR = 0.005                           # log(amp + floor); still wheel ~1e-4, moving ~0.05-0.3
GLITCH_RAD = 0.3                            # |dposition| in one 5 ms sample (= 60 rad/s)
GLITCH_PAD_S = 1.0


def manifest():
    return pd.read_csv(DM_DIR / 'manifest.csv')


def load_session(eid, mouse):
    dm = pd.read_parquet(DM_DIR / f'design_matrix_{eid}_{mouse}')
    trials = pd.read_parquet(DM_DIR / f'trials_{eid}_{mouse}')
    return dm, trials


def glitch_mask(position, fs=FS, thresh=GLITCH_RAD, pad_s=GLITCH_PAD_S):
    """True where the sample is within pad_s of an encoder jump."""
    jumps = np.flatnonzero(np.abs(np.diff(position)) > thresh)
    bad = np.zeros(len(position), bool)
    pad = int(pad_s * fs)
    for j in jumps:
        bad[max(j - pad, 0):j + pad + 2] = True
    return bad


def morlet_amplitude(x, freqs=FREQS, omega0=OMEGA0, dt=1 / FS):
    """Wavelet amplitude (n_freqs, n), same transform and normalisation as
    segmentation_functions.fast_wavelet_morlet_convolution_parallel, without keeping the complex
    coefficients (too large at 200 Hz)."""
    x = np.asarray(x, np.float64)
    M = len(x)
    N = M + M % 2
    xp = np.concatenate((np.zeros(N // 2), x, np.zeros(N - M), np.zeros(N // 2)))
    Np = len(xp)
    scales = (omega0 + np.sqrt(2 + omega0 ** 2)) / (4 * np.pi * freqs)
    omega = 2 * np.pi * np.arange(-Np // 2, Np // 2) / (Np * dt)
    x_hat = fftshift(fft(xp))
    idx = np.arange(N // 2, N // 2 + M)
    norm = np.pi ** -0.25 * np.exp(0.25 * (omega0 - np.sqrt(omega0 ** 2 + 2)) ** 2)
    amp = np.empty((len(freqs), M), np.float32)
    for i, s in enumerate(scales):
        m = np.pi ** (-1 / 4) * np.exp(-0.5 * (-omega * s - omega0) ** 2)
        q = ifft(m * x_hat) * np.sqrt(s)
        amp[i] = np.abs(q[idx]) * norm / np.sqrt(2 * s)
    return amp


def feature_names(rectify=True, freqs=FREQS):
    parts = ('R', 'L') if rectify else ('',)
    return [f'{p}{f:g}' for p in parts for f in freqs]


def wavelet_features(velocity, rectify=True, log=True, freqs=FREQS):
    """(n, n_features) float32: log wavelet amplitudes of velocity (rightward then leftward
    parts with rectify)."""
    v = np.asarray(velocity, np.float64)
    parts = (np.maximum(v, 0), np.maximum(-v, 0)) if rectify else (v,)
    X = np.concatenate([morlet_amplitude(p, freqs) for p in parts]).T
    return np.log(X + LOG_FLOOR) if log else X


def session_features(eid, mouse, rectify=True):
    """Features of one session + the valid mask (not near a glitch)."""
    dm, trials = load_session(eid, mouse)
    X = wavelet_features(dm['velocity'].values, rectify=rectify)
    valid = ~glitch_mask(dm['position'].values)
    return dm, trials, X, valid
