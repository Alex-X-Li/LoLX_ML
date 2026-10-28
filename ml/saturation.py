"""Digitizer saturation for LoLX Run-1: emulate it on simulated events and build saturation-aware model inputs.

Real data (lolx_ana PE files) report each SiPM MERCI channel as
  sat 0  unsaturated, pe_rec = measured PE
  sat 1  clipped, recovered best estimate pe_rec (+-pe_err)
  sat 2  deeply clipped, interval only: pe_lo (clipped charge, a hard lower bound) <= true <= pe_hi (may be inf)

SaturationEmulator turns simulated detected photons into the same fields:
  1. per-SiPM counts [n, 81] -> MERCI channels, SiPMs that are turned off removed (saturation acts on the channel sum)
  2. per channel, x = true PE / saturation level. The level is |clip voltage| / (pulse height per PE), measured per
     channel from data (yaml/saturation_calib.yaml, threshold_pe) and smeared per pulse by the measured spread.
  3. clipped (x > 1): plateau = a + tau ln x (+ spread), fitted on data sat 1 channels.
     plateau <= limit (150 ns FBK, 125 ns HPK) -> sat 1: pe_rec = true * (1 + bias + rel_err * g)
     longer                                     -> sat 2: a data sat 2 channel with a similar plateau is drawn, and its
        clipped charge / level and pe_hi / pe_lo are used (deep clips grow more slowly than the sat 1 fit predicts).
All numbers come from yaml/saturation_calib.yaml + yaml/saturation_hi_over_lo.npz, written by
scripts/calibrate_saturation.py from the lolx_ana PE files.

build_features turns either source (simulation or data) into the inverse-model input, so both see the same thing.
"""
import csv
from pathlib import Path

import numpy as np
import yaml

ML_DIR = Path(__file__).resolve().parent
DEFAULT_CALIB = ML_DIR.parent / "yaml" / "saturation_calib.yaml"
DEFAULT_MAPPING = ML_DIR / "SiPMid_vs_chans.csv"
RUN1_BAD_SIPMS = [5, 6, 7, 8, 11, 25, 36, 42, 44, 64, 66]   # turned off: no photons, add nothing to their channel sum
N_MERCI = 32          # per-channel arrays in the lolx_ana PE files are [n, 32]
PMT_MERCI = 31
LIVE = [c for c in range(26) if c not in (4, 18)]          # SiPM MERCI channels with live SiPMs
LOG_LIGHT_CENTER, LOG_LIGHT_SCALE = 3.0, 0.5


def sipm_to_merci_matrix(mapping_csv=DEFAULT_MAPPING, bad_sipm=RUN1_BAD_SIPMS, n_sipm_ids=81):
    """[n_sipm_ids, 32] 0/1 matrix: column c sums the live SiPMs read out by MERCI channel c (PMT -> 31)."""
    off = set(bad_sipm or [])
    matrix = np.zeros((n_sipm_ids, N_MERCI))
    with open(mapping_csv) as handle:
        for row in csv.DictReader(handle):
            sipm_id = int(row["SiPM ID"])
            if sipm_id not in off and sipm_id <= n_sipm_ids:
                matrix[sipm_id - 1, int(row["MERCI Channel #"])] = 1.0
    return matrix


class SaturationEmulator:
    def __init__(self, calib_path=DEFAULT_CALIB, bad_sipm=RUN1_BAD_SIPMS, mapping_csv=DEFAULT_MAPPING, seed=None):
        calib_path = Path(calib_path)
        c = yaml.safe_load(calib_path.read_text())
        hi_lo = np.load(calib_path.with_name(c["hi_over_lo_file"]))
        self.calib = c
        self.live = list(c["live_sipm_merci"])
        self.to_merci = sipm_to_merci_matrix(mapping_csv, bad_sipm)
        self.bias_range = c["sat1_bias"]
        self.rng = np.random.default_rng(seed)

        group_of = {ch: g for g, chans in c["groups"].items() for ch in chans}
        self.groups = [group_of[ch] for ch in self.live]

        def per_channel(fn):
            return np.array([fn(ch, g) for ch, g in zip(self.live, self.groups)], dtype=float)

        self.threshold = per_channel(lambda ch, g: c["threshold_pe"][ch])
        self.thr_sigma = per_channel(lambda ch, g: c["threshold_rel_sigma"][g])
        self.pl_a = per_channel(lambda ch, g: c["plateau"][g]["a_ns"])
        self.pl_tau = per_channel(lambda ch, g: c["plateau"][g]["tau_ns"])
        self.pl_sigma = per_channel(lambda ch, g: c["plateau"][g]["sigma_ns"])
        self.pl_limit = per_channel(lambda ch, g: c["plateau_limit_ns"][g])
        self.clip_b = per_channel(lambda ch, g: c["clip"][g]["b"])
        self.clip_sigma = per_channel(lambda ch, g: c["clip"][g]["sigma"])
        self.rel_err = per_channel(lambda ch, g: c["sat1"][g]["rel_err_median"])
        self.hi_over_lo = {g: (hi_lo[f"{g}_plateau"], hi_lo[f"{g}_lo_over_threshold"], hi_lo[f"{g}_ratio"])
                           for g in set(self.groups)}
        self.hi_neighbours = 100   # draw from the 100 data channels closest in plateau length

    def merci_pe(self, sipm_counts):
        """[n, 81] detected photons per SiPM ID -> ([n, 32] true PE per MERCI channel, [n] PMT PE)."""
        merci = np.asarray(sipm_counts, dtype=float) @ self.to_merci
        return merci, merci[:, PMT_MERCI].copy()

    def saturate(self, pe_true, noise_scale=1.0):
        """Apply saturation to [n, 32] true PE per MERCI channel. Returns lolx_ana-style fields (live channels only
        are saturated; dead/unused channels are NaN like in the data)."""
        rng = self.rng
        pe_true = np.asarray(pe_true, dtype=float)
        n = len(pe_true)
        shape = (n, N_MERCI)
        out = {
            "pe_rec": np.full(shape, np.nan), "pe_err": np.full(shape, np.nan), "pe_meas": np.full(shape, np.nan),
            "pe_lo": np.full(shape, np.nan), "pe_hi": np.full(shape, np.nan), "plateau": np.full(shape, np.nan),
            "sat": np.zeros(shape, dtype=np.int8),
        }
        pe = pe_true[:, self.live]
        k = len(self.live)

        threshold = self.threshold * (1.0 + self.thr_sigma * rng.standard_normal((n, k)))
        x = pe / threshold
        clipped = x > 1.0
        lx = np.log(np.where(clipped, x, 1.0))
        plateau = np.maximum(self.pl_a + self.pl_tau * lx + self.pl_sigma * rng.standard_normal((n, k)), 0.0)
        sat = np.where(clipped, np.where(plateau <= self.pl_limit, 1, 2), 0).astype(np.int8)

        meas = threshold * (1.0 + self.clip_b * lx) * (1.0 + noise_scale * self.clip_sigma * rng.standard_normal((n, k)))
        meas = np.where(clipped, np.minimum(meas, pe), pe)       # clipped charge never exceeds the true charge

        frac_of_limit = np.clip(plateau / self.pl_limit, 0.0, 1.0)
        bias = self.bias_range[0] + (self.bias_range[1] - self.bias_range[0]) * frac_of_limit
        rec1 = pe * (1.0 + bias + noise_scale * self.rel_err * rng.standard_normal((n, k)))

        # deep clips: clipped charge and upper bound from a data sat 2 channel with a similar plateau length
        ratio_all = np.zeros_like(meas)
        for g in sorted(set(self.groups)):
            cols = np.array([gg == g for gg in self.groups])
            sub = sat[:, cols] == 2
            data_plateau, data_lo_thr, data_ratio = self.hi_over_lo[g]
            pos = np.searchsorted(data_plateau, plateau[:, cols][sub])
            nn = self.hi_neighbours
            j = np.clip(pos - nn // 2, 0, len(data_ratio) - nn) + rng.integers(0, nn, size=len(pos))
            block_meas = meas[:, cols]
            nominal = np.broadcast_to(self.threshold[cols], block_meas.shape)[sub]
            block_meas[sub] = np.minimum(nominal * data_lo_thr[j], pe[:, cols][sub])
            meas[:, cols] = block_meas
            block_ratio = ratio_all[:, cols]
            block_ratio[sub] = data_ratio[j]
            ratio_all[:, cols] = block_ratio

        rec = np.where(sat == 0, pe, np.where(sat == 1, rec1, np.nan))
        lo = np.where(sat == 0, pe, np.where(sat == 1, np.minimum(meas, rec1), meas))
        hi = np.where(sat == 0, pe, np.where(sat == 1, rec1, meas * ratio_all))

        out["pe_rec"][:, self.live] = rec
        out["pe_err"][:, self.live] = np.where(sat == 1, self.rel_err * rec1, np.where(sat == 0, 0.0, np.nan))
        out["pe_meas"][:, self.live] = meas
        out["pe_lo"][:, self.live] = lo
        out["pe_hi"][:, self.live] = hi
        out["plateau"][:, self.live] = np.where(clipped, plateau, np.nan)
        out["sat"][:, self.live] = sat
        return out


def build_features(sat, pe_rec, pe_lo, pe_pmt):
    """Inverse-model inputs from lolx_ana-style fields (simulation or data). sat, pe_rec, pe_lo: [n, >=26] per MERCI
    channel (NaN allowed); pe_pmt: [n]. Returns (features, info); features["b"] is the Run-1 saturation-aware input:
    27 ratios (value / total; sat 2 at pe_lo; dead channels 0) + 48 flags (sat 1, sat 2 of the 24 live channels)
    + log light outside the sat 2 channels. "a" drops the log light; "c" masks the sat 2 channels instead."""
    sat = np.asarray(sat)[:, :26]
    value = np.nan_to_num(np.where(sat == 2, np.asarray(pe_lo)[:, :26], np.asarray(pe_rec)[:, :26]))
    v27 = np.concatenate([value, np.asarray(pe_pmt, dtype=float)[:, None]], 1)
    keep27 = np.concatenate([sat != 2, np.ones((len(sat), 1), bool)], 1)
    total = v27.sum(1, keepdims=True)
    light = (v27 * keep27).sum(1, keepdims=True)
    with np.errstate(divide="ignore", invalid="ignore"):
        ratio = np.where(total > 0, v27 / total, 0.0)
        ratio_masked = np.where(light > 0, v27 * keep27 / light, 0.0)
        log_light = (np.log10(np.maximum(light, 1.0)) - LOG_LIGHT_CENTER) / LOG_LIGHT_SCALE
    flags = np.concatenate([sat[:, LIVE] == 1, sat[:, LIVE] == 2], 1).astype(np.float32)
    feats = {
        "a": np.concatenate([ratio, flags], 1),
        "b": np.concatenate([ratio, flags, log_light], 1),
        "c": np.concatenate([ratio_masked, flags, log_light], 1),
    }
    info = {"ok": light[:, 0] > 0, "sat": sat, "v27": v27, "keep27": keep27,
            "ratio": ratio.astype(np.float32), "flags": flags, "light": light[:, 0]}
    return {k: v.astype(np.float32) for k, v in feats.items()}, info


def npe_central(v27, keep27, lce):
    """Emitted photons from the channels without sat 2: sum(measured) / sum(LCE) over those channels (the PMT always
    counts). v27, keep27 from build_features; lce [n, 27] from the forward model (FastPredictor(use_merci=True))."""
    return (v27 * keep27).sum(1) / (lce * keep27).sum(1)
