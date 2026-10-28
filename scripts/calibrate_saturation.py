#!/usr/bin/env python3
"""Calibrate the saturation emulator (ml/saturation.py) from lolx_ana Run-1 PE files (pe_run*.h5, RECO_INPUT.md).

Writes yaml/saturation_calib.yaml and yaml/saturation_hi_over_lo.npz:
  threshold_pe[ch]       saturation level of MERCI channel ch in PE: |clip voltage| / median(amp / pe_rec), on
                         unsaturated pulses above 40 PE. The digitizer clips at -1.05 V, so this is the PE whose
                         pulse just reaches the clip level (FBK ~230-365 PE, HPK ~83-139 PE).
  threshold_rel_sigma[g] pulse-to-pulse spread of amp / pe_rec (robust sigma / median, median over the group's
                         channels): the onset of saturation is smeared by this relative amount.
  plateau[g]             plateau = a + tau * ln(PE / threshold) [ns] (+ Gaussian residual), fit on sat 1 channels.
                         lolx_ana marks a channel sat 2 when the plateau exceeds plateau_limit_ns.
  clip[g]                clipped charge pe_meas / threshold = 1 + b * ln(PE / threshold), fit on sat 1 (where pe_rec is
                         the true value); b = 1 for a pulse with an exponential tail. Used for sat 1 clipped charge.
  sat1[g]                relative error of pe_rec on sat 1 (pe_err / pe_rec), median and 84th percentile.
  hi_over_lo (npz)       every data sat 2 channel, per group, sorted by plateau: plateau, pe_lo / threshold and
                         pe_hi / pe_lo (inf kept). Deep clips are emulated by drawing a data channel with a similar
                         plateau, because there is no true value to fit there.

Usage:  python scripts/calibrate_saturation.py --pe-dir /path/to/lolx_ana/run1/pe
"""
import argparse
from pathlib import Path

import h5py
import numpy as np
import yaml

REPO = Path(__file__).resolve().parents[1]
# Cs 31060/31061 and Ba 31056/31057: PMT-trigger runs with the reference PMT gain (31062 has another PMT HV,
# 31059 the SiPM trigger); only SiPM channels are calibrated.
DEFAULT_RUNS = ["pe_run31060_c41.h5", "pe_run31061_c44.h5", "pe_run31056_c45.h5", "pe_run31057_c46.h5"]
CLIP_VOLTAGE = -1.05                                   # V, digitizer clip level (RECO_INPUT §5)
GROUPS = {"fbk_single": [0, 1, 2, 3, 5, 6, 7],          # one SiPM per channel
          "fbk_summed": [8, 9, 10, 11, 12, 13, 14, 15],  # four SiPMs summed
          "hpk": [16, 17, 19, 20, 21, 22, 23, 24, 25]}   # four SiPMs summed; channels 4 and 18 are dead
PLATEAU_LIMIT_NS = {"fbk_single": 150, "fbk_summed": 150, "hpk": 125}   # lolx_ana sat 1 / sat 2 boundary
SAT1_BIAS = [0.006, 0.023]                             # RECO_INPUT §5: +0.6 to +2.3 % bias of pe_rec, grows with plateau
LIVE = sorted(c for chans in GROUPS.values() for c in chans)


def load(pe_dir, runs):
    fields = {"amp": "amp", "rec": "pe_rec", "meas": "pe_meas", "lo": "pe_lo", "hi": "pe_hi",
              "err": "pe_err", "sat": "sat", "plateau": "plateau"}
    out = {k: [] for k in fields}
    sources = {}
    for name in runs:
        with h5py.File(Path(pe_dir) / name, "r") as f:
            ok = f["evflag"][:] == 0
            for k, field in fields.items():
                out[k].append(f[field][:][ok][:, LIVE])
            sources[name] = int(ok.sum())
    return {k: np.concatenate(v) for k, v in out.items()}, sources


def calibrate(d, sources):
    col = {ch: i for i, ch in enumerate(LIVE)}
    amp = np.abs(d["amp"])
    threshold, rel_sigma = {}, {}
    for ch, i in col.items():
        u = (d["sat"][:, i] == 0) & (d["rec"][:, i] > 40) & np.isfinite(amp[:, i])
        k = amp[u, i] / d["rec"][u, i]
        med = np.median(k)
        threshold[ch] = float(round(abs(CLIP_VOLTAGE) / med, 1))
        rel_sigma[ch] = float((np.percentile(k, 75) - np.percentile(k, 25)) / 1.349 / med)

    calib = {"source_runs": sources, "clip_voltage": CLIP_VOLTAGE, "groups": GROUPS, "live_sipm_merci": LIVE,
             "plateau_limit_ns": PLATEAU_LIMIT_NS, "sat1_bias": SAT1_BIAS,
             "hi_over_lo_file": "saturation_hi_over_lo.npz", "threshold_pe": threshold,
             "threshold_rel_sigma": {g: round(float(np.median([rel_sigma[c] for c in ch])), 4) for g, ch in GROUPS.items()},
             "plateau": {}, "clip": {}, "sat1": {}}
    hi_over_lo = {}
    for g, chans in GROUPS.items():
        idx = [col[c] for c in chans]
        t = np.array([threshold[c] for c in chans])
        s = d["sat"][:, idx]
        rec, meas, plat, err = d["rec"][:, idx], d["meas"][:, idx], d["plateau"][:, idx], d["err"][:, idx]

        m1 = s == 1
        x = (rec / t)[m1]
        p = plat[m1]
        ok = np.isfinite(x) & np.isfinite(p) & (x > 1) & (p > 0)
        tau, a = np.polyfit(np.log(x[ok]), p[ok], 1)
        resid = float(np.std(p[ok] - (a + tau * np.log(x[ok]))))
        calib["plateau"][g] = {"a_ns": float(round(a, 2)), "tau_ns": float(round(tau, 2)), "sigma_ns": round(resid, 2),
                               "sat2_above_pe_over_threshold": float(round(np.exp((PLATEAU_LIMIT_NS[g] - a) / tau), 3))}

        frac = (meas / rec)[m1]
        ok = ok & np.isfinite(frac)
        lx = np.log(x[ok])
        y = frac[ok] * x[ok] - 1.0                     # pe_meas / threshold - 1
        b = float(np.sum(y * lx) / np.sum(lx * lx))
        calib["clip"][g] = {"b": round(b, 4), "sigma": round(float(np.std(frac[ok] - (1 + b * lx) / x[ok])), 4)}

        rel = (err / rec)[m1]
        rel = rel[np.isfinite(rel) & (rel > 0)]
        calib["sat1"][g] = {"rel_err_median": round(float(np.median(rel)), 4),
                            "rel_err_p84": round(float(np.percentile(rel, 84)), 4)}

        m2 = s == 2
        r = (d["hi"][:, idx] / d["lo"][:, idx])[m2]
        pl = plat[m2]
        keep = ~np.isnan(r) & np.isfinite(pl)
        order = np.argsort(pl[keep])
        hi_over_lo[f"{g}_ratio"] = r[keep][order].astype(np.float32)
        hi_over_lo[f"{g}_plateau"] = pl[keep][order].astype(np.float32)
        hi_over_lo[f"{g}_lo_over_threshold"] = (d["lo"][:, idx] / t)[m2][keep][order].astype(np.float32)
        print(f"{g:11s} plateau = {a:6.1f} + {tau:6.1f} ln(PE/T) ns   sat2 above "
              f"{calib['plateau'][g]['sat2_above_pe_over_threshold']:.2f} T   clip b = {b:.3f}   "
              f"sat1 rel err {calib['sat1'][g]['rel_err_median']:.3f}")
    return calib, hi_over_lo


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--pe-dir", required=True, help="directory with the lolx_ana pe_run*.h5 files")
    ap.add_argument("--runs", nargs="+", default=DEFAULT_RUNS)
    ap.add_argument("--out-dir", default=str(REPO / "yaml"))
    args = ap.parse_args()

    d, sources = load(args.pe_dir, args.runs)
    calib, hi_over_lo = calibrate(d, sources)
    out = Path(args.out_dir)
    (out / "saturation_calib.yaml").write_text(
        "# generated by scripts/calibrate_saturation.py from lolx_ana Run-1 PE files; definitions in that script\n"
        + yaml.safe_dump(calib, sort_keys=False))
    np.savez(out / calib["hi_over_lo_file"], **hi_over_lo)
    print("threshold_pe:", {c: calib["threshold_pe"][c] for c in LIVE})
    print(f"-> {out / 'saturation_calib.yaml'}")


if __name__ == "__main__":
    main()
