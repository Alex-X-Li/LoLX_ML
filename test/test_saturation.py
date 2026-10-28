import unittest

import numpy as np

from ml.saturation import LIVE, SaturationEmulator, build_features


class SaturationEmulatorTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.emu = SaturationEmulator(seed=1)
        rng = np.random.default_rng(2)
        # per-SiPM detected photons; a few events with bright SiPMs so every saturation state appears
        counts = rng.poisson(rng.uniform(1, 40, (4000, 81)))
        counts[:, rng.integers(0, 80, 4000)] += rng.integers(0, 2000, 4000)[:, None]
        cls.pe_true, cls.pmt = cls.emu.merci_pe(counts)
        cls.res = cls.emu.saturate(cls.pe_true)

    def test_calibration_covers_every_live_channel(self):
        self.assertEqual(sorted(self.emu.calib["threshold_pe"]), LIVE)
        self.assertTrue(all(t > 0 for t in self.emu.calib["threshold_pe"].values()))

    def test_turned_off_sipms_add_nothing(self):
        self.assertEqual(self.pe_true[:, 4].max(), 0)    # SiPM 25 is off
        self.assertEqual(self.pe_true[:, 18].max(), 0)   # SiPMs 5-8 are off
        self.assertEqual(self.emu.to_merci[:, 11].sum(), 2)   # SiPMs 42, 44 of channel 11 are off

    def test_saturation_fields_are_consistent(self):
        sat = self.res["sat"][:, LIVE]
        rec, lo, hi = (self.res[k][:, LIVE] for k in ("pe_rec", "pe_lo", "pe_hi"))
        true = self.pe_true[:, LIVE]
        self.assertTrue({0, 1, 2} <= set(np.unique(sat)))
        self.assertTrue(np.allclose(rec[sat == 0], true[sat == 0]))
        self.assertTrue(np.isnan(rec[sat == 2]).all())
        self.assertTrue((lo[sat == 2] <= true[sat == 2] + 1e-6).all())   # clipped charge is a lower bound
        fin = np.isfinite(rec)
        self.assertTrue((lo[fin] <= rec[fin] + 1e-6).all() and (rec[fin] <= hi[fin] + 1e-6).all())
        self.assertTrue(np.isnan(self.res["pe_rec"][:, [4, 18]]).all())

    def test_features(self):
        feats, info = build_features(self.res["sat"], self.res["pe_rec"], self.res["pe_lo"], self.pmt)
        self.assertEqual(feats["b"].shape, (len(self.pmt), 27 + 48 + 1))
        ok = info["ok"]
        self.assertTrue(np.allclose(feats["b"][ok, :27].sum(1), 1.0, atol=1e-5))


if __name__ == "__main__":
    unittest.main()
