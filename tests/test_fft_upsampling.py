"""The Nyquist adaptation must preserve the original likelihood analysis band."""
import unittest
import numpy as np
from scipy.signal import resample


class FourierUpsamplingTests(unittest.TestCase):
    def test_original_complex_frequency_bins_are_preserved(self):
        rng=np.random.default_rng(720);dt=1/512;n=4096;t=np.arange(n)*dt
        x=rng.normal(size=n)+3*np.sin(2*np.pi*199.375*t)+np.cos(2*np.pi*223.75*t)
        x*=np.hanning(n)
        old=np.fft.rfft(x)*dt;new=np.fft.rfft(resample(x,4*n))*(dt/4)
        band=np.fft.rfftfreq(n,dt)<=224
        np.testing.assert_allclose(new[:len(old)][band],old[band],rtol=1e-12,atol=1e-13)


if __name__=='__main__':unittest.main()
