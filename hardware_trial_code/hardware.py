import numpy as np
import matplotlib.pyplot as plt


class FrontEndSim(object):
    def __init__(self, f1=3, f2=15, gain=20, iip3=-10, tsamp=1e-9, bits=8):
        self.f1 = f1
        self.f2 = f2
        self.gain = gain
        self.iip3 = iip3
        self.tsamp = tsamp
        self.bits = bits

        """
        Calibrate the saturation level for the tanh nonlinearity
        
        The nonlinearity is given by:
            y = sqrt(gain * Esat) * tanh(vin / sqrt(Esat))
            
        where Esat is the maximum output energy per sample before the gain.
        
        The IIP3*Tsamp = 4/3 * alpha1 / alpha3 = 4 * Esat
        
        This sets Esat to dbmJ
        

        """
        self.Esat = self.iip3 + 10 * np.log10(self.tsamp / 4)

        # Convert to linear
        self.gain_lin = 10 ** (self.gain / 10)
        self.Esat_lin = 10 ** (self.Esat / 10)

        self.f1_lin = 10 ** (self.f1 / 10)
        self.f2_lin = 10 ** (self.f2 / 10)
        self.EkT_lin = 10 ** (-174 / 10)

    def simulate(self, x, nonoise=False, nonlinearity=True):
        """
        Simulate the front end

        Parameters
        ----------
        x : array-like, complex
            Input signal

        Returns
        -------
        y : array-like
            Output signal
        """

        s = x.shape

        w1std = np.sqrt(self.EkT_lin * self.f1_lin / 2)
        w1 = (
            np.random.normal(0, 1, size=s) + 1j * np.random.normal(0, 1, size=s)
        ) * w1std
        w2std = np.sqrt(self.EkT_lin * (self.f2_lin - 1) / 2)
        w2 = (
            np.random.normal(0, 1, size=s) + 1j * np.random.normal(0, 1, size=s)
        ) * w2std
        if nonoise:
            w1 = 0
            w2 = 0
        z1 = x + w1
        if nonlinearity:
            z2 = np.sqrt(self.gain_lin * self.Esat_lin) * np.tanh(
                z1 / np.sqrt(self.Esat_lin)
            )
        else:
            z2 = np.sqrt(self.gain_lin) * z1
        y = z2 + w2

        # Add the ADC quantization here
        #   y = quant(y)
        return y

    def set_adc_step(self, input_power):
        """
        Given an input power in dBm, sets the ADC step size to minimize the
        distortion.
        """
        pass


def main():

    tsamp = 1e-9
    fe = FrontEndSim(tsamp=tsamp)
    input_pow = np.linspace(-100, 0, 100)
    Ein = input_pow + 10 * np.log10(tsamp)

    x = np.sqrt(10 ** (Ein / 10))

    ylin = fe.simulate(x, nonoise=True, nonlinearity=False)
    ynonlin = fe.simulate(x, nonoise=True, nonlinearity=True)
    Eylin = 10 * np.log10(np.abs(ylin) ** 2)
    Eynonlin = 10 * np.log10(np.abs(ynonlin) ** 2)
    # plt.plot(input_pow, Eylin, label="Linear")
    # plt.plot(input_pow, Eynonlin, label="Nonlinear")
    plt.plot(x, ylin, label="Linear")
    plt.plot(x, ynonlin, label="Nonlinear")
    plt.legend()
    plt.grid()
    plt.xlabel("Input Power (dBm)")
    plt.show()


if __name__ == "__main__":
    main()
