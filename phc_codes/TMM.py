# This script contains functions and classes to analyze 1D Photonic crystal
# charecteristics by Transfer Matrix method\
# !! TODO After critical angle the results are not wrong.

from copy import deepcopy
import types

from numpy import (
    arcsin,
    array,
    complex128,
    cos,
    eye,
    loadtxt, 
    matmul,
    pi,
    sin,
    size,
    zeros,
    abs
)
from scipy.interpolate import interp1d
#from time import perf_counter as tm

# function to load refractive index data
def get_RI(file):
    """file: txt file with refractive index data as 'lambda n k' columns
            lambda in micrometers.
    returns: function which gives RI value at any wavelength(nm)"""
    RI_data = loadtxt(file)
    wl, n, k = RI_data[:, 0] * 1000, RI_data[:, 1], RI_data[:, 2]
    return lambda x: interp1d(wl, n)(x) + 1j * interp1d(wl, k)(x)


class PhCStructure:
    def __init__(self, RI_func, thickness):
        """
        Parameters
        ----------
        RI_func :

        """
        pass


class TransferMatrix:
    def __init__(self, n, d, lmda, theta=0):
        """
        calculates tranfer matrix of a layer

        Parameters
        -----------
        n : refractive index(indices)
        d : thickness
        lmda : wavelength(s) of light
        theta : incident angle(s) of light
        """
        p, q = n * cos(theta), cos(theta) / n
        delta = 2 * pi * d * p / lmda
        a, b = cos(delta, dtype=complex128), -1j * sin(delta, dtype=complex128)
        self.ME = array([[a, b / p], [b * p, a]]).transpose(2, 0, 1)
        self.MM = array([[a, b / q], [b * q, a]]).transpose(2, 0, 1)
        # return array([self.ME, self.MM])


class PhCTransferMatrix:
    def __init__(
        self,
        lmda,
        ni,
        RI_profile,
        nf,
        thickness,
        theta_i=0.0,
        frac_TE=0.5,
        calc_spectra=True,
        flip_phc=False,
    ):
        """
        Calculates transfer matrix of a 1-D Photonic crystal structure

        Parameters
        -----------
        lmda : array
            wavelengths for which the transfer matrix has to be calculated
        ni : function, float, int
            RI function/value at any wavelength for incident media
        RI_profile : array
            functions/values which give RI at any wavelength for each layer
        nf : function, float, int
            RI function/value at any wavelength for final media
        thickness :  array
            thickness of each layer
        theta_i : float, int
            incident angle(in radians) of light on the structure
        frac_TE  : float
            fraction of TE polarization in the incident light
        calc_spectra : bool
            calculate reflection, transmission and absorbtion spectra
        flip_phc : bool
            flip the crystal structure i.e. reverse the structure except inital and final layers
        """
        self.lmda = lmda
        self.N_i = ni
        self.N_f = nf
        if type(ni)!=types.FunctionType:
            self.N_i = lambda x: ni * x / x

        if flip_phc:
            self.RI_profile = RI_profile[::-1]
        else:
            self.RI_profile = RI_profile

        if type(nf)!=types.FunctionType:
            self.N_f = lambda x: nf * x / x

        self.thickness = thickness
        self.theta_i = theta_i
        self.frac_TE = frac_TE
        n_layer, n_wl = size(self.RI_profile), size(lmda)

        # Initialize Transfer matrics for TE and TM mode
        ME = zeros((n_wl, 2, 2), dtype=complex128)
        MM = zeros((n_wl, 2, 2), dtype=complex128)
        for i in range(n_wl):
            ME[i, :, :] = eye(2)
            MM[i, :, :] = eye(2)

        # cumulatively calculate transfer martix layer by layer
        timer = []
        for k in range(n_layer):
            try:
                NN_lmda = self.RI_profile[k](lmda)
            except TypeError:
                NN_lmda = self.RI_profile[k]

            theta_k = arcsin(self.N_i(lmda) * sin(theta_i) / NN_lmda, dtype=complex128)
            #start = tm()
            tr_matrix = TransferMatrix(NN_lmda, thickness[k], lmda, theta_k)
            #end = tm()
            #timer+=[end - start]
            ME, MM = matmul(ME, tr_matrix.ME), matmul(MM, tr_matrix.MM)
       # print('avg time in TransferMatrix',average(timer))
        self.ME = ME
        self.MM = MM
        if calc_spectra:
            self.calc_RTA()

    def calc_RTA(self):
        """
        Calculates the Reflection, Transmission, Absorbtion Spectra
        for the structure and assigns R, T, A method to the object
        to get the values.

        Returns:
        ----------
        array([R, T, A])
        """
        ni = self.N_i(self.lmda)
        nf = self.N_f(self.lmda)
        theta_f = arcsin(ni * sin(self.theta_i) / nf, dtype=complex128)
        p_i, p_f = ni * cos(self.theta_i), nf * cos(theta_f)
        A = (self.ME[:, 0, 0] + self.ME[:, 0, 1] * p_f) * p_i
        B = self.ME[:, 1, 0] + self.ME[:, 1, 1] * p_f
        q_i, q_f = cos(self.theta_i) / ni, cos(theta_f) / nf
        C = (self.MM[:, 0, 0] + self.MM[:, 0, 1] * q_f) * q_i
        D = self.MM[:, 1, 0] + self.MM[:, 1, 1] * q_f

        # Reflectance and Transmittance
        self.R_TE = abs((A - B) / (A + B)) ** 2
        self.R_TM = abs((C - D) / (C + D)) ** 2
        self.T_TE = abs(2 * p_i / (A + B)) ** 2
        self.T_TM = abs(2 * p_i / (C + D)) ** 2
        self.R = self.R_TE * self.frac_TE + self.R_TM * (1 - self.frac_TE)
        self.T = self.T_TE * self.frac_TE + self.T_TM * (1 - self.frac_TE)
        self.A = 1 - self.R - self.T
        return array([self.R, self.T, self.A])
