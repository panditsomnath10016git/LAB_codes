# This script contains functions and classes to analyze 1D Photonic crystal
# charecteristics by Transmission Line Transfer Matrix method
# !! TODO After critical angle the results are not wrong.

from copy import deepcopy
from types import FunctionType

from numpy import (append, arcsin, array, complex128, cos, sum, cumsum, exp, eye,
                   loadtxt, ndarray, pi, sin, size, zeros, zeros_like, abs)
from scipy.interpolate import interp1d


# function to load refractive index data
def get_RI(filepath: str) -> FunctionType:
    """
    filepath: path to txt file with refractive index data as 'lambda n k'
            columns lambda in micrometers.
    returns: function which gives RI value at any wavelength(nm)
    """
    RI_data = loadtxt(filepath)
    wl, n, k = RI_data[:, 0] * 1000, RI_data[:, 1], RI_data[:, 2]
    return lambda x: interp1d(wl, n)(x) + 1j * interp1d(wl, k)(x)


class LayerTransmissionMatrix:
    def __init__(self, n, d, theta, lmda):
        """
        Wave amplitude transfer matrix in a medium.

        Parameters
        -----------
        n : float, int, array
            Refractive index(s) of the layer.
        d : float, int
            Propagation distance.
        theta : float, int, array
            Angle(s) of propagation.
        lmda : float, int, array
            Wavelength(s).

        Returns
        --------
        L_mat : ndarray
            Amplitude transmission matrix(s) through layer.
        """
        delta = 2j * pi * n * d * cos(theta) / lmda
        self.L_mat = array(
            [
                [exp(delta), zeros_like(delta)],
                [zeros_like(delta), exp(-delta)],
            ]
        )
        # make matrix in last 2 indices for further multiplications
        if isinstance(delta, ndarray):
            self.L_mat = self.L_mat.transpose(2, 0, 1)

    def __call__(self):
        return self.L_mat


class LayerChangeMatrix:
    def __init__(self, n1, n2, theta1, theta2):
        """
        Transfer matrix for the discontinuity between two layers.

        Parameters
        -----------
        n1, n2 : float, int, array
            refractive index(s) of layer 1 and 2.
        theta1, theta2 : float, int, array
            light propagation angle(s) at layer 1 and 2.

        Returns
        --------
        array([I_TE, I_TM]) : ndarray
            Layer change matrix for TE and TM mode.
        """

        p_TE = n1 * cos(theta1) / (n2 * cos(theta2))
        p_TM = n1 * cos(theta2) / (n2 * cos(theta1))
        self.ME = 0.5 * array([[1 + p_TE, 1 - p_TE], [1 - p_TE, 1 + p_TE]])
        self.MM = 0.5 * array([[1 + p_TM, 1 - p_TM], [1 - p_TM, 1 + p_TM]])

        if isinstance(p_TE, ndarray):
            self.ME = self.ME.transpose(2, 0, 1)
            self.MM = self.MM.transpose(2, 0, 1)

    def __call__(self):
        return array([self.ME, self.MM])


class TLTransferMatrix:
    def __init__(self, n_p, n, d, lmda, theta_p=0, theta=0):
        """
        calculates transmission line tranfer matrix of a layer

        Parameters
        -----------
        n_p : refractive index(indices) of the previous layer.
        n : refractive index(indices) of the layer.
        d : thickness.
        lmda : wavelength(s) of light.
        theta_p: incident angle(s) of light in previous layer.
        theta : incident angle(s) of light.

        Returns
        --------
        array([ME, MM]) : ndarray
            Layer transfer matrix for TE and TM mode
            (l -> l+1 layer end to end).
        """
        I_mat = LayerChangeMatrix(n_p, n, theta_p, theta)
        L_mat = LayerTransmissionMatrix(n, d, theta, lmda)()
        self.ME, self.MM = L_mat @ I_mat.ME, L_mat @ I_mat.MM

    def __call__(self):
        return array([self.ME, self.MM])


class CalcRTA:
    def __init__(self, ni, nf, Tr_mat_TE, Tr_mat_TM, theta_i=0, frac_TE=0.5):
        """
        Calculates the Reflection, Transmission, Absorbtion Spectra
        for the given transmission-line transfer matrix and assigns
        R, T, A method to the object to get the values.

        Parameters
        -----------
        ni, nf : array, float, int
            Refractive indices of first and last layer
        Tr_matrix_TE, Tr_mat_TM : ndarrray
            Transfer matrix(0 -> l) where the first axis can be wavelength
            for TE and TM mode.
        theta_i : float
            Incident angle in radians.
        frac_TE  : float
            Fraction of TE polarization in the incident light.

        Returns:
        ----------
        __call__ : array([R, T, A])

        @methods R, T, A, r_TE, r_TM, t_TE, t_TM, R_TE, R_TM, T_TE, T_TM
        """
        self.ME, self.MM = Tr_mat_TE, Tr_mat_TM
        self.frac_TE = frac_TE
        # for single wavelengths make the matrix 3d for valid index slicing
        try:
            self.ME[0, :, :]
        except IndexError:
            self.ME, self.MM = array([self.ME]), array([self.MM])

        r_TE = -self.ME[:, 1, 0] / self.ME[:, 1, 1]
        t_TE = self.ME[:, 0, 0] + self.ME[:, 0, 1] * r_TE
        r_TM = -self.MM[:, 1, 0] / self.MM[:, 1, 1]
        t_TM = self.MM[:, 0, 0] + self.MM[:, 0, 1] * r_TM
        self.r_TE, self.r_TM, self.t_TE, self.t_TM = r_TE, r_TM, t_TE, t_TM
        self.R_TE, self.T_TE = abs(r_TE) ** 2, abs(t_TE) ** 2
        self.R_TM, self.T_TM = abs(r_TM) ** 2, abs(t_TM) ** 2
        self.R = self.R_TE * self.frac_TE + self.R_TM * (1 - self.frac_TE)
        self.T = self.T_TE * self.frac_TE + abs(
            nf * cos(arcsin(ni * sin(theta_i) / nf, dtype=complex128), dtype=complex128) / (ni * cos(theta_i))
        ) * self.T_TM * (1 - self.frac_TE)
        self.A = 1 - self.R - self.T

    def __call__(self):
        return array([self.R, self.T, self.A])


class PhC_TLTransferMatrix:
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
    ):
        """
        Calculates transfer matrix(0 -> l) of a 1-D Photonic crystal structure

        Parameters
        -----------
        lmda : array
            Wavelengths for which the transfer matrix has to be calculated.
        ni : function, float, int
            RI function/value at any wavelength for incident media.
        RI_profile : array
            Functions/values which give RI at any wavelength for each layer.
        nf : function, float, int
            RI function/value at any wavelength for final media.
        thickness :  array
            Thickness of each layer
        theta_i : float, int
            Incident angle(in radians) of light on the structure.
        frac_TE  : float
            Fraction of TE polarization in the incident light.
        calc_spectra : bool
            Calculate reflection, transmission and absorbtion spectra.

        Returns
        --------
        @methods : ME, MM, spectrum, ME_mats, MM_mats

        """
        self.lmda = lmda
        self.N_i = ni
        self.N_f = nf
        self.thickness = thickness
        self.theta_i = theta_i
        if (frac_TE < 0) | (frac_TE > 1.0):
            raise ValueError(
                f"0 > frac_TE={frac_TE} > 1; fraction of TE polatrization in the light."
            )
        self.frac_TE = frac_TE
        # Ensure ni, nf to be a funcion of wavelength
        if type(ni) != FunctionType:
            self.N_i = lambda x: ni * x / x
        if type(nf) != FunctionType:
            self.N_f = lambda x: nf * x / x

        # TLTMM needs RI of previous layer so..
        self.RI_profile = append(self.N_i, RI_profile)

        self.n_layer, self.n_wl = size(RI_profile), size(lmda)
        self.NSnell = self.N_i(lmda) * sin(theta_i)

        # Initialize Transfer matrics for TE and TM mode
        ME = zeros((self.n_wl, 2, 2), dtype=complex128)
        for i in range(self.n_wl):
            ME[i, :, :] = eye(2)
        MM = deepcopy(ME)

        # Collect cumulative transfer matrix after each layer
        TE_mats, TM_mats = [], []

        # cumulatively calculate transfer martix layer by layer
        for k in range(self.n_layer):
            # check if refractive index is function or value
            # _p is for previous layers
            try:
                NN_lmda_p = self.RI_profile[k](lmda)
            except TypeError:
                NN_lmda_p = self.RI_profile[k]
            try:
                NN_lmda = self.RI_profile[k + 1](lmda)
            except TypeError:
                NN_lmda = self.RI_profile[k + 1]

            theta_p = arcsin(self.NSnell / NN_lmda_p, dtype=complex128)
            theta = arcsin(self.NSnell / NN_lmda, dtype=complex128)

            tr_matrix = TLTransferMatrix(
                NN_lmda_p, NN_lmda, thickness[k], lmda, theta_p, theta
            )
            ME, MM = tr_matrix.ME @ ME, tr_matrix.MM @ MM
            TE_mats += [ME]
            TM_mats += [MM]

        # Layer change matrix from PhC last layer to substrate
        self.I_mat_f = LayerChangeMatrix(
            NN_lmda,
            self.N_f(lmda),
            theta,
            arcsin(self.NSnell / self.N_f(lmda), dtype=complex128),
        )
        # Total transfer matrix of the PhC structure for TE and TM mode
        self.ME, self.MM = self.I_mat_f.ME @ ME, self.I_mat_f.MM @ MM

        # Transfer matrix at start of each layer is recorded. Can be used for
        # electric field calculation.
        self.TE_mats, self.TM_mats = array(TE_mats), array(TM_mats)

        if calc_spectra:
            self.spectrum = CalcRTA(
                self.N_i(lmda),
                self.N_f(lmda),
                self.ME,
                self.MM,
                self.theta_i,
                frac_TE,
            )

    def T_mats(self):
        """Cumulative transfer matrices after each layer.

        Returns
        --------
        ndarray : ndarray with 5 axes.
            indices are [layerno, wavelength, TE/TM, T_row, T_col]

        """
        return array([self.TE_mats, self.TM_mats]).transpose(1, 2, 0, 3, 4)


class PhcE_FieldDensity(PhC_TLTransferMatrix):
    def __init__(
        self,
        lmda,
        ni,
        RI_profile,
        nf,
        thickness,
        theta_i=0.0,
        frac_TE=0.5,
        n_points=2000,
    ):
        super().__init__(
            array([lmda]),
            ni,
            RI_profile,
            nf,
            thickness,
            theta_i,
            frac_TE,
        )
        """
        Calculates electric field densiy inside a photonic crystal.

        Parameters
        -----------
        lmda : int, float
            Wavelength for which the electric field  has to be calculated.
        ni : function, float, int
            RI function/value at any wavelength for incident media.
        RI_profile : array
            Functions/values which give RI at any wavelength for each layer.
        nf : function, float, int
            RI function/value at any wavelength for final media.
        thickness :  array
            Thickness of each layer
        theta_i : float, int
            Incident angle(in radians) of light on the structure.
        frac_TE  : float
            Fraction of TE polarization in the incident light.
        n_points : int
            number of points for length discretization

        Returns
        --------
        __call__: array[(Z, E^2)]

        @methods : E_TE, E_TM, E, E_density, Z, layer_z, n_points
        """
        L = sum(thickness)  # total length of the structure
        layer_points = (n_points * thickness / L).astype(int)  # pts in layer
        n_points = sum(layer_points)  # points rounding off to interger
        # layer points include the point at interface.
        Z = zeros(n_points + 1)  # distance in the structure
        Z[-1] = L

        # Calculate transfer matrix after each layer
        T_mats = self.T_mats()[:, 0]

        # Initialize E_field matrix [[E+], [E-]] for TE and TM mode
        # indexes [n_point, TE/TM, E+/E-, 1]
        E_pm = zeros(
            (n_points + 1, 2, 2, 1), dtype=complex128
        )  # for each point
        E_pm[0, :, 0] = 1.0  # E+
        # Initialize E- field with reflection coefficent
        E_pm[0, 0, 1] += self.spectrum.r_TE
        E_pm[0, 1, 1] += self.spectrum.r_TM

        i = n_points  # index for n_points from end
        # calculate transfer martix for each point
        for j in range(1, self.n_layer + 1):
            try:
                n = self.RI_profile[-j](lmda)
            except TypeError:
                n = self.RI_profile[-j]
            theta = arcsin(self.NSnell / n, dtype=complex128)
            d = thickness[-j]
            points_j = layer_points[-j]
            dz = d / points_j

            E_pm[i] = T_mats[-j] @ E_pm[0]
            i -= 1

            # Layer transmission matrix for dz distance in the layer
            # from the end
            L_mat = LayerTransmissionMatrix(n, -dz, theta, lmda)()

            for _ in range(points_j - 1):
                E_pm[i] = L_mat @ E_pm[i + 1]  # Electric field at  i point
                Z[i] = Z[i + 1] - dz  # Distance in the structure
                i -= 1
            Z[i] = Z[i + 1] - dz

        self.n_points = n_points
        self.Z = Z
        self.layer_z = cumsum(append(0, thickness))
        self.E_TE = E_pm[:, 0]
        self.E_TM = E_pm[:, 1]
        self.E = self.E_TE * frac_TE + self.E_TM * (1 - frac_TE)
        self.E_density = abs(self.E[:, 0] + self.E[:, 1]) ** 2

    def __call__(self):
        return array([self.Z, self.E_density])
