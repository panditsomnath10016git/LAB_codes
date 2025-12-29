import numpy as np
import matplotlib.pyplot as plt
import scipy.signal as sig
import pywt


class Spectrometer:
    def __init__(self, codename, start_wl, end_wl, resolution):
        self.codename = codename
        self.start_wl = start_wl
        self.end_wl = end_wl
        self.resolution = resolution
        if codename == "1809041U1":
            self.nickname = "orange spectrometer"


def load_spectra(file: str, usecols=[0,4], **kwargs):
    """Load .TXT files saved with Avsoft as ASCII
    file : file, str
        full path to the file/ relative path
    usecols : list, ndarray
        choose the columns to read from the file
    Returns : ndarray
        [wavelength, reflection spectrum]
    """
    return np.loadtxt(file, skiprows=8, delimiter=";", usecols=usecols, **kwargs)

def get_wavelengths(spectrum_data: np.ndarray):
    """get the set of wavelengths from the spectrum data array
    spectrum_data : n x 2 ndarray
        spectrum data with wavelength and value columns
    """
    return spectrum_data[:,0]


def load_continuous_spectra(file:str, **kwargs):
    """Load continuous data (multicolumn) save in ASCII from avasoft

    Returns:
    (ndarray(Spectra), n_samples)"""
    data = np.loadtxt(file, skiprows=10, delimiter=";", **kwargs)
    n_samples = data.shape[1]-3
    return np.delete(data, [1,2], 1), n_samples


def identify_spectrometer(file: str):
    """Get details of spectrometer from .TXT files saved with Avsoft as ASCII
    file : file, str
        full path to the file/ relative path
    Returns : spectrometer object
    """
    data = load_spectra(file)
    codename = file[-13:-4]
    start_wl, end_wl = data[0, 0], data[-1, 0]
    resolution = np.round((end_wl - start_wl) / np.size(data, 0), 3)
    return Spectrometer(codename, start_wl, end_wl, resolution)


def clip_wl(data: np.ndarray, wl_1: int | float, wl_2: int | float):
    """Clip range of wavelength data from spectrum
    data  : ndarray containing spectrum data
    wl_1 : start wavelenght
    wl_2 : end wavelenght

    Returns: clipped data -> ndarray
    """
    i_1 = find_index(data[:, 0], wl_1)
    i_2 = find_index(data[:, 0], wl_2)
    return data[i_1:i_2]


def find_index(array: np.ndarray, value: int | float):
    """Find index of the given value in an array"""
    return np.argmin(abs(array - value))


def plot_spectrum(ax, data, **kwargs):
    """Plots data in the given axis of the matplotlib figure

    Parameters
    ------------
    ax : handle
        axis of a matplotlib figure
    data : array
        [x, y] data

    Returns
    ---------
    ax : axis of matplotlib"""
    ax.plot(data[:, 0], data[:, 1], **kwargs)
    return ax


def smooth_spectrum(spectrum_data: np.ndarray, axis=0, cut_freq=85):
    """Smooth the spectrum data ndarray[wl, intensity]

    cut freq = % of high frequencies to cut"""
    freq_coeff = np.fft.fft(spectrum_data[:,1], axis=axis)
    max_freq = spectrum_data.shape[0]
    half_freq = int(max_freq/2)
    cut_freq = int(max_freq*cut_freq/200)
    #print(freq_coeff.size)
    freq_coeff[half_freq-cut_freq:half_freq+cut_freq]=0
    #freq_coeff[half_freq+cut_freq:]=0
    spectrum_new = np.fft.ifft(freq_coeff)
    return np.array([spectrum_data[:,0], spectrum_new.real]).T#fft.ifft(freq_coeff).real]).T


def spectrum_denoise(signal, wavelet="db8", level=3):
    # --- 1. Savitzky–Golay smoothing ---
    # Choose window length (~5–21 points depending on sampling) and polynomial order (2–4)
    signal = sig.savgol_filter(signal, window_length=15, polyorder=3)

    # --- 2. Wavelet denoising ---
    coeffs = pywt.wavedec(signal, wavelet, mode='smooth')
    sigma = np.median(np.abs(coeffs[-1])) / 0.6745
    uthresh = sigma * np.sqrt(2 * np.log(len(signal)))
    coeffs_thresh = [pywt.threshold(c, uthresh, mode='soft') for c in coeffs]
    return pywt.waverec(coeffs_thresh, wavelet, mode='smooth')

def lmda_theta_to_omega_k(lmda, theta, nsub):
    """convert lambda-theta data to omega/c-k_x data for phc

    lmda : float , in units of nm
    theta : float , in units of degree
    nsub : refractive index of substrate"""
    omega = 2 * np.pi / lmda *1e9
    k = nsub * omega * np.sin(theta * np.pi / 180)
    return omega, k

def freq_spectrum(y_data):
    """y_data : ndarray[y]

    Output : ndarray[coeffts]"""
    return np.fft.fft(y_data)


def filter_spectrum(signal, axis=0, btype='lowpass', fs=1000, cutoff_freq = 500, filter_order = 5)->np.ndarray:
    '''Butterworth low-pass filter

    signal : ndarray,    spectrum data without wl column
    axis : axis along which filter applied
    fs : Sampling frequency (Hz)
    cutoff_freq : Frequencies above this will be attenuated
    filter_order :  Filter order (higher means sharper cutoff)

    Returns : ndarray, filtered spectrum data 
    '''

    # Design the filter using signal.butter and output='sos' (second-order sections for stability)
    sos = sig.butter(filter_order, cutoff_freq, btype=btype, fs=fs, output='sos')

    # Use signal.sosfiltfilt for zero-phase filtering (prevents time shift)
    return sig.sosfiltfilt(sos, signal, axis=axis)

def polyfit_peak(data, poly_n=6,**kwargs):
    """fit the peak with polynomial
    
    data : 2 column ndarray [wavelength, reflectance]
    
    Returns : 2 column ndarray [wavelength, reflectance]
    """
    x, y = data[:,0], data[:,1]
    c = np.polyfit(x, y, poly_n)
    return np.array([x, np.poly1d(c)(x)], dtype=float).T

def fanofit_peak(data, ):
    pass



def get_fwhm(data, **kwargs):
    """find fwhm from the given spectrum data

    data : 2 column ndarray [wavelength, reflectance]
    
    Returns : fwhm, (peak indics, peak values)
    """
    x, y = data[:,0], data[:,1]
    dx = x[1] - x[0]
    peaks, _ = sig.find_peaks(y, **kwargs)
    results_half = sig.peak_widths(y, peaks, rel_height=0.5)
    #print(type(results_half[0]))
    fwhm = results_half[0] * dx
    fwhm = empty_arr_to_zero(fwhm)

    return fwhm, (peaks, y[peaks])

def get_Q(data, **kwargs):
    """find Q factor of a resonance peak from the given spectrum data

    data : 2 column ndarray [wavelength, reflectance]
    
    Returns : Q factor, (peak indics, peak values, fwhms)
    """
    x, y = data[:,0], data[:,1]
    dx = x[1] - x[0]
    peaks, _ = sig.find_peaks(y, **kwargs)
    results_half = sig.peak_widths(y, peaks, rel_height=0.5)
    fwhm = results_half[0] * dx
    q_vals = x[peaks] / (fwhm)
    q_vals = empty_arr_to_zero(q_vals)

    return q_vals, (peaks, y[peaks], fwhm)


def empty_arr_to_zero(arr, dim=(1)):
    """fill any empty array with zeros of given dimension (default (1)). Mainly used on the scipy.find_peaks() returns"""
    try: 
        arr[0]
    except IndexError:
        arr = np.zeros(dim)
    return arr