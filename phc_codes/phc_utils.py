from types import FunctionType
from scipy.interpolate import interp1d
import numpy as np
import matplotlib.pyplot as plt

import os

this_dir, this_filename = os.path.split(__file__)
RI_DIR = os.path.join(this_dir, "RI_data")
#RI_DIR = "G:\Other computers\My Computer\IITKgp documents\PhD\LAB\Codes\RI_data"

#print("Refractive Index DIR: ", RI_DIR)

# function to load refractive index data
def get_RI(filepath: str) -> FunctionType:
    """
    filepath: path to txt file with refractive index data as 'lambda n k'
            columns lambda in micrometers.
    returns: function which gives RI value at any wavelength(nm)
    """
    RI_data = np.loadtxt(filepath, encoding='utf-8')
    wl, n, k = RI_data[:, 0] * 1000, RI_data[:, 1], RI_data[:, 2]
    return lambda x: interp1d(wl, n)(x) + 1j * interp1d(wl, k)(x)

def generate_lambda(wl1, wl2, n_lambda = 1000):
    return np.linspace(wl1, wl2, n_lambda)

def refractive_index(material):
    return get_RI(this_dir+rf".\RI_data\RI_{material}.txt")


def find_phc_params(*args):
    pass

def plot_layers(structure:object, xlim, ylim=1.1, wavelength:float=532.0,  lim_eps_colors=[.9, 4], ):
        """plot layerstack from pymoosh structure

        evaluate materials at given wavelength for labels

        structure : PyMoosh structure object
        wavelength : for refractive index

        @modified code from pymoosh plot_stack
        """
            
        _mats = []
        _mats_names = []
        _index_diff = []
        for i, mat in enumerate(np.array(structure.materials)[structure.layer_type]):
            if hasattr(mat, 'name') and len(mat.name)>0:
                # Named materials
                _mats_names.append(mat.name)
                _index_diff.append(i)
            # Simply using the permittivity at a given wavelength (default is 500 nm)
            _mats.append(mat.get_permittivity(wavelength))
        _index_diff = np.array(_index_diff)

        _thick = structure.thickness

        ## give sub- and superstrate a finite thickness
        if _thick[0] == 0: _thick[0] = 50
        if _thick[-1] == 0: _thick[-1] = 50


        ## define colors for layers of different ref.indices
        if any(isinstance(x, str) for x in _mats):
            colors = [f'C{i}'for i in range(len(_thick))] 
        else:
            cmap = plt.cm.Greys
            colors = [cmap((n.real-lim_eps_colors[0])/lim_eps_colors[1])
                                       for n in _mats]

        for i, di in enumerate(_thick):
            d0 = np.sum(_thick[:i])
            #print(colors[i])
            n = _mats[i]

            if type(n) is not str:
                if abs(np.imag(n)) < 1e-7:
                    n = np.real(n)
                n = str(np.round(n, 2))

            if i < len(_thick)-1:
                plt.axvline(d0+di, color='k', lw=0.8)
            plt.axvspan(d0, d0+di, color=colors[i], alpha=0.35)

        # define the limits of the plot
        plt.ylim(0, ylim)
        plt.xlim(0,xlim)
        #plt.xlim(0,np.sum(structure.thickness))