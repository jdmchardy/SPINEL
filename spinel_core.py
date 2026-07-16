"""SPINEL core functions extracted from spinel.py (Stage 1).

Self-contained science/plot helpers. Kept behaviour-identical; only relocated.
"""
import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib.colors import Normalize
import pandas as pd
import io
import pyFAI
from scipy.interpolate import interp1d
from scipy.interpolate import RegularGridInterpolator
from scipy.signal import fftconvolve
from lmfit import Parameters, minimize, fit_report
from pyFAI import AzimuthalIntegrator
import tempfile
import zipfile

#For interactive plotting
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from PIL import Image
from pathlib import Path

#Import personal modules
import PO #Preferred Orientation Model
import cake_processing as cp #Dioptas cake (.txt) import/processing

#3d plotting
from mpl_toolkits.mplot3d import Axes3D

from scipy.interpolate import RegularGridInterpolator


#### Functions -----------------------------------------------------

def Gaussian(x, x0, sigma):
    return np.exp(-0.5 * ((x - x0) / sigma) ** 2)

#NOTE HAS BEEN MODIFIED TO MATCH NOTATION IN NYE
def stress_tensor_to_voigt(sigma_tensor):
    # Input shape (..., 3, 3)
    sig11 = sigma_tensor[..., 0, 0]
    sig22 = sigma_tensor[..., 1, 1]
    sig33 = sigma_tensor[..., 2, 2]
    sig23 = sigma_tensor[..., 1, 2]
    sig13 = sigma_tensor[..., 0, 2]
    sig12 = sigma_tensor[..., 0, 1]
    return np.stack([sig11, sig22, sig33, sig23, sig13, sig12], axis=-1) #Output shape is (..., 6) #Nye convention

def voigt_to_strain_tensor(e_voigt):
    #Also modified to use Nye convention
    e11 = e_voigt[..., 0]
    e22 = e_voigt[..., 1]
    e33 = e_voigt[..., 2]
    e23 = 0.5*e_voigt[..., 3]
    e13 = 0.5*e_voigt[..., 4]
    e12 = 0.5*e_voigt[..., 5]
    e_tensor = np.zeros(e_voigt.shape[:-1] + (3, 3))
    e_tensor[..., 0, 0] = e11
    e_tensor[..., 1, 1] = e22
    e_tensor[..., 2, 2] = e33
    e_tensor[..., 0, 2] = e_tensor[..., 2, 0] = e13
    e_tensor[..., 1, 2] = e_tensor[..., 2, 1] = e23
    e_tensor[..., 0, 1] = e_tensor[..., 1, 0] = e12
    return e_tensor

def get_d0(symmetry,h,k,l,a,b,c):
    """Evaluates the lattice plane spacing"""
    if symmetry == "cubic":
        d0 = a / np.linalg.norm([h, k, l])
    elif symmetry == "hexagonal":
        d0 = np.sqrt((3*a**2*c**2)/(4*c**2*(h**2+h*k+k**2)+3*a**2*l**2))
    elif symmetry in ["tetragonal_A","tetragonal_B"]:
        d0 = np.sqrt((a**2*c**2)/((h**2+k**2)*c**2+a**2*l**2))
    elif symmetry == "orthorhombic":
        d0 = np.sqrt(1/(h**2/a**2+k**2/b**2+l**2/c**2))
    elif symmetry == "trigonal_A":
        d0 = np.sqrt((3*a**2*c**2)/(4*c**2*(h**2+h*k+k**2)+3*a**2*l**2))
    else:
        st.write("Support not yet provided for {} symmetry".format(symmetry))
        d0 = 0
    return d0

def get_elastic(symmetry, hkl, lattice_params, cij_params):
    """Returns normalised H,K,L values and the symmetry specific elastic compliance matrix.
    
    Parameters:
    -----------
    symmetry : str
        The crystal symmetry
        cubic
        hexagonal
        tetragonal_A
        tetragonal_B
        orthorhombic
        trigonal_A
    hkl : tuple
        Miller indices (h, k, l)
    lattice_params : dict
        Lattice parameter dictionary
        "a_val" : float (Ang)
        "b_val" : float (Ang)
        "c_val" : float (Ang)
        "alpha" : float (deg)
        "beta" : float (deg)
        "gamma" : float (deg)
    cij_params : dict
        Elastic constants
        Can be extended to arbitrary length as required
        c11 : float (GPa)
        c12 : float (GPa)
        c44 : float (GPa) 

    Returns:
    --------
    H, K, L : floats
        The normalised Miller indices needed for the B matrix
    elastic : 2d.array
        The elastic compliance matrix
    """

    #Unpack the lattice parameters
    a = lattice_params.get("a_val")
    b = lattice_params.get("b_val")
    c = lattice_params.get("c_val")
    alpha = lattice_params.get("alpha")
    beta = lattice_params.get("beta")
    gamma = lattice_params.get("gamma")

    h, k, l = hkl
    if h == 0: h = 0.0000000001
    if k == 0: k = 0.0000000001
    if l == 0: l = 0.0000000001

    if symmetry == "cubic":
        # Normalize
        H = h / a
        K = k / a
        L = l / a
        #Unpack the elastic constants
        c11 = cij_params.get("c11")
        c12 = cij_params.get("c12")
        c44 = cij_params.get("c44")
        # Elastic constants matrix
        elastic = np.array([
            [c11, c12, c12, 0, 0, 0],
            [c12, c11, c12, 0, 0, 0],
            [c12, c12, c11, 0, 0, 0],
            [0, 0, 0, c44, 0, 0],
            [0, 0, 0, 0, c44, 0],
            [0, 0, 0, 0, 0, c44]
        ])
    elif symmetry == "hexagonal":
        # Normalize
        H = h / a
        K = (h+2*k) / (np.sqrt(3)*a)
        L = l / c
        #Unpack the elastic constants
        c11 = cij_params.get("c11")
        c12 = cij_params.get("c12")
        c13 = cij_params.get("c13")
        c33 = cij_params.get("c33")
        c44 = cij_params.get("c44")
        elastic = np.array([
            [c11, c12, c13, 0, 0, 0],
            [c12, c11, c13, 0, 0, 0],
            [c13, c13, c33, 0, 0, 0],
            [0, 0, 0, c44, 0, 0],
            [0, 0, 0, 0, c44, 0],
            [0, 0, 0, 0, 0, 0.5*(c11-c12)]
        ])
    elif symmetry == "tetragonal_A":
        # Normalize
        H = h / a
        K = k / a
        L = l / c
        #Unpack the elastic constants
        c11 = cij_params.get("c11")
        c12 = cij_params.get("c12")
        c13 = cij_params.get("c13")
        c33 = cij_params.get("c33")
        c44 = cij_params.get("c44")
        c66 = cij_params.get("c66")
        elastic = np.array([
            [c11, c12, c13, 0, 0, 0],
            [c12, c11, c13, 0, 0, 0],
            [c13, c13, c33, 0, 0, 0],
            [0, 0, 0, c44, 0, 0],
            [0, 0, 0, 0, c44, 0],
            [0, 0, 0, 0, 0, c66]
        ])
    elif symmetry == "tetragonal_B":
        # Normalize
        H = h / a
        K = k / a
        L = l / c
        #Unpack the elastic constants
        c11 = cij_params.get("c11")
        c12 = cij_params.get("c12")
        c13 = cij_params.get("c13")
        c33 = cij_params.get("c33")
        c44 = cij_params.get("c44")
        c66 = cij_params.get("c66")
        c16 = cij_params.get("c16")
        elastic = np.array([
            [c11, c12, c13, 0, 0, c16],
            [c12, c11, c13, 0, 0, -c16],
            [c13, c13, c33, 0, 0, 0],
            [0, 0, 0, c44, 0, 0],
            [0, 0, 0, 0, c44, 0],
            [c16, -c16, 0, 0, 0, c66]
        ])
    elif symmetry == "orthorhombic":
        # Normalize
        H = h / a
        K = k / b
        L = l / c
        #Unpack the elastic constants
        c11 = cij_params.get("c11")
        c22 = cij_params.get("c22")
        c33 = cij_params.get("c33")
        c12 = cij_params.get("c12")
        c13 = cij_params.get("c13")
        c23 = cij_params.get("c23")
        c44 = cij_params.get("c44")
        c55 = cij_params.get("c55")
        c66 = cij_params.get("c66")
        elastic = np.array([
            [c11, c12, c13, 0, 0, 0],
            [c12, c22, c23, 0, 0, 0],
            [c13, c23, c33, 0, 0, 0],
            [0, 0, 0, c44, 0, 0],
            [0, 0, 0, 0, c55, 0],
            [0, 0, 0, 0, 0, c66]
        ])
    elif symmetry == "trigonal_A":
        # Normalize
        H = h / a
        K = (h+2*k) / (np.sqrt(3)*a)
        L = l / c
        #Unpack the elastic constants
        c11 = cij_params.get("c11")
        c12 = cij_params.get("c12")
        c13 = cij_params.get("c13")
        c14 = cij_params.get("c14")
        c33 = cij_params.get("c33")
        c44 = cij_params.get("c44")
        elastic = np.array([
            [c11, c12, c13, c14, 0, 0],
            [c12, c11, c13, -c14, 0, 0],
            [c13, c13, c33, 0, 0, 0],
            [c14, -c14, 0, c44, 0, 0],
            [0, 0, 0, 0, c44, c14],
            [0, 0, 0, 0, c14, 0.5*(c11-c12)]
        ])
    else:
        st.write("Error! {} symmetry not supported".format(symmetry))
        H = 0
        K = 0
        L = 0
        elastic = 0

    return H, K, L, elastic

def cake_dict_to_2Dcake(cake_dict, step_2th=0.2, step_delta=5, broadening=True):
    """
    Rasterize cake_dict onto a regular 2D grid using bilinear weighting.
    
    Parameters
    ----------
    cake_dict : dict
        HKL label -> DataFrame with '2th', 'delta (degrees)', and intensity column
    step_2th : float
        grid spacing in 2θ direction
    step_delta : float
        grid spacing in δ direction

    Returns
    -------
    grid_2th : 1D array
        Grid values for 2θ (length n_2th)
    grid_delta : 1D array
        Grid values for δ (length n_delta)
    intensity_grid : 2D array
        Rasterized intensity map (shape = n_2th x n_delta)
    """
    
    # --- Collect all data from all HKLs ---
    all_2th = []
    all_delta = []
    all_intensity = []

    #Check whether broadening is on or off
    if broadening == True:
        for df in cake_dict.values():
            ideal_I = df["intensity"].iloc[0]
            n_points = len(df)
            if ideal_I == 0 or n_points == 0:
                continue
            # Each row contributes equally to the total intensity
            norm_intensity = df["intensity"] * df["PO_intensity"] / n_points
            all_2th.extend(df["2th"])
            all_delta.extend(df["delta (degrees)"])
            all_intensity.extend(norm_intensity)
    else:
        #Axial or transverse geometry with broadening off
        for df in cake_dict.values():
            unique = df.drop_duplicates(subset="delta (degrees)") #Pick out the unique delta values
            ideal_I = df["intensity"].iloc[0] 
            n_points = unique.shape[0]
            if ideal_I == 0 or n_points == 0:
                continue
            deltas = unique["delta (degrees)"].values
            # Average PO_intensity across phi for each delta
            mean_PO_intensity = (
                df.groupby("delta (degrees)")["PO_intensity"]
                  .mean()
                  .reindex(deltas)  # ensure same order as deltas
                  .values
            )
            norm_intensity = ideal_I * mean_PO_intensity / n_points
            #Get the mean values for each delta
            all_delta.extend(unique["delta (degrees)"].values)
            all_2th.extend(unique["Mean two_th @ delta"].values)
            all_intensity.extend(norm_intensity)
            
    all_2th = np.array(all_2th)
    all_delta = np.array(all_delta)
    all_intensity = np.array(all_intensity)

    # --- Create regular grid ---
    grid_2th = np.arange(all_2th.min()-0.5, all_2th.max()+0.5, step_2th)
    grid_delta = np.arange(all_delta.min(), all_delta.max()+step_delta, step_delta)
    n_2th = len(grid_2th)
    n_delta = len(grid_delta)

    intensity_grid = np.zeros((n_2th, n_delta), dtype=float)

    # --- Map each point to 4 nearest pixels (bilinear) ---
    for x, y, I in zip(all_2th, all_delta, all_intensity):
        # Floating grid indices
        i_f = (x - grid_2th[0]) / step_2th
        j_f = (y - grid_delta[0]) / step_delta

        i0 = int(np.floor(i_f))
        j0 = int(np.floor(j_f))
        i1 = i0 + 1
        j1 = j0 + 1

        # Fractions
        fi = i_f - i0
        fj = j_f - j0

        # Weights
        w00 = (1 - fi) * (1 - fj)
        w10 = fi * (1 - fj)
        w01 = (1 - fi) * fj
        w11 = fi * fj

        # Add contributions if indices are in bounds
        if 0 <= i0 < n_2th and 0 <= j0 < n_delta:
            intensity_grid[i0, j0] += I * w00
        if 0 <= i1 < n_2th and 0 <= j0 < n_delta:
            intensity_grid[i1, j0] += I * w10
        if 0 <= i0 < n_2th and 0 <= j1 < n_delta:
            intensity_grid[i0, j1] += I * w01
        if 0 <= i1 < n_2th and 0 <= j1 < n_delta:
            intensity_grid[i1, j1] += I * w11

    return grid_2th, grid_delta, intensity_grid

def compute_bin_indices(x_exp_common, hkl_peak_centers, window_width=0.2):
    """
    Compute index ranges (bins) around each peak center in x_exp_common.
    
    Parameters:
        x_exp_common (np.ndarray): Experimental 2θ values, common domain.
        peak_centers (List[float]): Estimated peak centers (from HKLs).
        window_width (float): Total width of the window (e.g., 0.2 for ±0.2).
        
    Returns:
        List of slice objects (or index arrays) to use for residual slicing.
    """
    
    hkl_peak_centers = np.sort(hkl_peak_centers)
    
    bin_indices = []
    for center in hkl_peak_centers:
        low = center - 2*window_width 
        high = center + 2*window_width 
        mask = (x_exp_common >= low) * (x_exp_common <= high)
        indices = np.where(mask)[0]
        if len(indices) > 0:
            bin_indices.append(indices)

    return bin_indices

def generate_1D_XRD_plot(XRD_df):
    twotheta_grid = XRD_df["2th"]
    total_pattern = XRD_df["Total Intensity"]

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=twotheta_grid,
        y=total_pattern,
        mode="lines",
        line=dict(width=1, color="black"),
        name="Simulated XRD"
    ))

    xmin = np.min(twotheta_grid)
    xmax = np.max(twotheta_grid)
    # Scale axes
    fig.update_layout(height=500)

    fig.update_xaxes(title="2th (degrees)", title_font=dict(size=18), tickfont=dict(size=14))
    fig.update_yaxes(title="Intensity (arb. u.)", title_font=dict(size=18), tickfont=dict(size=14))
    st.plotly_chart(fig, width='stretch')

def generate_1D_XRD_overlay(XRD_df, x_exp, y_exp):
    
    twoth_sim = XRD_df["2th"]
    intensity_sim = XRD_df["Total Intensity"]
    
    #Determine common data and interpolate
    x_min_sim = np.min(twoth_sim)
    x_max_sim = np.max(twoth_sim)
    mask = (x_exp >= x_min_sim) & (x_exp <= x_max_sim)
    x_exp_common = x_exp[mask]
    y_exp_common = y_exp[mask]
    interp_sim = interp1d(twoth_sim, intensity_sim, bounds_error=False, fill_value=np.nan)
    y_sim_common = interp_sim(x_exp_common)
    #Compute residuals
    residuals = y_exp_common - y_sim_common
    #Generate plotly figure

    fig = make_subplots(
        rows=2,
        cols=1,
        shared_xaxes=True,
        row_heights=[3, 1],
        vertical_spacing=0.05,
    )
    #Plot the simulated data
    fig.add_trace(go.Scatter(x=x_exp_common,
                             y=y_sim_common,
                             mode="lines",
                             line=dict(width=1, color="red"),
                             name="Simulated XRD"),
                  row=1, col=1
        )
    #Plot the experimental data
    fig.add_trace(go.Scatter(x=x_exp_common,
                             y=y_exp_common,
                             mode="lines",
                             line=dict(width=1, color="black"),
                             name="Experimental"),
                  row=1, col=1
        )

    #Plot the residual data
    fig.add_trace(go.Scatter(x=x_exp_common,
                             y=residuals,
                             mode="lines",
                             line=dict(width=1, color="blue"),
                             name="Residual"),
                  row=2, col=1
        )

    # Top subplot (XRD patterns)
    fig.update_yaxes(title_text="Intensity (arb.u.)", title_font=dict(size=18), tickfont=dict(size=14), row=1, col=1)
    # Bottom subplot (Residuals)
    fig.update_yaxes(title_text="Residuals", title_font=dict(size=18), tickfont=dict(size=14), row=2, col=1)
    # Shared X-axis label (only needs to be set once)
    fig.update_xaxes(title_text="2θ (degrees)", title_font=dict(size=18), tickfont=dict(size=14), row=2, col=1)

    fig.update_layout(height=700, legend=dict(font=dict(size=14)))

    st.plotly_chart(fig, width='stretch')

#Helper function for storing downloadable data
def store_download(key, datasource, buffer, filename, mime):
        st.session_state.download_data[key] = {
            "datasource": datasource,
            "buffer": buffer,
            "filename": filename,
            "mime": mime,
        }
