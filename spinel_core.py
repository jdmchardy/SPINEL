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

def compute_strain(hkl, intensity, symmetry, lattice_params, wavelength, cij_params, sigma_params, chi, phi_values, psi_values, alpha_values=None, po_model=None):
    """
    Evaluates strain_33 component for given hkl reflection.
    
    Parameters
    ----------
    hkl : tuple
        Miller indices (h, k, l)
    intensity : float
        ideal peak intensity assuming no preferred orientation
    symmetry : str
        Crystal symmetry
    lattice_params : dict
        Lattice parameter dictionary
        "a_val" : float (Ang)
        "b_val" : float (Ang)
        "c_val" : float (Ang)
        "alpha" : float (deg)
        "beta" : float (deg)
        "gamma" : float (deg)
    wavelength : float
        X-ray wavelength
    cij_params : dict
        Elastic constants
        Can be extended to arbitrary length as required
        c11 : float (GPa)
        c12 : float (GPa)
        c44 : float (GPa) 
    sigma_params : dict
        Stress matirx components
        sigma_11 : float (GPa)
        sigma_22 : float (GPa)
        sigma_33 : float (GPa)
        sigma_12 : float (GPa)
        sigma_13 : float (GPa)
        sigma_23 : float (GPa)
    chi : float
        The angle (degrees) between incident x-rays and the principle stress axis
    phi_values : np.array
        Array of phi values in radians
    psi_values : np.array or scalar
        Array of psi values in radians (or 0 to auto-calculate)

    Returns
    -------
    hkl_label : str
        String label of hkl
    df : pd.DataFrame
        DataFrame with columns:
            - strain_33
            - psi (deg)
            - phi (deg)
            - delta (deg) (the detector azimuth angle)
            - chi (deg) (the X-ray to laboratory strain axis (X3 in Funamori) angle)
            - d strain
            - 2theta (deg)
            - intensity
    psi_list : list
    strain_33_list : list
    """

    #Unpack the lattice parameters
    a = lattice_params.get("a_val")
    b = lattice_params.get("b_val")
    c = lattice_params.get("c_val")
    alpha = lattice_params.get("alpha")
    beta = lattice_params.get("beta")
    gamma = lattice_params.get("gamma")

    h, k, l = hkl

    H, K, L , elastic = get_elastic(symmetry, hkl, lattice_params, cij_params)
    elastic_compliance = np.linalg.inv(elastic)

    # N and M from normalised hkls
    N = np.sqrt(K**2 + L**2)
    M = np.sqrt(H**2 + K**2 + L**2)

    #Unpack the stress components
    sigma_11 = sigma_params['sigma_11']
    sigma_22 = sigma_params['sigma_22']
    sigma_33 = sigma_params['sigma_33']
    sigma_12 = sigma_params['sigma_12']
    sigma_13 = sigma_params['sigma_13']
    sigma_23 = sigma_params['sigma_23']

    #The stress matrix is symmetrical about the diagonal
    sigma = np.array([
        [sigma_11, sigma_12, sigma_13],
        [sigma_12, sigma_22, sigma_23],
        [sigma_13, sigma_23, sigma_33]
    ])

    # theta0 is needed both for psi-from-delta and for the lab-azimuth correction below
    d0 = get_d0(symmetry,h,k,l,a,b,c)
    sin_theta0 = wavelength / (2 * d0)
    theta0 = np.arcsin(sin_theta0)

    #Check if psi_values are given or if it must be calculated for XRD generation
    if isinstance(psi_values, int):
        if psi_values==0: #Standard setting for fine-resolution XRD generation
            deltas = np.arange(-180,180,2)
            #Set alphas to zero to trigger computation later
            alpha_values = None
            #Check if chi value is zero (axial case) or non-zero (radial)
            if chi == 0: 
                # return only one psi_value assuming compression axis aligned with X-rays
                psi_values = np.asarray([np.pi/2 - theta0])
            else:
                #Assume chi is non-zero (radial) and compute a psi for each azimuth bin (delta)
                deltas_rad = np.radians(deltas)
                chi_rad = np.radians(chi)
                psi_values = np.arccos(np.sin(chi_rad)*np.cos(deltas_rad)*np.cos(theta0)+np.cos(chi_rad)*np.sin(theta0))
            #phi_values are always passed to the function
            phi_values = np.asarray(phi_values)
        else: #A coarser resolution option for XRD refinement (less expensive due to fewer refinement iterations required)
            deltas = np.arange(-180,180,12)
            #Set alphas to None
            alpha_values = None
            #Check if chi value is zero (axial case) or non-zero (radial)
            if chi == 0: 
                # return only one psi_value assuming compression axis aligned with X-rays
                psi_values = np.asarray([np.pi/2 - theta0])
            else:
                #Assume chi is non-zero (radial) and compute a psi for each azimuth bin (delta)
                deltas_rad = np.radians(deltas)
                chi_rad = np.radians(chi)
                psi_values = np.arccos(np.sin(chi_rad)*np.cos(deltas_rad)*np.cos(theta0)+np.cos(chi_rad)*np.sin(theta0))
        #phi_values are always passed to the function
        phi_values = np.asarray(phi_values)
    
    else:
        # Funamori-style: caller supplies psi (output/plot axis) and phi, alpha
        # (integration axes) directly, in radians. Use them as-is -- no re-derivation.
        # phi and alpha are periodic and built endpoint=False by the caller so the
        # 0deg==360deg / -180deg==180deg wrap point is not double-counted in the average.
        psi_values = np.asarray(psi_values)
        phi_values = np.asarray(phi_values)
        if alpha_values is None:                       # fallback if a caller omits alpha sampling
            alpha_values = np.radians(np.linspace(-180, 180, 18, endpoint=False))
        else:
            alpha_values = np.asarray(alpha_values)
        deltas = np.array([0])
            
    #modified GRID construction to preserve psi-delta relationship
    n_phi = len(phi_values)
    n_psi = len(psi_values)
    n_delta = len(deltas)

    # --- Case 1: Axial (chi == 0 → single psi, many deltas) ---
    if n_psi == 1 and n_delta > 1:
        phi_grid, delta_grid = np.meshgrid(phi_values, deltas, indexing='ij')  # (n_phi, n_delta)
        psi_grid = np.full((n_phi, n_delta), psi_values[0])  # constant psi
    
    # --- Case 2: Radial (psi derived from delta) ---
    elif n_psi == n_delta and n_delta > 1:
        phi_grid, delta_grid = np.meshgrid(phi_values, deltas, indexing='ij')  # (n_phi, n_delta)
        psi_grid = np.tile(psi_values, (n_phi, 1))
    
    # --- Case 3: Independent psi (Funamori-style input) ---
    else:
        phi_grid, psi_grid = np.meshgrid(phi_values, psi_values, indexing='ij')  # (n_phi, n_psi)
        delta_grid = np.zeros_like(psi_grid)

    #Angle grids then constructed from these values
    cos_phi = np.cos(phi_grid)
    sin_phi = np.sin(phi_grid)
    cos_psi = np.cos(psi_grid)
    sin_psi = np.sin(psi_grid)

    #This is the Singh rotation matrix setup - rotate around x2 by psi and then x3' by phi
    # Rotation matrix A (shape: [n_phi, n_psi, 3, 3])
    #A = np.empty((cos_phi.shape[0], cos_phi.shape[1], 3, 3))
    #A[..., 0, 0] = cos_phi * cos_psi
    #A[..., 0, 1] = -sin_phi
    #A[..., 1, 0] = sin_phi * cos_psi
    #A[..., 0, 2] = cos_phi * sin_psi
    #A[..., 1, 1] = cos_phi
    #A[..., 1, 2] = sin_phi * sin_psi
    #A[..., 2, 0] = -sin_psi
    #A[..., 2, 1] = 0
    #A[..., 2, 2] = cos_psi

    #This is the Uchida rotation definition - rotate around x1 by psi and then x3' by phi
    A = np.empty((cos_phi.shape[0], cos_phi.shape[1], 3, 3))
    A[..., 0, 0] = cos_phi
    A[..., 0, 1] = -sin_phi*cos_psi
    A[..., 0, 2] = sin_phi * sin_psi
    A[..., 1, 0] = sin_phi
    A[..., 1, 1] = cos_phi * cos_psi
    A[..., 1, 2] = -cos_phi * sin_psi
    A[..., 2, 0] = 0
    A[..., 2, 1] = sin_psi
    A[..., 2, 2] = cos_psi

    # --- Lab-azimuth correction (Merkel 2006, alpha rotation about Z_S) -----
    # Uchida's a_ij (Eq. 11) places x'_3 in the x_2-x_3 plane regardless of delta. 
    # This is correct only for axially symmetric stress about Z_S.
    # For general stress, x'_3 must track the diffracting-plane normal Q in the sample frame K_S as delta varies.
    # We derive the alpha values from the delta, theta and chi values using a function above
    # based on the constraint that the dot-product of k and the x axis of the stress coordinates must be zero
    #
    # A_full = A_Uchida @ R_z(-alpha): mixes columns 0 and 1, leaves col 2.
    # For axial sigma (sigma_11 = sigma_22, off-diagonals zero) this collapses
    # back to the original Uchida result; for non-axial sigma it reproduces
    # the lab-azimuth dependence (Merkel 2006 Fig. 3 c-f).

    # --- Build list of alpha_grids to iterate over -----------------------------
    # Case A: alpha_values is None  -> single delta-derived alpha_grid (Merkel correction)
    # Case B: alpha_values is an array -> loop (Funamori-style non-axial sigma)
    if alpha_values is None:
        delta_grid_rad = np.radians(delta_grid)
        chi_rad = np.radians(chi)
        alpha_grid_list = [PO.compute_alpha(theta0, chi_rad, delta_grid_rad)]
    else:
        alpha_grid_list = [
            np.full_like(phi_grid, a) for a in alpha_values
        ]

    PO_on = st.session_state.params.get("PO_toggle")
    direct_PO = PO_on and (alpha_values is not None)   # Funamori: per-orientation eval
    if PO_on:
        po_components = [
            {"tau": st.session_state.params.get("tau"),
             "omega": st.session_state.params.get("omega"),
             "R": st.session_state.params.get("R"),
             "weight": st.session_state.params.get("weight")}
        ]
        PO_MODEL = PO.PO_Model(
            po_model=po_model, 
            components=po_components,
            baseline=st.session_state.params.get("baseline"),
            symmetry=symmetry, 
            wavelength=wavelength,
            lattice_params=lattice_params, 
            chi_deg=chi,
            POD_xtal=st.session_state.params.get("hkl_POD"),
        )

    # --- Accumulators for per-alpha flattened outputs --------------------------
    strain_33_chunks = []
    phi_chunks = []
    psi_chunks = []
    delta_chunks = []
    alpha_chunks = []
    I_chunks = []
    
    for alpha_grid in alpha_grid_list:
        cos_alpha = np.cos(alpha_grid)[..., None]
        sin_alpha = np.sin(alpha_grid)[..., None]
    
        # A_full = A_Uchida @ R_z(-alpha)
        #A rotation by -alpha is the same as the inverse rotation of alpha R_z^(-1)(alpha) which is how we implement below, i.e cos_alpha remains unchanged and sin(-alpha) = -1*sin_alpha
        A_full = np.empty_like(A)
        #original
        #To be clear, the implementation below is equivalent to an alpha rotation matrix 
        #M = (cos(alpha), sin(alpha), 0)
        #    (-sin(alpha), cos(alpha), 0)
        #    (    0            0       1)
        #We have been careful to get this correct. The python convention can mess it up. 
        A_full[..., 0] = A[..., 0] * cos_alpha + A[..., 1] * -1*sin_alpha
        A_full[..., 1] = A[..., 0] * sin_alpha + A[..., 1] * cos_alpha
        A_full[..., 2] = A[..., 2]

        # Matrix B is constant
        B = np.array([
            [N/M, 0, H/M],
            [-H*K/(N*M), L/N, K/M],
            [-H*L/(N*M), -K/N, L/M]
        ])
        
        # sigma' = A_full @ sigma @ A_full.T  (batched transpose of last two axes)
        sigma_prime = A_full @ sigma @ np.transpose(A_full, (0, 1, 3, 2))
    
        # sigma'' = B @ sigma' @ B.T
        sigma_double_prime = B @ sigma_prime @ B.T  # [n_phi, n_psi, 3, 3]
    
        # Voigt round-trip for compliance contraction
        sigma_double_prime_voigt = stress_tensor_to_voigt(sigma_double_prime)
        epsilon_double_prime_voigt = np.einsum(
            'ij,xyj->xyi', elastic_compliance, sigma_double_prime_voigt
        )
        epsilon_double_prime = voigt_to_strain_tensor(epsilon_double_prime_voigt)
    
        # Invert B-transform without assuming orthonormality:  eps' = B.T @ eps'' @ B
        epsilon_prime = np.einsum(
            'ab,...bc,cd->...ad', B.T, epsilon_double_prime, B
        )
        strain_33_prime = epsilon_prime[..., 2, 2]
    
        # Collect this iteration's flattened outputs
        strain_33_chunks.append(strain_33_prime.ravel(order='F'))
        phi_chunks.append(np.degrees(phi_grid).ravel(order='F'))
        psi_chunks.append(np.degrees(psi_grid).ravel(order='F'))
        delta_chunks.append(delta_grid.ravel(order='F'))  # already in degrees
        alpha_chunks.append(np.degrees(alpha_grid).ravel(order='F'))

        if direct_PO:
            I_PO = PO_MODEL.intensity_from_orientation(hkl, phi_grid, psi_grid, alpha_grid)
            I_chunks.append(I_PO.ravel(order='F'))
    
    # --- Concatenate across all alpha iterations -------------------------------
    strain_33_list = np.concatenate(strain_33_chunks)
    phi_list       = np.concatenate(phi_chunks)
    psi_list       = np.concatenate(psi_chunks)
    delta_list     = np.concatenate(delta_chunks)
    alpha_list     = np.concatenate(alpha_chunks)

    if not PO_on:
        I_list = np.ones(strain_33_list.size)
    elif direct_PO:
        I_list = np.concatenate(I_chunks)  # aligns with strain_33_list shape
    else:
        # Evaluate the PO model at the (phi, delta) points actually in use. Interpolating
        # from a separate grid contributed most of the PO sampling error for no speed gain
        # (that grid was the same size as this one). A cubic interpolant is not a safe
        # alternative: it overshoots to negative values, and these weight the averages below.
        # PO surface plots are unaffected -- they sample intensity_for_hkl on their own grid.
        I_grid, phi_grid_PO, delta_grid_PO = PO_MODEL.intensity_for_hkl(
            hkl, np.degrees(phi_values), deltas)
        I_list = I_grid.ravel(order='F')

    # d0 and 2th
    d0 = get_d0(symmetry,h,k,l,a,b,c)
    if d0 == 0:
        d_strain = 0
        two_th = 0
    else:
        # strains
        d_strain = d0*(1-strain_33_list) #Positive t yields negative strains yields expanded d values
        # 2ths
        sin_th = wavelength / (2 * d_strain)
        two_th = 2 * np.degrees(np.arcsin(sin_th))

    hkl_label = f"{int(h)}{int(k)}{int(l)}"
    df = pd.DataFrame({
        "hkl" : hkl_label,
        "h": int(h),
        "k": int(k),
        "l": int(l),
        "strain_33": strain_33_list,
        "psi (degrees)": psi_list,
        "phi (degrees)": phi_list,
        "chi (degrees)": float(chi),
        "delta (degrees)": delta_list,
        "alpha (degrees)": alpha_list,
        "d strain": d_strain,
        "2th" : two_th,
        "intensity": intensity,
        "PO_intensity": I_list
    })

    #Insert a placeholder column for the average strain, 2th, intensity at each psi
    df["Mean strain @ psi"] = np.nan
    df["Mean two_th @ psi"] = np.nan
    df["Mean I @ psi"] = np.nan
    #Compute the average strains and append to df
    for psi in np.unique(psi_list):
        #Obtain all the strains at this particular psi
        #mask = psi_list == psi
        mask = np.isclose(psi_list, psi, atol=1e-4) #safer implementation
        strains = strain_33_list[mask]
        PO_intensity = I_list[mask]
        mean_strain = np.average(strains, weights = PO_intensity) #Average of the strains weighted by the PO
        mean_dstrain = d0*(1-mean_strain)
        mean_sin_th = wavelength / (2 * mean_dstrain)
        mean_two_th = 2 * np.degrees(np.arcsin(mean_sin_th))
        #Compute the average peak intensity at this psi
        av_I = intensity*np.mean(PO_intensity)
        #Update the mean_strain, mean_two_th column at the correct psi values
        df.loc[df["psi (degrees)"] == psi, ["Mean strain @ psi", "Mean two_th @ psi", "Mean I @ psi"]] = [mean_strain, mean_two_th, av_I]

    #Repeat but instead compute averages over deltas
    df["Mean strain @ delta"] = np.nan
    df["Mean two_th @ delta"] = np.nan
    df["Mean I @ delta"] = np.nan
    #Only compute if deltas are meaningful (skip Funamori-style placeholder case)
    if n_delta > 1:
        #Compute the average strains and append to df
        for delta in np.unique(delta_list):
            #Obtain all the strains at this particular delta
            mask = np.isclose(delta_list, delta, atol=1e-4) #safer implementation
            strains = strain_33_list[mask]
            PO_intensity = I_list[mask]
            mean_strain = np.average(strains, weights = PO_intensity) #Average of the strains weighted by the PO
            mean_dstrain = d0*(1-mean_strain)
            mean_sin_th = wavelength / (2 * mean_dstrain)
            mean_two_th = 2 * np.degrees(np.arcsin(mean_sin_th))
            #Compute the average peak intensity at this delta
            av_I = intensity*np.mean(PO_intensity)
            #Update the mean_strain, mean_two_th column at the correct psi values
            df.loc[df["delta (degrees)"] == delta, ["Mean strain @ delta", "Mean two_th @ delta", "Mean I @ delta"]] = [mean_strain, mean_two_th, av_I]

    # Group by hkl label and sort by azimuth
    df = df.sort_values(by=["hkl", "delta (degrees)"], ignore_index=True)

    return hkl_label, df, psi_list, strain_33_list

#Uses convolution of delta and Gaussian kernal for fast evaluation
def Generate_XRD(selected_hkls, intensities, Gaussian_FWHM, strain_sim_params, broadening=True, po_model=None):
    # --- Compute strain results ---
    all_dfs = [compute_strain(hkl, inten, *strain_sim_params, po_model=po_model)[1]
               for hkl, inten in zip(selected_hkls, intensities)]
    
    combined_df = pd.concat(all_dfs, ignore_index=True)

    # --- Define grid ---
    sigma_gauss = Gaussian_FWHM / (2 * np.sqrt(2 * np.log(2)))
    twotheta_min = combined_df["2th"].min() - 1
    twotheta_max = combined_df["2th"].max() + 1
    step = 0.0005 # In degrees
    twotheta_grid = np.arange(twotheta_min, twotheta_max, step)

    # --- Build normalized Gaussian kernel ---
    kernel_extent = 5 * sigma_gauss  # ±3σ window
    theta_kernel = np.arange(-kernel_extent, kernel_extent + step, step)
    gaussian_kernel = Gaussian(theta_kernel, 0, sigma_gauss)

    #Extract chi value from strain_sim_params (8th value in list)
    chi = strain_sim_params[7]

    # --- Build single global histogram with scaled contributions ---
    if broadening: 
        # Count number of contributions per (h,k,l)
        counts = combined_df.groupby(["h","k","l"])['intensity'].transform('size')
        
        # Vectorized weights: intensity / count
        weights = combined_df['intensity']*combined_df['PO_intensity'] / counts
        
        # Build histogram
        hist, _ = np.histogram(
            combined_df['2th'],
            bins=len(twotheta_grid),
            range=(twotheta_min, twotheta_max),
            weights=weights
        )
    else:
        if chi == 0: #Unique axial pattern with precomputed means
            # Singh pattern: one average peak per reflection
            mean_df = combined_df.drop_duplicates(subset=["h", "k", "l"])
            #Compute the mean intensity over 
            hist, _ = np.histogram(
                mean_df['Mean two_th @ delta'],
                bins=len(twotheta_grid),
                range=(twotheta_min, twotheta_max),
                weights=mean_df['Mean I @ delta']
            )
        else: 
            #Compute the mean across all the computed values
            mean_df = combined_df.groupby(["h","k","l"]).agg(
                {"2th": "mean",  # mean of the actual 2θ values per reflection
                "intensity": "mean", 
                "PO_intensity": "mean"
                })
            hist, _ = np.histogram(
                mean_df["2th"],  # the averaged 2θ
                bins=len(twotheta_grid),
                range=(twotheta_min, twotheta_max),
                weights=mean_df["intensity"]*mean_df["PO_intensity"]
            )
    # Convolve using FFT
    total_pattern = fftconvolve(hist, gaussian_kernel, mode="same")
    # Output as DataFrame
    total_df = pd.DataFrame({
        "2th": twotheta_grid[::5],
        "Total Intensity": total_pattern[::5]
    })
    return total_df

def batch_XRD(batch_upload, selected_hkls=None, intensities=None, Gaussian_FWHM=None, Funamori_broadening=None, po_model=None):
    batch_upload.seek(0)  # reset pointer
    # Read everything into a DataFrame
    df = pd.read_csv(batch_upload)

    # Convert numerical columns where possible
    for col in df.columns:
        try:
            df[col] = pd.to_numeric(df[col])
        except:
            pass

    # Store parameters in one DataFrame
    parameters_df = df.copy()
    # Store results side-by-side
    results_blocks = []

    phi_values = np.arange(0,360,2)
    phi_values = np.radians(phi_values)
    psi_values = 0

    for idx, row in df.iterrows():
        #Check the required columns are given for the respective symmetry
        symmetry = row["symmetry"]
        if symmetry == "cubic":
            required_keys = {'a','b','c','alpha','beta','gamma','wavelength','C11','C12','C44','sig11','sig22','sig33','chi'}
        elif symmetry == "hexagonal":
            required_keys = {'a','b','c','alpha','beta','gamma','wavelength','C11','C33','C12','C13','C44','sig11','sig22','sig33','chi'}
        elif symmetry == "tetragonal_A":
            required_keys = {'a','b','c','alpha','beta','gamma','wavelength','C11','C33','C12','C13','C44','C66','sig11','sig22','sig33','chi'}
        elif symmetry == "tetragonal_B":
            required_keys = {'a','b','c','alpha','beta','gamma','wavelength','C11','C33','C12','C13','C16','C44','C66','sig11','sig22','sig33','chi'}
        elif symmetry == "orthorhombic":
            required_keys = {'a','b','c','alpha','beta','gamma','wavelength','C11','C22','C33','C12','C13','C23','C44','C55','C66','sig11','sig22','sig33','chi'}
        elif symmetry == "trigonal_A":
            required_keys = {'a','b','c','alpha','beta','gamma','wavelength','C11','C33','C12','C13','C14','C44','sig11','sig22','sig33','chi'}
        else:
            st.error("{} symmetry is not yet supported".format(symmetry))
            required_keys = {}
        if not required_keys.issubset(df.columns):
            st.error(f"CSV must contain: {', '.join(required_keys)}")
            st.stop()
        # Extract row parameters for strain_sim_params
        #Get the lattice parameters
        # Extract lattice parameters
        lat_params = {
            "a_val": row["a"],
            "b_val": row["b"],
            "c_val": row["c"],
            "alpha": row["alpha"],
            "beta": row["beta"],
            "gamma": row["gamma"],
        }
        #Get the cij_params
        cij_params = {
            col.lower(): row[col]
            for col in df.columns
            if col.upper().startswith("C") and col[1:].isdigit()
        }
        #Get the stress params
        sig_params = {
            key: row[key]
            for key in ['sigma_11','sigma_22','sigma_33','sigma_12','sigma_13','sigma_23']
        }
        # Combine into strain_sim_params
        strain_sim_params = (
            row["symmetry"],
            lat_params,
            row["wavelength"],
            cij_params,
            sig_params,
            row["chi"],
            phi_values,
            psi_values,
        )
        # Run Generate_XRD for this row
        xrd_df = Generate_XRD(selected_hkls, intensities, Gaussian_FWHM, strain_sim_params, Funamori_broadening, po_model=po_model)
        # Rename columns so each block is unique
        xrd_df = xrd_df.rename(columns={
            "2th": f"2th_iter{idx+1}",
            "Total Intensity": f"Intensity_iter{idx+1}"
        }).reset_index(drop=True)

        results_blocks.append(xrd_df)

    # Align all result blocks by index and combine
    results_df = pd.concat(results_blocks, axis=1)

    return parameters_df, results_df, results_blocks

def cake_data(selected_hkls, intensities, symmetry, lattice_params, wavelength, cijs, sigma_params, chi, po_model=None):
    """
    Computes the azimuth vs 2th strain data for each hkl and combines into a dictionary with entries for each hkl

    Returns:
    cake_dict
    keys (hkl_labels) : values (df of information for this hkl)
    """
    cake_dict = {}

    # 72 phi points (5 deg) as standard; finer only when the March-Dollase surface is
    # sharply peaked, which happens for both small and large R (hence max(R, 1/R)).
    n_phi = (max(72, cp.adaptive_n_phi(st.session_state.params.get("R")))
             if st.session_state.params.get("PO_toggle") else 72)

    for hkl, intensity in zip(selected_hkls, intensities):
        phi_values = np.radians(np.linspace(0, 360, n_phi, endpoint=False))
        psi_values = 0  # let compute_strain calculate psi for each HKL
        hkl_label, df, psi_list, strain_33_list = compute_strain(
            hkl, intensity, symmetry, lattice_params, wavelength, cijs,
            sigma_params, chi, phi_values, psi_values, po_model=po_model
        )
        cake_dict[hkl_label] = df
    
    return cake_dict

def run_refinement(params, refine_flags, selected_hkls, selected_indices, intensities, Gaussian_FWHM, phi_values, psi_values, wavelength, symmetry, x_exp, y_exp, lattice_params, cijs,
                   sigma_params, chi, Funamori_broadening, po_model=None):
    """
    Parameters:
        params (dict): Current parameter values
        refine_flags (dict): Dict of booleans indicating which params to refine
        selected_hkls, selected_indices, intensities, Gaussian_FWHM, phi_values, psi_values, wavelength, symmetry:
            Experimental/simulation data and settings.
        x_exp, y_exp: Experimental x (2θ) and intensity data.
    
    Returns:
        result (lmfit.MinimizerResult): Refinement result object.
    """
    # Build lmfit.Parameters
    lm_params = Parameters()
    for name, val in params.items():
        if name in ["t",'sigma_12','sigma_13', 'sigma_23']:
            min_val, max_val = -25, 25
        elif "c" in name.lower():  # elastic constants
            min_val, max_val = 0.5 * val, 1.5 * val
        elif name == "a_val" or name == "b_val" or name == "c_val":
            min_val, max_val = 0.75 * val, 1.25 * val
        elif name == "chi":
            min_val, max_val = -90, 90
        else:
            min_val, max_val = None, None

        if refine_flags.get(name, False):
            lm_params.add(name, value=val, min=min_val, max=max_val)
        else:
            lm_params.add(name, value=val, vary=False)
        
    # Handle peak intensities separately 
    if refine_flags.get("peak_intensity", False):
        for i, inten in zip(selected_indices, intensities):
            lm_params.add(f"intensity_{i}", value=inten, min=0, max=1000)
    else:
        for i, inten in zip(selected_indices, intensities):
            lm_params.add(f"intensity_{i}", value=inten, vary=False)

    st.write(lm_params)

    # Run first iteration of refinement to determine common 2th domain
    intensities_opt = [lm_params[f"intensity_{i}"].value for i in selected_indices]
    strain_sim_params = (symmetry, lattice_params, wavelength, cijs, sigma_params, chi, phi_values, psi_values)
    
    # Generate simulated pattern
    XRD_df = Generate_XRD(selected_hkls, intensities_opt, Gaussian_FWHM, strain_sim_params, Funamori_broadening, po_model=po_model)
    twoth_sim = XRD_df["2th"].values

    # Use overlap between simulation and experiment to set interpolation range. Fixed for subsequent iterations
    #The range is slightly less than that returned by the simulation to eliminate NaN values in evaluating the interpolated data
    x_min_sim = np.min(twoth_sim) + 0.5
    x_max_sim = np.max(twoth_sim) - 0.5
    mask = (x_exp >= x_min_sim) & (x_exp <= x_max_sim)
    x_exp_common = x_exp[mask]
    y_exp_common = y_exp[mask]

    #Here we also determine the x_indices definining the binning around each peak for residual weighting
    #First we need the 2th center positions of each hkl reflection d (use the mean "Singh" position)
    hkl_peak_centers = []
    a = lattice_params.get("a_val")
    b = lattice_params.get("b_val")
    c = lattice_params.get("c_val")
    for hkl, inten in zip(selected_hkls, intensities_opt):
        df = compute_strain(hkl, inten, *strain_sim_params, po_model=po_model)[1]
        #Compute the average of the mean_2th values (For axial, this averages over many identical values, for radial, we average across a range of psi at fixed delta)
        mean_2th = np.mean(df["Mean two_th @ delta"])
        h, k, l = hkl
        #Compute d0 and 2th
        d0 = get_d0(symmetry,h,k,l,a,b,c)
        #Compute 2ths
        sin_th = wavelength / (2 * d0)
        two_th = 2 * np.degrees(np.arcsin(sin_th))
        hkl_peak_centers = np.append(hkl_peak_centers, mean_2th)

    #Get the residual bin indices using these centers
    bin_indices = compute_bin_indices(x_exp_common, hkl_peak_centers, Gaussian_FWHM)

    # --- Wrapped cost function that implements this fixed domain ---
    def wrapped_cost_function(lm_params):
        return cost_function(lm_params, refine_flags, selected_hkls, selected_indices, Gaussian_FWHM,
            phi_values, psi_values, wavelength, symmetry,
            x_exp_common, y_exp_common, bin_indices, Funamori_broadening, global_lattice_params=lattice_params, global_cijs=cijs, global_sigmas=sigma_params, po_model=po_model
        )

    # Run optimization
    result = minimize(wrapped_cost_function, lm_params, method="leastsq", gtol=1e-8,)
    #-------------------------------------------------

    return result

def cost_function(lm_params, refine_flags, selected_hkls, selected_indices,
                  Gaussian_FWHM, phi_values, psi_values, wavelength, symmetry,
                  x_exp_common, y_exp_common, bin_indices,
                  Funamori_broadening, global_lattice_params, global_cijs, global_sigmas, po_model=None):
    """
    lm_params: current parameters from lmfit
    global_lattice: dictionary containing full lattice info (a_val, b_val, c_val, alpha, beta, gamma)
    global_cijs: dictionary containing the full set of elastic constants
    global_sigma: dictionary containing the full set of stress coefficients
    """

    # --- Lattice parameters: use lm_params if refining, else global values ---
    lattice_params = {}
    for key in ["a_val", "b_val", "c_val", "alpha", "beta", "gamma"]:
        if key in lm_params:
            lattice_params[key] = lm_params[key].value
        else:
            lattice_params[key] = global_lattice_params[key]

    cijs = {}
    for k in global_cijs:
        cijs[k] = lm_params[k].value if k in lm_params else global_cijs[k]
        
    # Stress parameters
    t = lm_params["t"].value
    sigma_params = {
        'sigma_11' : -t / 3,
        'sigma_22' : -t / 3,
        'sigma_33' : 2 * t / 3
    }
    for key in ['sigma_12','sigma_13','sigma_23']:
        sigma_params[key] = lm_params[key].value

    chi = lm_params["chi"].value

    intensities_opt = [lm_params[f"intensity_{i}"].value for i in selected_indices]

    strain_sim_params = (symmetry, lattice_params, wavelength, cijs, sigma_params, chi, phi_values, psi_values)
    XRD_df = Generate_XRD(selected_hkls, intensities_opt, Gaussian_FWHM, strain_sim_params, Funamori_broadening, po_model=po_model)
    twoth_sim = XRD_df["2th"]
    intensity_sim = XRD_df["Total Intensity"]

    interp_sim = interp1d(twoth_sim, intensity_sim, bounds_error=False, fill_value=0)
    y_sim_common = interp_sim(x_exp_common)

    residuals = np.asarray(y_exp_common - y_sim_common)

    # Peak position binned normalization of residuals
    norm_residuals = []
    for idx_range in bin_indices:
        if len(idx_range) == 0:
            continue  # skip empty bins
        res_bin = residuals[idx_range]
        y_bin = y_exp_common[idx_range]

        norm = np.max(np.abs(y_bin)) if np.max(np.abs(y_bin)) != 0 else 1
        norm_residuals.append(res_bin / norm)

    #Combine bins into a single array of weighted residuals
    weighted_residuals = np.concatenate(norm_residuals)
    return weighted_residuals

def generate_epsilon_psi_curves(selected_hkls, psi_steps, phi_steps, alpha_steps, intensities=None, symmetry=None, lattice_params=None, wavelength=None, cijs=None, sigma_params=None, chi=None, po_model=None):

    results_dict = {}
    psi_values   = np.linspace(0, np.pi / 2, int(psi_steps))                         # output/plot axis (endpoints kept)
    phi_values   = np.linspace(0, 2 * np.pi, int(phi_steps),   endpoint=False)       # periodic integration axis
    alpha_values = np.linspace(-np.pi, np.pi, int(alpha_steps), endpoint=False)      # periodic integration axis

    fig = make_subplots(
        rows=len(selected_hkls),
        cols=1,
        shared_xaxes=False,
        vertical_spacing=0.06,
        subplot_titles=[f"ε′₃₃ [hkl = ({hkl})]" for hkl in selected_hkls]
    )

    for i, (hkl, intensity) in enumerate(zip(selected_hkls, intensities), start=1):
        hkl_label, df, psi_list, strain_33_list = compute_strain(hkl, intensity, symmetry, lattice_params,
                                                                 wavelength, cijs, sigma_params,
                                                                 chi, phi_values, psi_values, alpha_values, po_model=po_model
        )

        results_dict[hkl_label] = df
        psi_array = np.asarray(psi_list)
        strain_array = np.asarray(strain_33_list)

        #Get the combined intensity from PO model and ideal intensity
        combined_I = df["intensity"]*df["PO_intensity"]
        norm = Normalize(vmin=0, vmax=np.max(combined_I))
        normed_I = norm(combined_I)
        #Set the opacity if PO model is in use
        if st.session_state.params.get("PO_toggle"):
            OPACITY = normed_I
        else: #isotropic case
            OPACITY = 0.15

        fig.add_trace(
            go.Scattergl(
                x=psi_array,
                y=strain_array,
                mode="markers",
                marker=dict(
                    size=2,
                    color="black",
                    opacity = OPACITY
                ),
                showlegend=False
            ),
            row=i, col=1
        )

        # Plot the mean strain curve (vectorised)
        mean_df = (df.groupby("psi (degrees)", sort=True)["Mean strain @ psi"].first().reset_index())

        fig.add_trace(go.Scatter(x=mean_df["psi (degrees)"],
                                 y=mean_df["Mean strain @ psi"],
                                 mode="lines",
                                 line=dict(width=2, color="red"),
                                 name="Mean strain" if i == 1 else None,
                                 showlegend=(i == 1)),
                      row=i, col=1
        )
        # Reference lines
        fig.add_hline(y=0, line_width=1, row=i, col=1)
        fig.add_vline(x=54.7, line_dash="dash", line_width=1, row=i, col=1) #Magic angle
        
        fig.update_yaxes(autorange=True, row=i, col=1)

    fig.update_xaxes(title="ψ (degrees)", title_font=dict(size=18), tickfont=dict(size=14), range=[0, 90])
    fig.update_yaxes(title="ε′₃₃", title_font=dict(size=18), tickfont=dict(size=14))
    fig.update_layout(height=450 * len(selected_hkls),hovermode="closest")

    st.plotly_chart(fig,
                    width="stretch",
                    config={"scrollZoom": False}  # Disables wheel zoom
    )
    return results_dict

def setup_refinement_toggles(lattice_params, symmetry=None, **additional_fields):
    """
    Returns editable parameter fields and refinement toggles dynamically.
    
    Returns:
        params (dict): Updated parameter values.
        refine_flags (dict): Booleans for whether each parameter is set to refine.
    """
    combined_params = {}

    # Start with lattice parameters
    combined_params.update(lattice_params)

    # Merge any additional dictionaries passed as keyword arguments
    for name, subdict in additional_fields.items():
        if not isinstance(subdict, dict):
            raise TypeError(f"Expected dict for '{name}', got {type(subdict).__name__}")
        combined_params.update(subdict)
        
    #Build appropriate parameter dictionary
    p_dict = {}
    p_dict["a_val"] = combined_params["a_val"]
    p_dict["c11"] = combined_params["c11"]
    p_dict["c12"] = combined_params["c12"]
    p_dict["c44"] = combined_params["c44"]
    p_dict["t"] = combined_params["sigma_33"] - combined_params["sigma_11"]
    #Off diagonal stress terms
    p_dict["sigma_12"] = combined_params["sigma_12"]
    p_dict["sigma_13"] = combined_params["sigma_13"]
    p_dict["sigma_23"] = combined_params["sigma_23"]
    p_dict["chi"] = combined_params["chi"]

    #Symmetry specific refineable parameters
    if symmetry == "cubic":
        pass #Already all included
    elif symmetry == "hexagonal":
        p_dict["c_val"] = combined_params["c_val"]
        p_dict["c33"] = combined_params["c33"]
        p_dict["c13"] = combined_params["c13"]
    elif symmetry == "tetragonal_A":
        p_dict["c_val"] = combined_params["c_val"]
        p_dict["c33"] = combined_params["c33"]
        p_dict["c13"] = combined_params["c13"]
        p_dict["c66"] = combined_params["c66"]
    elif symmetry == "tetragonal_B":
        p_dict["c_val"] = combined_params["c_val"]
        p_dict["c33"] = combined_params["c33"]
        p_dict["c13"] = combined_params["c13"]
        p_dict["c16"] = combined_params["c16"]
        p_dict["c66"] = combined_params["c66"]
    elif symmetry == "orthorhombic":
        p_dict["b_val"] = combined_params["b_val"]
        p_dict["c_val"] = combined_params["c_val"]
        p_dict["c22"] = combined_params["c22"]
        p_dict["c33"] = combined_params["c33"]
        p_dict["c13"] = combined_params["c13"]
        p_dict["c23"] = combined_params["c23"]
        p_dict["c55"] = combined_params["c55"]
        p_dict["c66"] = combined_params["c66"]
    elif symmetry == "trigonal_A":
        p_dict["c_val"] = combined_params["c_val"]
        p_dict["c33"] = combined_params["c33"]
        p_dict["c13"] = combined_params["c13"]
        p_dict["c13"] = combined_params["c14"]
    else:
        st.error("{} symmetry is not yet supported".format(symmetry))
        
    if "refinement_params" not in st.session_state:
        st.session_state.ref_params = p_dict.copy()

    if "refine_flags" not in st.session_state:
        # If no refine defaults given, all False
        st.session_state.refine_flags = {k: False for k in p_dict}
        st.session_state.refine_flags["peak_intensity"] = False  # default for peak intensities

    st.subheader("Refinement Parameters (Select to refine)")

    for key, default_val in p_dict.items():
        col1, col2 = st.columns([1, 1])
        with col1:
            st.session_state.refine_flags[key] = st.checkbox(
                f"{key}",
                value=st.session_state.refine_flags.get(key, False),
                key=f"chk_{key}"
            )
    with col1:
        # --- Add peak intensity refinement checkbox separately ---
        st.session_state.refine_flags["peak_intensity"] = st.checkbox(
        "Refine peak intensities",
        value=st.session_state.refine_flags.get("peak_intensity", False),
        key="chk_peak_intensity"
        )
    return st.session_state.ref_params, st.session_state.refine_flags

def generate_cake_figures(results_dict, selected_hkls, broadening, chi=None):

    fig, axs = plt.subplots(1, 1, figsize=(8, 5))
    fig2, axs2 = plt.subplots(len(selected_hkls), 1, figsize=(8, 5 * len(selected_hkls)))
    
    # Cake plot
    if broadening == True:
        for df in results_dict.values():
            #Normalise the intensities to get the opacity
            combined_I = df["intensity"]*df["PO_intensity"]
            norm = Normalize(vmin=0, vmax=np.max(combined_I))
            normed_I = norm(combined_I)
            #Plot all the data
            axs.scatter(df["2th"], df["delta (degrees)"], 
                        color="black",
                        marker = '.', 
                        s=2, 
                        alpha = normed_I
                       )
    else:
        if chi == 0: #unique option for axial geometry
            for df in results_dict.values():
                #Plot only the mean value for each delta
                deltas = np.unique(df["delta (degrees)"].values)
                mean_2ths = np.full(len(np.unique(df["delta (degrees)"].values)),df["Mean two_th @ delta"].iloc[0])
                #Need to average the intensities across phi for each delta
                # Average PO_intensity across phi for each delta
                mean_PO_intensity = (
                    df.groupby("delta (degrees)")["PO_intensity"]
                      .mean()
                      .reindex(deltas)  # ensure same order as deltas
                      .values
                )
                norm = Normalize(vmin=0, vmax=np.max(mean_PO_intensity))
                normed_I = norm(mean_PO_intensity)
                axs.scatter(mean_2ths, deltas, 
                            color="black",
                            marker = '.', 
                            s=2,
                            alpha=normed_I
                           )
        else: #Transverse geometry with broadening off
            for df in results_dict.values():
                unique = df.drop_duplicates(subset="delta (degrees)") #Pick out the entries for unique delta values
                mean_2th = unique["Mean two_th @ delta"].values
                deltas = unique["delta (degrees)"].values
                # Average PO_intensity across phi for each delta
                mean_PO_intensity = (
                    df.groupby("delta (degrees)")["PO_intensity"]
                      .mean()
                      .reindex(deltas)  # ensure same order as deltas
                      .values
                )
                norm = Normalize(vmin=0, vmax=np.max(mean_PO_intensity))
                normed_I = norm(mean_PO_intensity)
                axs.scatter(mean_2th, deltas, 
                            color="black",
                            marker = '.', 
                            s=2,
                            alpha=normed_I
                           )
    axs.set_xlabel("2th (degrees)")
    axs.set_ylabel("azimuth (degrees)")
    axs.set_title("Cake")
    axs.set_ylim(-180, 180)
    plt.tight_layout()
    st.pyplot(fig)
    
    if len(selected_hkls) == 1:
        axs2 = [axs2]
    for ax, hkl_label in zip(axs2, results_dict.keys()):
        df = results_dict[hkl_label]
        delta_list = df["delta (degrees)"]
        strain_33_list = df["strain_33"]
        scatter = ax.scatter(delta_list, strain_33_list, color="black", s=0.2, alpha=0.1)
        ax.hlines(0,-180,180, color="black", lw=0.8)

        #Plot the mean strain curve
        unique_delta = np.unique(delta_list)
        mean_strain_list = [df[df["delta (degrees)"]==d]["Mean strain @ delta"].iloc[0] for d in unique_delta]
        ax.plot(unique_delta, mean_strain_list, color="red", lw=0.8, label="mean strain (δ)")
        #Add average over all crystallites
        complete_mean = np.mean(mean_strain_list)
        ax.hlines(complete_mean,-180,180, color="black", ls="dashed", lw=0.8, label="Average:{}".format(np.round(complete_mean,6)))
        
        ax.set_xlabel("azimuth (degrees)")
        ax.set_ylabel("ε′₃₃")
        ax.set_title(f"Strain ε′₃₃ for hkl = ({hkl_label})")
        plt.tight_layout()
        ax.legend()
    st.pyplot(fig2)
