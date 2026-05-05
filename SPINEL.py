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

#3d plotting
from mpl_toolkits.mplot3d import Axes3D

from scipy.interpolate import RegularGridInterpolator

st.markdown("""
<style>
html, body, [class*="css"]  {
    font-size: 12px !important;   /* Adjust this value to your desired size */
}

/* Smaller widget labels */
label, .stTextInput label, .stNumberInput label, .stSelectbox label {
    font-size: 12px !important;
}

/* Smaller number + text input text */
input, textarea, select {
    font-size: 12px !important;
}

/* Smaller checkbox labels */
.stCheckbox label {
    font-size: 12px !important;
}

/* Smaller markdown text */
p, span, div {
    font-size: 12px !important;
}

/* Make headers smaller too */
h1, h2, h3, h4, h5 {
    font-size: 18px !important;
}

/* Reduce vertical gaps between all widgets */
.stNumberInput, .stTextInput, .stSelectbox, .stSlider, .stCheckbox {
    margin-top: 0.1rem !important;
    margin-bottom: 0.1rem !important;
}

/* Reduce extra padding around Streamlit containers */
div[data-testid="stVerticalBlock"] {
    gap: 0.1rem !important;
}
</style>
""", unsafe_allow_html=True)

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
    
def compute_strain(hkl, intensity, symmetry, lattice_params, wavelength, cij_params, sigma_params, chi, phi_values, psi_values):
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
 
    #Check if psi_values are given or if it must be calculated for XRD generation
    if isinstance(psi_values, int):
        d0 = get_d0(symmetry,h,k,l,a,b,c)
        sin_theta0 = wavelength / (2 * d0)
        theta0 = np.arcsin(sin_theta0)
        if psi_values==0: #Standard setting for fine-resolution XRD generation
            deltas = np.arange(-180,180,5)
            #Check if chi value is zero (axial case) or non-zero (radial)
            if chi == 0: 
                # return only one psi_value assuming compression axis aligned with X-rays
                psi_values = np.asarray([np.pi/2 - theta0])
            else:
                #Assume chi is non-zero (radial) and compute a psi for each azimuth bin (delta)
                deltas_rad = np.radians(deltas)
                chi_rad = np.radians(chi)
                psi_values = np.arccos(np.sin(chi_rad)*np.cos(deltas_rad)*np.cos(theta0)+np.cos(chi_rad)*np.sin(theta0))
        else: #A coarser resolution option for XRD refinement (less expensive due to fewer refinement iterations required)
            deltas = np.arange(-180,180,12)
            #Check if chi value is zero (axial case) or non-zero (radial)
            if chi == 0: 
                # return only one psi_value assuming compression axis aligned with X-rays
                psi_values = np.asarray([np.pi/2 - theta0])
            else:
                #Assume chi is non-zero (radial) and compute a psi for each azimuth bin (delta)
                deltas_rad = np.radians(deltas)
                chi_rad = np.radians(chi)
                psi_values = np.arccos(np.sin(chi_rad)*np.cos(deltas_rad)*np.cos(theta0)+np.cos(chi_rad)*np.sin(theta0))
    else:
        # Assume phi_values and psi_values are 1D numpy arrays. This part is needed for Funamori plots as psi is not evaluated from delta here
        psi_values = np.asarray(psi_values)
        #Add deltas placeholder for completeness
        deltas = np.zeros(len(psi_values))
    #phi_values are always passed to the function
    phi_values = np.asarray(phi_values)

    #OLD meshgrid construction generating incorrect pairing
    #cos_phi = np.cos(phi_values)
    #sin_phi = np.sin(phi_values)
    #cos_psi = np.cos(psi_values)
    #sin_psi = np.sin(psi_values)
    # Create meshgrids for broadcasting
    #cos_phi, cos_psi = np.meshgrid(cos_phi, cos_psi, indexing='ij')
    #sin_phi, sin_psi = np.meshgrid(sin_phi, sin_psi, indexing='ij')

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
    #A[..., 0, 2] = cos_phi * sin_psi
    #A[..., 1, 0] = sin_phi * cos_psi
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
    # Uchida's a_ij (Eq. 11) places x'_3 in the x_2-x_3 plane regardless of
    # delta. This is correct only for axially symmetric stress about Z_S.
    # For general stress, x'_3 must track the diffracting-plane normal Q in
    # the sample frame K_S as delta varies. Q in K_S (chi = 90 case, X-rays
    # along +Y_S):
    #     Q = (cos(theta) sin(delta), -sin(theta), cos(theta) cos(delta))
    # Solving R_z(alpha) . (0, sin psi, cos psi) = Q gives
    #     alpha = atan2(-cos(theta) sin(delta), -sin(theta))
    # so cos(psi) = Q_z = cos(theta) cos(delta), consistent with the Singh
    # formula already used to build psi_grid above.
    #
    # A_full = A_Uchida @ R_z(-alpha): mixes columns 0 and 1, leaves col 2.
    # For axial sigma (sigma_11 = sigma_22, off-diagonals zero) this collapses
    # back to the original Uchida result; for non-axial sigma it reproduces
    # the lab-azimuth dependence (Merkel 2006 Fig. 3 c-f).
    delta_grid_rad = np.radians(delta_grid)
    alpha_grid = np.arctan2(-np.cos(theta0) * np.sin(delta_grid_rad),
                            -np.sin(theta0))
    cos_alpha = np.cos(alpha_grid)[..., None]
    sin_alpha = np.sin(alpha_grid)[..., None]

    A_full = np.empty_like(A)
    A_full[..., 0] = A[..., 0] * cos_alpha - A[..., 1] * sin_alpha
    A_full[..., 1] = A[..., 0] * sin_alpha + A[..., 1] * cos_alpha
    A_full[..., 2] = A[..., 2]
    
    # Matrix B is constant
    B = np.array([
        [N/M, 0, H/M],
        [-H*K/(N*M), L/N, K/M],
        [-H*L/(N*M), -K/N, L/M]
    ])
    
    # Apply rotation: sigma' = A @ sigma @ A.T
    # This transposes the last two axes of A, swapping the 2 and 3 dimensions, e.g. If A has shape (N, M, 3, 3), then np.transpose(A, (0, 1, 3, 2)) gives shape (N, M, 3, 3), 
    #equivalent of computing A.T for each element of the batch. We cannot simply transpose everything since the batch structure would break down.
    #sigma_prime = A @ sigma @ np.transpose(A, (0, 1, 3, 2))
    sigma_prime = A_full @ sigma @ np.transpose(A_full, (0, 1, 3, 2)) #Performs transformation including alpha rotation
    
    # Apply B transform: sigma'' = B @ sigma' @ B.T
    sigma_double_prime = B @ sigma_prime @ B.T  # shape: [n_phi, n_psi, 3, 3]

    #Convert sigma tensor to voigt form [N,M,3,3] to [N,M,6]
    sigma_double_prime_voigt = stress_tensor_to_voigt(sigma_double_prime)  

    # Computes the strain from the elastic compliance and the stress matrix using einsum
    # ε'' = S ⋅ σ''
    #Here is the equivalent code written explicitly for explaination that the einsum is performing where stress matrix has shape (X, Y, 6) and the resulting strain has shape (X, Y, 6)
        #for x in range(X):
        #for y in range(Y):
        #    for i in range(6):
        #        epsilon[x, y, i] = sum(S[i, j] * sigma[x, y, j] for j in range(6))
    epsilon_double_prime_voigt = np.einsum('ij,xyj->xyi', elastic_compliance, sigma_double_prime_voigt)

    #Convert from Voigt to full strain tensor
    ε_double_prime = voigt_to_strain_tensor(epsilon_double_prime_voigt)
    
    # Get ε'_33 component
    #b13, b23, b33 = B[0, 2], B[1, 2], B[2, 2]
    #strain_33_prime = (
    #    b13**2 * ε_double_prime[..., 0, 0] +
    #    b23**2 * ε_double_prime[..., 1, 1] +
    #    b33**2 * ε_double_prime[..., 2, 2] +
    #    2 * b13 * b23 * ε_double_prime[..., 0, 1] +
    #    2 * b13 * b33 * ε_double_prime[..., 0, 2] +
    #    2 * b23 * b33 * ε_double_prime[..., 1, 2]
    #)

    #Avoid assumption of orthonormality of B when mapping back from double_prime to prime coordinates
    #Inverts the B matrix transformation without specifying the componets
    epsilon_prime = np.einsum(
        'ab,...bc,cd->...ad',
        B.T,
        ε_double_prime,
        B
    )
    #The strain component is now just the sigma'_33 term
    strain_33_prime = epsilon_prime[..., 2, 2]

    #CODE BLOCK IS REDUNDANT FROM OLD METHOD OF FLATTENING
    # Ensure deltas match the length of flattened psi/phi/strain lists
    #if psi_values.size == 1 and len(deltas) > 1:
        # Single psi, multiple deltas (axial simulation case) — replicate results for each delta
    #    n_phi = len(phi_values)
    #    n_delta = len(deltas)
    
        # Flatten the strain grid (shape [n_phi]) and replicate for each delta
    #    strain_33_prime = np.tile(strain_33_prime, (n_delta, 1)).T  # shape (n_phi, n_delta)
    
        # Also replicate psi and phi grids so they align with deltas
    #    psi_grid = np.full((n_phi, n_delta), psi_values[0])
    #    phi_grid = np.tile(phi_values[:, np.newaxis], (1, n_delta))
    #    delta_grid = np.tile(deltas, (n_phi, 1))
    #else:
        # Normal case — psi and phi already form a meshgrid
    #    phi_grid, psi_grid = np.meshgrid(phi_values, psi_values, indexing='ij')
    #    x, delta_grid = np.meshgrid(phi_values, deltas, indexing='ij') #Generate delta_grid needed for PO

    # Convert psi and phi grid to degrees for output
    #psi_deg_grid = np.degrees(psi_grid)
    #phi_deg_grid = np.degrees(phi_grid)
    #delta_deg_grid = delta_grid
    #psi_list = psi_deg_grid.ravel(order='F')
    #phi_list = phi_deg_grid.ravel(order='F')
    #strain_33_list = strain_33_prime.ravel(order='F')
    # Repeat deltas so every phi/psi pair gets one. This way the ordering of the deltas is correct to match up the delta,psi,phi,strain
    #delta_list = np.repeat(deltas, len(phi_values))

    #New flattening code (works because all output are already grids and have same consistent shapes)
    psi_deg_grid = np.degrees(psi_grid)
    phi_deg_grid = np.degrees(phi_grid)
    delta_deg_grid = delta_grid
    psi_list = psi_deg_grid.ravel(order='F')
    phi_list = phi_deg_grid.ravel(order='F')
    delta_list = delta_deg_grid.ravel(order='F')
    strain_33_list = strain_33_prime.ravel(order='F')

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
        "d strain": d_strain,
        "2th" : two_th,
        "intensity": intensity
    })

    #Insert a placeholder column for the intensity for each phi, psi pair computed from the PO model
    I_list = np.ones(np.shape(delta_list)) #It will have the shape of the delta_list
    df["PO_intensity"] = I_list

    if st.session_state.params.get("PO_toggle"):
        components = [
            {"tau": st.session_state.params.get("tau"), "rho": st.session_state.params.get("rho"),"R": st.session_state.params.get("R") , "weight" : st.session_state.params.get("weight")
            }
        ]
        hkl_POD = st.session_state.params.get("hkl_POD")
        PO_MODEL = PO.PO_Model(po_model=po_model,
                               components=components,
                               baseline=st.session_state.params.get("baseline"),
                               symmetry = symmetry,
                               wavelength = wavelength,
                               lattice_params = lattice_params,
                               chi_deg = chi,
                               POD_xtal = hkl_POD
                              )
        
        phi_PO = np.linspace(0,360,32)
        delta_PO = np.linspace(-180,180,32)                   
        I_grid, phi_grid_PO, delta_grid_PO, = PO_MODEL.intensity_for_hkl(hkl, phi_PO, delta_PO)

        #Evaluate the PO intensity
        x = phi_grid_PO[:, 0] 
        y = delta_grid_PO[0, :] 
        interp_func = RegularGridInterpolator((x, y), I_grid)
        new_points = np.stack([phi_deg_grid.ravel(), delta_deg_grid.ravel()], axis=-1)
        I_new = interp_func(new_points).reshape(len(phi_values), len(deltas))
        I_list = I_new.ravel(order='F')
        df["PO_intensity"] = I_list

    #Insert a placeholder column for the average strain, 2th, intensity at each psi
    df["Mean strain"] = np.nan
    df["Mean two_th"] = np.nan
    df["Mean I @ psi"] = np.nan
    #Initialise a list of the mean strains
    #mean_strain_list = []
    #mean_2th_list = []
    #mean_I_list = []
    #Compute the average strains and append to df
    for psi in np.unique(psi_list):
        #Obtain all the strains at this particular psi
        #mask = psi_list == psi
        mask = np.isclose(psi_list, psi, atol=1e-8) #safer implementation
        strains = strain_33_list[mask]
        PO_intensity = I_list[mask]
        mean_strain = np.average(strains, weights = PO_intensity) #Average of the strains weighted by the PO
        mean_dstrain = d0*(1-mean_strain)
        mean_sin_th = wavelength / (2 * mean_dstrain)
        mean_two_th = 2 * np.degrees(np.arcsin(mean_sin_th))
        #Compute the average peak intensity at this psi
        av_I = intensity*np.mean(PO_intensity)
        #Update the mean_strain, mean_two_th column at the correct psi values
        df.loc[df["psi (degrees)"] == psi, ["Mean strain", "Mean two_th", "Mean I @ psi"]] = [mean_strain, mean_two_th, av_I]

    # Group by hkl label and sort by azimuth
    df = df.sort_values(by=["hkl", "delta (degrees)"], ignore_index=True)

    return hkl_label, df, psi_list, strain_33_list

#Uses convolution of delta and Gaussian kernal for fast evaluation
def Generate_XRD(selected_hkls, intensities, Gaussian_FWHM, strain_sim_params, broadening=True):
    # --- Compute strain results ---
    all_dfs = [compute_strain(hkl, inten, *strain_sim_params)[1]
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
                mean_df['Mean two_th'],
                bins=len(twotheta_grid),
                range=(twotheta_min, twotheta_max),
                weights=mean_df['Mean I @ psi']
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

def batch_XRD(batch_upload):
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
        xrd_df = Generate_XRD(selected_hkls, intensities, Gaussian_FWHM, strain_sim_params, Funamori_broadening)
        # Rename columns so each block is unique
        xrd_df = xrd_df.rename(columns={
            "2th": f"2th_iter{idx+1}",
            "Total Intensity": f"Intensity_iter{idx+1}"
        }).reset_index(drop=True)

        results_blocks.append(xrd_df)

    # Align all result blocks by index and combine
    results_df = pd.concat(results_blocks, axis=1)

    return parameters_df, results_df, results_blocks

def cake_data(selected_hkls, intensities, symmetry, lattice_params, wavelength, cijs, sigma_params, chi):
    """
    Computes the azimuth vs 2th strain data for each hkl and combines into a dictionary with entries for each hkl

    Returns:
    cake_dict
    keys (hkl_labels) : values (df of information for this hkl)
    """
    cake_dict = {}
    
    for hkl, intensity in zip(selected_hkls, intensities):
        phi_values = np.radians(np.arange(0, 360, 5))
        psi_values = 0  # let compute_strain calculate psi for each HKL
        hkl_label, df, psi_list, strain_33_list = compute_strain(
            hkl, intensity, symmetry, lattice_params, wavelength, cijs,
            sigma_params, chi, phi_values, psi_values
        )
        cake_dict[hkl_label] = df
    
    return cake_dict

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
            #Get the mean values for each psi
            all_delta.extend(unique["delta (degrees)"].values)
            all_2th.extend(unique["Mean two_th"].values)
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

def setup_refinement_toggles(lattice_params, **additional_fields):
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
    
def run_refinement(params, refine_flags, selected_hkls, selected_indices, intensities, Gaussian_FWHM, phi_values, psi_values, wavelength, symmetry, x_exp, y_exp, lattice_params, cijs,
                   sigma_params, chi, Funamori_broadening):
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
            min_val, max_val = 0, 1500
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
    XRD_df = Generate_XRD(selected_hkls, intensities_opt, Gaussian_FWHM, strain_sim_params, Funamori_broadening)
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
        df = compute_strain(hkl, inten, *strain_sim_params)[1]
        #Compute the average of the mean_2th values (For axial, this averages over many identical values, for radial, we average across a range of psi)
        mean_2th = np.mean(df["Mean two_th"])
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
            x_exp_common, y_exp_common, bin_indices, Funamori_broadening, global_lattice_params=lattice_params, global_cijs=cijs, global_sigmas=sigma_params
        )

    # Run optimization
    result = minimize(wrapped_cost_function, lm_params, method="leastsq", gtol=1e-8,)
    #-------------------------------------------------

    return result

def cost_function(lm_params, refine_flags, selected_hkls, selected_indices,
                  Gaussian_FWHM, phi_values, psi_values, wavelength, symmetry,
                  x_exp_common, y_exp_common, bin_indices,
                  Funamori_broadening, global_lattice_params, global_cijs, global_sigmas):
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
    XRD_df = Generate_XRD(selected_hkls, intensities_opt, Gaussian_FWHM, strain_sim_params, Funamori_broadening)
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

### Figure Generation --------------------------------------------------

def generate_epsilon_psi_curves(selected_hkls, psi_steps, phi_steps):

    results_dict = {}
    phi_values = np.linspace(0, 2 * np.pi, phi_steps)
    psi_values = np.linspace(0, np.pi / 2, psi_steps)

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
                                                                 chi, phi_values, psi_values
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
        mean_df = (df.groupby("psi (degrees)", sort=True)["Mean strain"].first().reset_index())

        fig.add_trace(go.Scatter(x=mean_df["psi (degrees)"],
                                 y=mean_df["Mean strain"],
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

#Old matplotlib iplementation
#def generate_epsilon_psi_curves(selected_hkls, psi_steps, phi_steps):
#    fig, axs = plt.subplots(len(selected_hkls), 1, figsize=(8, 5 * len(selected_hkls)))
#    if len(selected_hkls) == 1:
#        axs = [axs]

#    results_dict = {} #Generate empty dictionary to hold results

#    phi_values = np.linspace(0, 2 * np.pi, phi_steps)
#    psi_values = np.linspace(0, np.pi/2, psi_steps)

#    for ax, hkl, intensity in zip(axs, selected_hkls, intensities):
#        hkl_label, df, psi_list, strain_33_list = compute_strain(hkl, intensity, symmetry, lattice_params, wavelength, cijs, sigma_11, sigma_22, sigma_33, chi, phi_values, psi_values)
#        results_dict[hkl_label] = df

#        scatter = ax.scatter(psi_list, strain_33_list, color="black", s=0.2, alpha=0.1)
#        ax.hlines(0,0,90, color="black", lw=0.8)
#        ax.vlines(54.7,np.min(strain_33_list), np.max(strain_33_list),color="black", ls="dashed", lw=0.8)
        
#        #Plot the mean strain curve
#        unique_psi = np.unique(psi_list)
#        mean_strain_list = []
#       for psi in np.unique(psi_list):
#           #Obtain all the strains at this particular psi
#           mask = df["psi (degrees)"] == psi
#           strains = strain_33_list[mask]
#            mean_strain = df["Mean strain"][mask].iloc[0]
#            #Append to list
#            mean_strain_list.append(mean_strain)
#        ax.plot(unique_psi, mean_strain_list, color="red", lw=0.8, label="mean strain")
#        ax.set_xlabel("ψ (degrees)")
#        ax.set_ylabel("ε′₃₃")
#        ax.set_xlim(0,90)
#        ax.set_title(f"ε′₃₃ [hkl = ({hkl_label})]")
#        ax.legend()
#        plt.tight_layout()
#    st.pyplot(fig)
#    return results_dict

def generate_cake_figures(results_dict, selected_hkls, broadening):

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
                mean_2ths = np.full(len(np.unique(df["delta (degrees)"].values)),df["Mean two_th"].iloc[0])
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
                mean_2th = unique["Mean two_th"].values
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
        mean_strain_list = [df[df["delta (degrees)"]==d]["Mean strain"].iloc[0] for d in unique_delta]
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

# Old matplotlib implementation
    #fig, ax = plt.subplots(figsize=(8, 4))
    #ax.plot(twotheta_grid, total_pattern, label="Simulated XRD", lw=0.5, color="black")
    #ax.set_xlabel("2θ (deg)")
    #ax.set_ylabel("Intensity (a.u.)")
    #ax.set_title("Simulated XRD Pattern")
    #ax.legend()
    #st.pyplot(fig)

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
    
    #Old matplotlib implementation
    #fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(8, 6), sharex=True, gridspec_kw={'height_ratios': [3, 1]})
    #ax1.plot(x_exp, y_exp, label="Experimental", color='black', lw=0.5)
    #ax1.plot(x_exp, y_sim, label="Simulated", linestyle='--', color='red', lw=0.5)
    #ax1.legend()
    #ax1.set_ylabel("Intensity")
    #ax1.set_title(title)

    #ax2.plot(x_exp, residuals, color='blue', lw=0.5)
    #ax2.axhline(0, color='gray', lw=0.5)
    #ax2.set_xlabel("2θ (degrees)")
    #ax2.set_ylabel("Residuals")

    #st.pyplot(fig)

#Helper function for storing downloadable data
def store_download(key, datasource, buffer, filename, mime):
        st.session_state.download_data[key] = {
            "datasource": datasource,
            "buffer": buffer,
            "filename": filename,
            "mime": mime,
        }
# -----------------------------------------------------------------------
#### Main App -----------------------------------------------------
# -----------------------------------------------------------------------
    
st.set_page_config(layout="wide")

BASE_DIR = Path(__file__).parent
logo_path = BASE_DIR / "spinel_logo.png"

img = Image.open(logo_path)

col_img, col_title = st.columns([1, 3])

with col_img:
    st.image(img, width='stretch')

col1, col2, col3, col4, col5, col6 = st.columns(6)

with col1:
    st.subheader("Upload Files")
    uploaded_file = st.file_uploader("Elastic and hkl csv", type=["csv"])

if uploaded_file is not None:
    with col2:
        st.subheader("")
        poni_file = st.file_uploader("Poni", type=["poni"])
    with col3:
        st.subheader("")
        batch_upload = st.file_uploader("Batch XRD file", type=["csv"])
    with col4:
        st.subheader("")
        twoD_XRD = st.file_uploader("2D XRD tiff", type=["tiff"])

    #Define download_data if not initialised
    if "download_data" not in st.session_state:
        st.session_state.download_data = {}
    #Initialise the "previous" download format to track changes. Defaults to "Excel (.xlsx)"
    if "prev_download_format" not in st.session_state:
        st.session_state.prev_download_format = st.session_state.get("download_format", "Excel (.xlsx)")

    #Section for downloading computed data
    columns = st.columns(6)
    with columns[0]:
        st.subheader("Download Data")
        with st.form("download_form"):
                st.selectbox(
                    "Set download format",
                    ["Excel (.xlsx)", "OpenDocument (.ods)", "ZIP of CSVs (.zip)"],
                    index=0,
                    key = "download_format"
                )
                submitted = st.form_submit_button("Set format")
        st.write(st.session_state.download_format)
        
        #Reformat the data only if selection changed
        if submitted:
            if st.session_state.download_format != st.session_state.prev_download_format:
                if st.session_state.download_data:
                    st.write("Format changed → reprocessing data")
                #Reformat the available data accordingly
                for key,data in st.session_state.download_data.items():
                    if key in ["epsilon_psi", "cake"]:
                        datasource = data["datasource"]
                        if st.session_state.download_format == "Excel (.xlsx)":
                            output_buffer = io.BytesIO()
                
                            with pd.ExcelWriter(output_buffer, engine='xlsxwriter') as writer:
                                for hkl_label, df in datasource.items():
                                    sheet_name = f"hkl_{hkl_label}"
                                    df.to_excel(writer, sheet_name=sheet_name, index=False)
                
                                    worksheet = writer.sheets[sheet_name]
                                    for i, col in enumerate(df.columns):
                                        max_width = max(
                                            df[col].astype(str).map(len).max(),
                                            len(col)
                                        ) + 2
                                        worksheet.set_column(i, i, max_width)
                
                            output_buffer.seek(0)
                            buffer = output_buffer
                            file = data["filename"].split(".")
                            filename = "{}.xlsx".format(file[0])
                            mime =("application/vnd.openxmlformats-officedocument.spreadsheetml.sheet")
                            store_download(key, datasource, buffer, filename, mime)
                
                        elif st.session_state.download_format == "OpenDocument (.ods)":
                            output_buffer = io.BytesIO()
                
                            with pd.ExcelWriter(output_buffer, engine='odf') as writer:
                                for hkl_label, df in datasource.items():
                                    df.to_excel(writer, sheet_name=f"hkl_{hkl_label}", index=False)
                
                            output_buffer.seek(0)
                            buffer = output_buffer
                            file = data["filename"].split(".")
                            filename = "{}.ods".format(file[0])
                            mime =("application/vnd.oasis.opendocument.spreadsheet")
                            store_download(key, datasource, buffer, filename, mime)

                        elif st.session_state.download_format == "ZIP of CSVs (.zip)":
                            output_buffer = io.BytesIO()
        
                            with zipfile.ZipFile(output_buffer, "w") as zf:
                                for hkl_label, df in datasource.items():
                                    csv_buffer = io.StringIO()
                                    df.to_csv(csv_buffer, index=False)
                                    zf.writestr(f"{hkl_label}.csv", csv_buffer.getvalue())
                            
                            output_buffer.seek(0)
                            buffer = output_buffer
                            file = data["filename"].split(".")
                            filename = "{}.zip".format(file[0])
                            mime =("application/zip")
                            store_download(key, datasource, buffer, filename, mime)
                        else:
                            pass
                # update stored value
                st.session_state.prev_download_format = st.session_state.download_format
            else:
                pass
            
    if st.session_state.download_data:
        columns = st.columns(12)
        download_data = st.session_state.download_data
        items = list(download_data.items())
        for i, (key, data) in enumerate(items):
            with columns[i]:
                # Persistent download buttons
                if st.download_button(
                    label=f"📥 Download {data["filename"]}",
                    data=data["buffer"],
                    file_name=data["filename"],
                    mime=data["mime"],
                    key=f"download_{i}"  # unique key required
                ):
                    # Auto-clear
                    st.session_state.download_data.pop(key, None)
            
    col1, col2, col3, col4, col5, col6, col7, col8 = st.columns([2,3,1,2,1,1,1,1])
    with col1:
        st.subheader("Reflections/Intensities")
    with col2:
        st.subheader("Material")
    with col3:
        st.subheader("Elastic")
    with col4:
        st.subheader("Stress")
    with col5:
        st.subheader("Computation")
    with col6:
        st.subheader("Preferred Orientation")

col1, col2, col3, col4, col5, col6, col7, col8, col9, col10, col11 = st.columns([2,1,1,1,1,1,1,1,1,1,1])

if uploaded_file is not None:
    st.session_state["uploaded_file"] = uploaded_file
    file_obj = st.session_state.get("uploaded_file", None)
    # --- Read and split file ---
    content = file_obj.getvalue().decode("utf-8")
    lines = content.strip().splitlines()
    # --- Separate metadata and data lines ---
    metadata = {}
    data_lines = []
    for line in lines:
        line = line.strip()
        if not line:
            continue
        if line.startswith('#'):
            # Extract metadata lines of form "# key: value"
            if ':' in line:
                key, val = line[1:].split(':', 1)
                try:
                    metadata[key.strip()] = float(val)
                except:
                    metadata[key.strip()] = val.strip()
        else:
            data_lines.append(line)

    symmetry = metadata["symmetry"]
    #Check the correct data has been included for the respective symmetry
    if symmetry == "cubic":
        required_keys = {'a','b','c','alpha','beta','gamma','wavelength','C11','C12','C44','sig11','sig22','sig33','chi',}
    elif symmetry == "hexagonal":
        required_keys = {'a','b','c','alpha','beta','gamma','wavelength','C11','C33','C12','C13','C44','sig11','sig22','sig33','chi'}
    elif symmetry == "tetragonal_A":
        required_keys = {'a','b','c','alpha','beta','gamma','wavelength','C11','C33','C12','C13','C44','C66','sig11','sig22','sig33','chi'}
    elif symmetry == "tetragonal_B":
        required_keys = {'a','b','c','alpha','beta','gamma','wavelength','C11','C33','C12','C13','C16','C44','C66','sig11','sig22','sig33','chi'}
    elif symmetry == "orthorhombic":
        required_keys = {'a','b','c','alpha','beta','gamma','wavelength','C11','C22','C33','C12','C13','C23','C44','C55','C66','sig11','sig22','sig33','chi'}
    elif symmetry == "trigonal_A":
        required_keys = {'a','b','c','alpha','beta','gamma','wavelength','C11','C33','C12','C13', 'C14','C44','sig11','sig22','sig33','chi'}
    else:
        st.error("{} symmetry is not yet supported".format(symmetry))
        required_keys = {}
    
    #Optional for off-diagonal stress terms - otherwise default to zero
    optional_keys = {'sig12', 'sig13', 'sig23'}

    metadata_keys = set(metadata) - {'symmetry'} #Drops the symmetry key from the check as it is assumed present

    missing_keys = required_keys - metadata_keys
    allowed_keys = required_keys | optional_keys
    extra_keys = metadata_keys - allowed_keys
    
    if missing_keys:
        st.error(f"Missing required keys: {', '.join(missing_keys)}")
        st.write(f"CSV must contain at least: {', '.join(required_keys)}")
        st.stop()
    
    if extra_keys:
        st.warning(f"Unexpected keys found: {', '.join(extra_keys)}")

    #Set sig12, 13, 23 if not provided in the input file
    for key in ['sig12', 'sig13', 'sig23']:
        if key in metadata.keys():
            pass
        else:
            metadata[key] = 0.0 #Set default value to zero
    
    # --- Parse HKL + intensity section ---
    try:
        hkl_df = pd.read_csv(io.StringIO("\n".join(data_lines)))
    except Exception as e:
        st.error(f"Error reading HKL section: {e}")
        st.stop()
    # Validate required columns
    required_cols = {'h', 'k', 'l', 'intensity'}
    if not required_cols.issubset(hkl_df.columns):
        st.error(f"HKL section must have columns: {', '.join(required_cols)}")
        st.stop()
    else:
        # Ensure numeric conversion
        hkl_df[['h', 'k', 'l']] = hkl_df[['h', 'k', 'l']].apply(pd.to_numeric, errors='coerce').fillna(0).astype(int)
        hkl_df['intensity'] = pd.to_numeric(hkl_df['intensity'], errors='coerce').fillna(1.0)
        hkl_list = hkl_df[['h', 'k', 'l']].drop_duplicates().values.tolist()
        
        #Initialise lists/dictionaries
        selected_hkls = []
        intensities = []
        selected_indices = []
        peak_intensity_default = {}

        if "params" not in st.session_state:
            st.session_state.params = {
                "a_val": float(metadata['a']),
                "b_val": float(metadata['b']),
                "c_val": float(metadata['c']),
                "alpha": float(metadata['alpha']),
                "beta": float(metadata['beta']),
                "gamma": float(metadata['gamma']),
                "chi": float(metadata['chi']),
                "wavelength": float(metadata['wavelength']),
                **{k.lower(): metadata[k] for k in metadata.keys() if k.startswith("C")},
                "sigma_11": float(metadata["sig11"]),
                "sigma_22": float(metadata["sig22"]),
                "sigma_33": float(metadata["sig33"]),
                "sigma_12": float(metadata["sig12"]),
                "sigma_13": float(metadata["sig13"]),
                "sigma_23": float(metadata["sig23"])
            }
        with col1:
            for i, hkl in enumerate(hkl_list):
                    # Find matching row to get intensity
                    h_match = (hkl_df['h'] == hkl[0]) & (hkl_df['k'] == hkl[1]) & (hkl_df['l'] == hkl[2])
                    default_intensity = float(hkl_df[h_match]['intensity'].values[0]) if h_match.any() else 1.0
                    peak_intensity_default[f"intensity_{i}"] = default_intensity
                
            # Initialize state for peak intensity
            if "intensities" not in st.session_state:
                st.session_state.intensities = peak_intensity_default.copy()

            for i, hkl in enumerate(hkl_list):
                cols = st.columns(2)    
                with cols[0]:
                    label = f"hkl = ({int(hkl[0])}, {int(hkl[1])}, {int(hkl[2])})"
                    selected = st.checkbox(label, value=True, key=f"chk_{i}")
                with cols[1]:
                    st.session_state.intensities[f"intensity_{i}"] = st.number_input(
                        f"Intensity_{i}",
                        min_value=0.0,
                        value=st.session_state.intensities[f"intensity_{i}"],
                        step=1.0,
                        label_visibility="collapsed"
                    )

                if selected:
                    selected_hkls.append(hkl)
                    selected_indices.append(i)  # Save which index was selected
                    intensities.append(st.session_state.intensities[f"intensity_{i}"])

        with col2:
            symmetry_options = ["cubic", "hexagonal", "tetragonal_A", "tetragonal_B", "orthorhombic", "trigonal_A"]
            if metadata['symmetry'] in symmetry_options:
                default_index = symmetry_options.index(metadata['symmetry'])
            else:
                default_index = 0  # fallback
                st.write("{} symmetry not supported. Choose from the options below". format(metadata['symmetry']))
            symmetry = st.selectbox("Symmetry:",symmetry_options, index=default_index)
            st.session_state.params["wavelength"] = st.number_input("Wavelength (Å)", value=st.session_state.params["wavelength"], step=0.01, format="%.4f")
            st.session_state.params["chi"] = st.number_input("Chi angle (deg)", value=st.session_state.params["chi"], step=0.01, format="%.3f")            
        with col3:
            st.session_state.params["a_val"] = st.number_input("Lattice a (Å)", value=st.session_state.params["a_val"], step=0.01, format="%.4f")
            st.session_state.params["b_val"] = st.number_input("Lattice b (Å)", value=st.session_state.params["b_val"], step=0.01, format="%.4f")
            st.session_state.params["c_val"] = st.number_input("Lattice c (Å)", value=st.session_state.params["c_val"], step=0.01, format="%.4f")
        with col4:
            st.session_state.params["alpha"] = st.number_input("alpha (deg)", value=st.session_state.params["alpha"], step=0.1, format="%.3f")
            st.session_state.params["beta"] = st.number_input("beta (deg)", value=st.session_state.params["beta"], step=0.1, format="%.3f")
            st.session_state.params["gamma"] = st.number_input("gamma (deg)", value=st.session_state.params["gamma"], step=0.1, format="%.3f")
        with col5:
            # Dynamically build the list of Cij keys present in params
            c_keys = [key for key in st.session_state.params.keys() if key.startswith('c') and key not in ["c_val", "chi"]]
            cijs = {}
            for key in c_keys:
                st.session_state.params[key] = st.number_input(key, value=st.session_state.params[key])
                cijs[key] = st.session_state.params.get(key)
        with col6:
            st.session_state.params["sigma_11"] = st.number_input("σ₁₁", value=st.session_state.params["sigma_11"], step=0.1, format="%.3f")
            st.session_state.params["sigma_22"] = st.number_input("σ₂₂", value=st.session_state.params["sigma_22"], step=0.1, format="%.3f")
            st.session_state.params["sigma_33"] = st.number_input("σ₃₃", value=st.session_state.params["sigma_33"], step=0.1, format="%.3f")
            st.markdown("t: {}".format(round(st.session_state.params["sigma_33"] - st.session_state.params["sigma_11"],3)))
        with col7:
            st.session_state.params["sigma_12"] = st.number_input("σ₁₂", value=st.session_state.params["sigma_12"], step=0.1, format="%.3f")
            st.session_state.params["sigma_13"] = st.number_input("σ₁₃", value=st.session_state.params["sigma_13"], step=0.1, format="%.3f")
            st.session_state.params["sigma_23"] = st.number_input("σ₂₃", value=st.session_state.params["sigma_23"], step=0.1, format="%.3f")
        with col8:
            Funamori_broadening = st.checkbox("Include broadening", value=True)
            total_points = st.number_input("Total points (φ × ψ)", value=5000, min_value=10, step=5000)
            Gaussian_FWHM = st.number_input("Gaussian FWHM", value=0.1, min_value=0.005, step=0.005, format="%.3f")
        with col9:
            st.session_state.params["PO_toggle"] = st.checkbox("Preferred Orientation", value=False)
            if st.session_state.params.get("PO_toggle"):
                po_model = st.selectbox("PO Model:",["March-Dollase"])
                #po_model = st.text_input("PO Model", value="March-Dollase")
                if po_model == "March-Dollase":
                    POD_hkl_input = st.text_input("POD hkl", value="001")
                    #Convert hkl_POD to tuple
                    if len(POD_hkl_input) != 3 or not POD_hkl_input.isdigit():
                        st.write("hkl of POD must be three digets.")
                        st.session_state.params["hkl_POD"] = (0,0,1)
                    else:
                        st.session_state.params["hkl_POD"] = tuple(map(int, POD_hkl_input))
                    st.session_state.params["baseline"] = st.number_input("Baseline (between 0 and 1)", value=0.0, step=0.1, format="%.2f")
                    st.session_state.params["R"] = st.number_input("R", value=0.5, step=0.1, format="%.2f")
                    st.session_state.params["tau"] = st.number_input("tau (deg)", value=0.0, step=5.0, format="%.1f")
                    st.session_state.params["rho"] = st.number_input("rho (deg)", value=0.0, step=5.0, format="%.1f")
                    st.session_state.params["weight"] = st.number_input("weight", value=1.0, step=0.1, format="%.1f")
                else:
                    st.write("{} model is not supported".format(po_model))
                    st.write("Choose from below:")
                    st.write("March-Dollase")
        lattice_params = {
            "a_val" : st.session_state.params.get("a_val"),
            "b_val" : st.session_state.params.get("b_val"),
            "c_val" : st.session_state.params.get("c_val"),
            "alpha" : st.session_state.params.get("alpha"),
            "beta" : st.session_state.params.get("beta"),
            "gamma" : st.session_state.params.get("gamma"),
        }
        wavelength = st.session_state.params.get("wavelength")
        chi = st.session_state.params.get("chi")
        # Dynamically build the list of sigma_ij keys present in params
        sigma_keys = ['sigma_11','sigma_22','sigma_33','sigma_12','sigma_13','sigma_23']
        sigma_params = {}
        for key in sigma_keys:
            sigma_params[key] = st.session_state.params.get(key)
        
        # Determine grid sizes
        psi_steps = int(2 * np.sqrt(total_points))
        phi_steps = int(np.sqrt(total_points) / 2)
        results_dict = {}  # Store results per HKL reflection
            
        col1, col2 = st.columns(2)
        with col1:
            st.subheader("Execute Calculations")
            #---------------------         
            #Generating epsilon-psi curves
            #--------------------- 
            epsilon_psi_dict = None
            if st.button("ε-ψ Curves") and selected_hkls:
                epsilon_psi_dict = generate_epsilon_psi_curves(selected_hkls, psi_steps, phi_steps)

            #Format the data and save to session_state
            if epsilon_psi_dict is not None:
                if st.session_state.download_format == "Excel (.xlsx)":
                    output_buffer = io.BytesIO()
        
                    with pd.ExcelWriter(output_buffer, engine='xlsxwriter') as writer:
                        for hkl_label, df in epsilon_psi_dict.items():
                            sheet_name = f"hkl_{hkl_label}"
                            df.to_excel(writer, sheet_name=sheet_name, index=False)
        
                            worksheet = writer.sheets[sheet_name]
                            for i, col in enumerate(df.columns):
                                max_width = max(
                                    df[col].astype(str).map(len).max(),
                                    len(col)
                                ) + 2
                                worksheet.set_column(i, i, max_width)
        
                    output_buffer.seek(0)
                    datasource = epsilon_psi_dict
                    key = "epsilon_psi"
                    buffer = output_buffer
                    filename = "epsilon_psi.xlsx"
                    mime =("application/vnd.openxmlformats-officedocument.spreadsheetml.sheet")
                    store_download(key, datasource, buffer, filename, mime)
        
                elif st.session_state.download_format == "OpenDocument (.ods)":
                    output_buffer = io.BytesIO()
        
                    with pd.ExcelWriter(output_buffer, engine='odf') as writer:
                        for hkl_label, df in epsilon_psi_dict.items():
                            df.to_excel(writer, sheet_name=f"hkl_{hkl_label}", index=False)
        
                    output_buffer.seek(0)
                    datasource = epsilon_psi_dict
                    key = "epsilon_psi"
                    buffer = output_buffer
                    filename = "epsilon_psi.ods"
                    mime =("application/vnd.oasis.opendocument.spreadsheet")
                    store_download(key, datasource, buffer, filename, mime)

                elif st.session_state.download_format == "ZIP of CSVs (.zip)":
                    output_buffer = io.BytesIO()

                    with zipfile.ZipFile(output_buffer, "w") as zf:
                        for hkl_label, df in epsilon_psi_dict.items():
                            csv_buffer = io.StringIO()
                            df.to_csv(csv_buffer, index=False)
                            zf.writestr(f"{hkl_label}.csv", csv_buffer.getvalue())
                    
                    output_buffer.seek(0)
                    datasource = epsilon_psi_dict
                    key = "epsilon_psi"
                    buffer = output_buffer
                    filename = "epsilon_psi.zip"
                    mime =("application/zip")
                    store_download(key, datasource, buffer, filename, mime)
                else:
                    pass
    
                st.success("File available for download above")
            
            #---------------------         
            #Generating cake plots
            #---------------------  
            if st.button("Cake Plot") and selected_hkls:
                cake_dict = cake_data(selected_hkls, intensities, symmetry, lattice_params, 
                                                    wavelength, cijs, sigma_params, chi)
                generate_cake_figures(cake_dict, selected_hkls, Funamori_broadening)
                
                if cake_dict != {}:
                    #Format the data and save to session_state
                    if st.session_state.download_format == "Excel (.xlsx)":
                        output_buffer = io.BytesIO()
            
                        with pd.ExcelWriter(output_buffer, engine='xlsxwriter') as writer:
                            for hkl_label, df in cake_dict.items():
                                sheet_name = f"hkl_{hkl_label}"
                                df.to_excel(writer, sheet_name=sheet_name, index=False)
            
                                worksheet = writer.sheets[sheet_name]
                                for i, col in enumerate(df.columns):
                                    max_width = max(
                                        df[col].astype(str).map(len).max(),
                                        len(col)
                                    ) + 2
                                    worksheet.set_column(i, i, max_width)
            
                        output_buffer.seek(0)
                        datasource = cake_dict
                        key = "cake"
                        buffer = output_buffer
                        filename = "cake.xlsx"
                        mime =("application/vnd.openxmlformats-officedocument.spreadsheetml.sheet")
                        store_download(key, datasource, buffer, filename, mime)
            
                    elif st.session_state.download_format == "OpenDocument (.ods)":
                        output_buffer = io.BytesIO()
            
                        with pd.ExcelWriter(output_buffer, engine='odf') as writer:
                            for hkl_label, df in cake_dict.items():
                                df.to_excel(writer, sheet_name=f"hkl_{hkl_label}", index=False)
            
                        output_buffer.seek(0)
                        datasource = cake_dict
                        key = "cake"
                        buffer = output_buffer
                        filename = "cake.ods"
                        mime =("application/vnd.oasis.opendocument.spreadsheet")
                        store_download(key, datasource, buffer, filename, mime)
    
                    elif st.session_state.download_format == "ZIP of CSVs (.zip)":
                        output_buffer = io.BytesIO()
    
                        with zipfile.ZipFile(output_buffer, "w") as zf:
                            for hkl_label, df in cake_dict.items():
                                csv_buffer = io.StringIO()
                                df.to_csv(csv_buffer, index=False)
                                zf.writestr(f"{hkl_label}.csv", csv_buffer.getvalue())
                        
                        output_buffer.seek(0)
                        datasource = cake_dict
                        key = "cake"
                        buffer = output_buffer
                        filename = "cake.zip"
                        mime =("application/zip")
                        store_download(key, datasource, buffer, filename, mime)
                    else:
                        pass
        
                    st.success("File available for download above")

            #Plotting preferred orientation
            if st.session_state.params.get("PO_toggle"):
                st.subheader("Preferred Orientation")
                if st.button("Plot PO Model"):
                    components = [
                        {"tau": st.session_state.params.get("tau"), "rho": st.session_state.params.get("rho"),"R": st.session_state.params.get("R") , "weight" : st.session_state.params.get("weight")
                        }
                    ]
                    hkl_POD = st.session_state.params.get("hkl_POD")
                    PO_MODEL = PO.PO_Model(po_model=po_model,
                                           components=components,
                                           baseline=st.session_state.params.get("baseline"),
                                           chi_deg = chi,
                                           POD_xtal=hkl_POD
                                          )
                    fig = PO_MODEL.make_intensity_pole_figure()
                    st.pyplot(fig)

                if st.button("TEST PO Model"):
                    components = [
                    {"tau": st.session_state.params.get("tau"), "rho": st.session_state.params.get("rho"),"R": st.session_state.params.get("R") , "weight" : st.session_state.params.get("weight")
                    }
                    ]
                    hkl_POD = st.session_state.params.get("hkl_POD")
                    PO_MODEL = PO.PO_Model(po_model=po_model,
                                           components=components,
                                           baseline=st.session_state.params.get("baseline"),
                                           chi_deg = chi,
                                           POD_xtal=hkl_POD
                                          )
                    phi = np.linspace(0,360,32)
                    delta = np.linspace(-180,180,32)
                    I_grid, delta_grid, phi_grid = PO_MODEL.intensity_for_hkl((1,0,0), phi, delta)
                    #Plot the intensity distribution
                    fig = plt.figure(figsize=(6, 4))
                    ax_3d = fig.add_subplot(111, projection='3d')
                    ax_3d.view_init(elev=30, azim=-30)
                    surf = ax_3d.plot_surface(
                    delta_grid, phi_grid, I_grid,
                    cmap='viridis', edgecolor='k', alpha=0.9
                    )

                    ax_3d.set_xlabel("delta")
                    ax_3d.set_ylabel("phi")
                    ax_3d.set_zlabel("intensity")
                    st.pyplot(fig)
            
            st.subheader("Generate XRD patterns")
            if st.button("1D-XRD") and selected_hkls:
                phi_values = np.radians(np.arange(0, 360, 2))
                psi_values = 0
                strain_sim_params = (symmetry, lattice_params, wavelength, cijs, sigma_params, chi, phi_values, psi_values)

                XRD_df = Generate_XRD(selected_hkls, intensities, Gaussian_FWHM, strain_sim_params, broadening=Funamori_broadening)

                generate_1D_XRD_plot(XRD_df)

                twotheta_grid = XRD_df["2th"]
                total_pattern = XRD_df["Total Intensity"]

                #Prepare .xy file
                # .xy format is two columns, 2th and intensity
                output_buffer = io.StringIO()
                for tth, intensity in zip(twotheta_grid, total_pattern):
                    output_buffer.write(f"{tth:.5f} {intensity:.5f}\n")
                
                # Move cursor to start for reading
                output_buffer.seek(0)
                datasource = XRD_df
                key = "1D XRD"
                buffer = output_buffer.getvalue()
                filename = "1D XRD.xy"
                mime =("text/plain")
                store_download(key, datasource, buffer, filename, mime)
                st.success("File available for download above")

            if poni_file is not None:
                if st.button("2D-XRD") and selected_hkls:
                    # Save to a temporary file
                    with tempfile.NamedTemporaryFile(suffix=".poni") as tmp:
                        tmp.write(poni_file.read())
                        tmp.flush()
                        
                        # Load the geometry
                        ai = AzimuthalIntegrator()
                        ai.load(tmp.name)
                    
                    #Compute the cake data
                    cake_dict = {}
                    cake_dict = cake_data(selected_hkls, intensities, symmetry, lattice_params, 
                                            wavelength, cijs, sigma_params, chi)
                    cake_two_thetas, cake_deltas, cake_intensity = cake_dict_to_2Dcake(cake_dict, broadening=Funamori_broadening)

                    fig, ax = plt.subplots()
                    
                    im = ax.imshow(
                        cake_intensity.T,
                        extent=[cake_two_thetas.min(), cake_two_thetas.max(),
                                cake_deltas.min(), cake_deltas.max()],
                        aspect='auto', 
                        origin='lower',
                        vmin=0,
                        vmax=np.percentile(cake_intensity, 98),
                        cmap='binary_r'
                    )

                    ax.set_xlabel("2θ (degrees)")
                    ax.set_ylabel("δ (degrees)")
                    ax.set_title("Cake")
                    plt.colorbar(im, ax=ax, label="Intensity")
                    st.pyplot(fig)

                    # Generate the raw detector image
                    # convert two_th to radians (requirement of pyFAI)
                    delta_axis_rad = np.deg2rad(cake_deltas)
                    tth_axis_rad = np.deg2rad(cake_two_thetas)

                    poni_file.seek(0)
                    text = poni_file.read().decode("utf-8")
                    # Parse line by line
                    for line in text.splitlines():
                        if "Detector_config" in line:
                            # Find the part after "max_shape"
                            idx = line.find("max_shape")
                            if idx != -1:
                                # Example: max_shape: [2048, 2048]
                                start = line.find("[", idx)
                                end = line.find("]", idx)
                                if start != -1 and end != -1:
                                    shape_str = line[start+1:end]  # '2048, 2048'
                                    height, width = map(int, shape_str.split(","))
                                    st.write("Detector size:", height, width)

                    det_shape = (height, width)  # (height, width)
                    
                    # Step 1: Compute angular field for each detector pixel
                    tth_pix = ai.twoThetaArray((height, width))  # radians
                    chi_pix = ai.chiArray((height, width))       # radians
                    
                    # Step 2: Build interpolator from cake space
                    interp = RegularGridInterpolator(
                        (tth_axis_rad, delta_axis_rad),
                        cake_intensity,
                        bounds_error=False,
                        fill_value=0
                    )

                    # Step 3: Sample cake intensities at detector angular coordinates
                    coords = np.stack([tth_pix, chi_pix], axis=-1)
                    det_image = interp(coords)

                    fig, ax = plt.subplots(figsize=(8, 6))
                    im = ax.imshow(det_image, 
                                   origin='lower', 
                                   cmap='binary_r', 
                                   aspect='equal', 
                                   vmin=0, 
                                   vmax=np.percentile(det_image, 98))
                    fig.colorbar(im, ax=ax, label='Intensity')
                    ax.set_xlabel('Pixel X')
                    ax.set_ylabel('Pixel Y')
                    st.pyplot(fig)
                
            #Make batch processing section
            if batch_upload:
                parameters_df, results_df, results_blocks = batch_XRD(batch_upload)

                #Plot up the data
                fig, ax = plt.subplots(figsize=(10, 6))

                #Get the first y dataset to compute the offset
                y_initial = results_df["Intensity_iter1"]
                y_offset = 0
                offset_step = np.max(y_initial)*0.5
                
                for idx in range(len(results_blocks)):
                    x_col = f"2th_iter{idx+1}"
                    y_col = f"Intensity_iter{idx+1}"
                    x = results_df[x_col]
                    y = results_df[y_col]
                    ax.plot(x, y + y_offset, color="black", lw=1, label=f"Iteration {idx+1}")
                    #Increase the offset
                    y_offset = y_offset+offset_step
                    
                ax.set_xlabel("2θ (degrees)")
                ax.set_ylabel("Intensity (a.u.)")
                ax.set_title("Batch XRD")
                plt.tight_layout()
                #Display the plot
                st.pyplot(fig)
                
                # Now you have two parts: parameters_df and results_df
                # Export format: parameters first, then results
                st.subheader("Download Computed Data")
                output_buffer = io.BytesIO()
                with pd.ExcelWriter(output_buffer, engine='xlsxwriter') as writer:
                    parameters_df.to_excel(writer, sheet_name="Parameters", index=False)
                    results_df.to_excel(writer, sheet_name="Results", index=False)

                    # Auto-width adjustment for Parameters sheet
                    worksheet_params = writer.sheets["Parameters"]
                    for i, col in enumerate(parameters_df.columns):
                        max_width = max(parameters_df[col].astype(str).map(len).max(), len(str(col))) + 2
                        worksheet_params.set_column(i, i, max_width)

                    # Auto-width adjustment for "Results" sheet
                    worksheet = writer.sheets["Results"]
                    for i, col in enumerate(results_df.columns):
                        max_width = max(results_df[col].astype(str).map(len).max(), len(str(col))) + 2
                        worksheet.set_column(i, i, max_width)

                output_buffer.seek(0)
            
                st.download_button(
                    label="📥 Download Batch XRD as Excel (.xlsx)",
                    data=output_buffer,
                    file_name="XRD_results.xlsx",
                    mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
                )
        
                #st.write("Parameters", parameters_df)
                #st.write("Results", results_df)

    ### XRD Comparison/Refinement ----------------------------------------------------------------
    with col2:
        st.subheader("Overlay/refine with XRD")
        uploaded_XRD = st.file_uploader("Upload .xy experimental XRD file", type=[".xy"])

    if uploaded_XRD is not None:
        raw_lines = uploaded_XRD.read().decode("utf-8").splitlines()
        data_lines = [line for line in raw_lines if not line.strip().startswith("#") and line.strip()]
        data = pd.read_csv(io.StringIO("\n".join(data_lines)), sep=r'\s+', header=None, names=['2th', 'intensity'])
        x_exp = data['2th'].values
        y_exp = data['intensity'].values
        #Normalise exp data
        y_exp = y_exp/ np.max(y_exp)*100

        with col2:
            if st.button("Overlay XRD"):
                phi_values = np.radians(np.arange(0, 360, 2))
                psi_values = 0
                #t = st.session_state.params.get("sigma_33") - st.session_state.params.get("sigma_11")
                strain_sim_params = (symmetry, lattice_params, wavelength, cijs, sigma_params, chi, phi_values, psi_values)
                XRD_df = Generate_XRD(selected_hkls, intensities, Gaussian_FWHM, strain_sim_params, Funamori_broadening)
                generate_1D_XRD_overlay(XRD_df, x_exp, y_exp)
        
            #Construct the default parameter dictionary for refinement
            other = {"chi" : chi}
        
            setup_refinement_toggles(lattice_params, cijs=cijs, stress=sigma_params, other=other)
            
            if st.button("Refine XRD"):
                phi_values = np.radians(np.arange(0, 360, 10))
                psi_values = 1
                
                result = run_refinement(st.session_state.ref_params, st.session_state.refine_flags, selected_hkls, selected_indices, intensities, Gaussian_FWHM, 
                                        phi_values, psi_values, wavelength, symmetry, x_exp, y_exp, lattice_params, cijs,
                                        sigma_params, chi, Funamori_broadening)
            
                if result.success:
                    st.success("Refinement successful!")
                    # Extract refined values from result.params
                    for key in st.session_state.params:
                        if key in result.params:
                            st.session_state.params[key] = result.params[key].value
                        else:
                            #Update the other lattice parameters that dont get refined for cubic etc
                            if key in ["b_val", "c_val"]:
                                if symmetry == "cubic":
                                    st.session_state.params[key] = result.params["a_val"].value
                                elif symmetry in ["hexagonal", "tetragonal_A", "tetragonal_B", "trigonal_A"]:
                                    if key == "b_val":
                                        st.session_state.params[key] = result.params["a_val"].value
                    
                    #Update the t and sigma values
                    t_opt = result.params["t"]
                    st.session_state.params["sigma_11"] = -t_opt / 3
                    st.session_state.params["sigma_22"] = -t_opt / 3
                    st.session_state.params["sigma_33"] = 2 * t_opt / 3
                    st.session_state.params["sigma_12"] = result.params["sigma_12"].value
                    st.session_state.params["sigma_13"] = result.params["sigma_13"].value
                    st.session_state.params["sigma_23"] = result.params["sigma_23"].value
    
                    #Update the intensity widgets and state values
                    
                    for key in st.session_state.intensities:
                        if key in result.params:
                            st.session_state.intensities[key] = result.params[key].value
                    
                    intensities = []
                    for i in selected_indices: 
                        intensities.append(st.session_state.intensities[f"intensity_{i}"])

                    #Ensure the parameters are updated for the plot
                    lattice_params = {
                        "a_val" : st.session_state.params.get("a_val"),
                        "b_val" : st.session_state.params.get("b_val"),
                        "c_val" : st.session_state.params.get("c_val"),
                        "alpha" : st.session_state.params.get("alpha"),
                        "beta" : st.session_state.params.get("beta"),
                        "gamma" : st.session_state.params.get("gamma"),
                    }
                    wavelength = st.session_state.params.get("wavelength")
                    chi = st.session_state.params.get("chi")
                    for key in sigma_keys:
                        sigma_params[key] = st.session_state.params.get(key)
                    c_keys = [key for key in st.session_state.params.keys() if key.startswith('c') and key not in ["c_val", "chi"]]
                    cijs = {}
                    for key in c_keys:
                        cijs[key] = st.session_state.params.get(key)
                            
                    st.markdown("### Fit Report")
                    report_str = fit_report(result)
                    st.code(report_str)
        
                    # Pack parameters for Generate_XRD
                    strain_sim_params = (
                        symmetry,
                        lattice_params,
                        wavelength,
                        cijs,
                        sigma_params,
                        chi,
                        phi_values,
                        psi_values
                    )
                    
                    XRD_df = Generate_XRD(selected_hkls, intensities, Gaussian_FWHM, strain_sim_params, Funamori_broadening)
                    #twoth_sim = XRD_df["2th"]
                    #intensity_sim = XRD_df["Total Intensity"]
                    #x_min_sim = np.min(twoth_sim)
                    #x_max_sim = np.max(twoth_sim)
                    #mask = (x_exp >= x_min_sim) & (x_exp <= x_max_sim)
                    #x_exp_common = x_exp[mask]
                    #y_exp_common = y_exp[mask]
                    #interp_sim = interp1d(twoth_sim, intensity_sim, bounds_error=False, fill_value=0)
                    #y_sim_common = interp_sim(x_exp_common)
        
                    #plot_overlay(x_exp_common, y_exp_common, x_exp_common, y_sim_common, title="Refined Fit")
                    generate_1D_XRD_overlay(XRD_df, x_exp, y_exp)
                    
                else:
                    st.error("Refinement failed.")
