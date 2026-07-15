import numpy as np
import matplotlib.pyplot as plt
import itertools
import streamlit as st
import math

def compute_alpha(theta0, chi_rad, delta_grid_rad):
    """
    Rotation angle alpha about z_S that keeps the scattering vector K in the
    y_S-z_S plane of the stress frame. Reduces to alpha = delta
    in the axial case (chi = 0). Shared with SPINEL.compute_strain.
    Derived from the constraint dot product x1.K^s = 0, i.e the scattering vector is perpendicular to x1 of the stress coordinates which
    yields the relation

        tan α = (cos θ₀ sin δ) /
                (cos χ cos θ₀ cos δ - sin χ sin θ₀).

    Limiting cases:

      * χ = 0  (axial): denominator reduces to cos θ₀ cos δ, giving
                  α = arctan2(sin δ, cos δ) = δ. z_x-ray and z_s are
                  aligned, so α tracks δ over the full (−π, π].
      * χ = π/2 (radial): denominator reduces to sin θ₀, giving
                  tan α = -cos θ₀ sin δ / sin θ₀ = -sin δ / tan θ₀.
                  α is bounded in (−(π/2 − θ₀), π/2 − θ₀) and smooth.

    Parameters
    ----------
    theta0 : float
        Bragg angle θ₀ in radians. Assumed to lie in (0, π/2).
    chi_rad : float or ndarray
        Sample/detector tilt angle χ in radians (angle between the x-ray
        and stress z-axes).
    delta_grid_rad : ndarray
        Azimuth δ around the Debye–Scherrer ring, in radians, typically
        spanning (−π, π].

    Returns
    -------
    alpha : ndarray
        Signed rotation angle in radians, in (−π, π], broadcast over the
        inputs. Continuous in δ except for the arctan2 branch wrap at
        δ = ±π.

    """
    num = np.cos(theta0) * np.sin(delta_grid_rad)
    den = (np.cos(chi_rad) * np.cos(theta0) * np.cos(delta_grid_rad)
           - np.sin(chi_rad) * np.sin(theta0))
    return np.arctan2(num, den)

class PO_Model:
    """
    Preferred Orientation Model
    Models the effects of preferred orientation on diffracted x-ray intensity
    """
    def __init__(self, po_model="MarchDollase",
                 components=[{"tau": 0, "omega": 0,  "R": 1, "weight": 1}], #default of one component direction (R=1 is isotropic = no PO) aligned with stress z-axis
                 baseline=0, #A constant baseline value
                 symmetry = "cubic", 
                 wavelength = 0.4,
                 lattice_params = {"a_val": 3,
                                   "b_val": 3,
                                   "c_val": 3,
                                   "alpha": 90,
                                   "beta": 90,
                                   "gamma":90
                                  },
                 chi_deg = 0,
                 POD_xtal = (0,0,1) #Define the plane normal for POD in xtal coordinates (default is z-axis)
                ):
        """
        Parameters
        ----------
        po_model : str
            Preferred orientation model to use.
        components : list of dic
            One dictionary per component direction. Each dictionary contains
            "tau" : float (degrees)
                  The tilt angle from the stress axis
            "omega" : float (degrees)
                  The rotation angle around the stress axis
            "R" : float (Typically between 0 and 1)
                  The March-Dollase factor
            "weight" : float (Between 0 and 1)
                  The relative weight of the component.
        baseline : float
            A constant baseline value for the intensity. Between 0 and 1
        symmetry : str
            Crystal symmetry
        wavelength : float
            X-ray wavelength (Ang)
        lattice_params : dict
            The lattice parameter dictionary
        chi_deg : float (degrees)
            The chi angle between stress axis and x-ray axis
        """
        #PO model parameters
        self.po_model = po_model
        self.components = components
        self.baseline = baseline
        #Crystal/geometry parameters
        self.symmetry = symmetry
        self. wavelength = wavelength
        self.lattice_params = lattice_params
        self.chi = np.radians(chi_deg) #Convert to radians
        self.POD_xtal = POD_xtal
        self.pref_directions = self.build_preferred_directions()

    def get_permutations(self, hkl):
        """Generates all the permutaions given some seed hkl)"""
        
        # Step 1: generate all sign variations for each hkl
        signed_variations = [(n, -n) for n in hkl]
        
        # Step 2: generate Cartesian product of all sign combinations
        all_sign_combinations = itertools.product(*signed_variations)
        
        # Step 3: generate permutations for each combination
        all_permutations = set()
        for combo in all_sign_combinations:
            for perm in itertools.permutations(combo):
                all_permutations.add(perm)
        
        # Convert set to list and print
        all_permutations = list(all_permutations)
        num_perms = len(all_permutations)
        return num_perms, all_permutations

    def get_d0(self, hkl):
        """Evaluates the lattice plane spacing"""
        symmetry = self.symmetry
        a = self.lattice_params.get("a_val")
        b = self.lattice_params.get("b_val")
        c = self.lattice_params.get("c_val")
        h,k,l = hkl
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

    def get_theta(self, d):
        #Returns theta (Bragg angle) in radians
        wavelength = self.wavelength
        sin_theta = wavelength / (2 * d)
        theta = np.arcsin(sin_theta)
        return theta

    def get_psi(self, hkl, delta_deg):
        d0 = self.get_d0(hkl)
        theta0 = self.get_theta(d0)
        chi = self.chi
        deltas = np.radians(delta_deg)
    
        cos_psi = np.sin(chi)*np.cos(theta0)*np.cos(deltas)+np.cos(chi)*np.sin(theta0)
        psi = np.degrees(np.arccos(cos_psi))
        return psi

    def A_matrix_vectorised(self, phi, psi):
        """
        Matrix to transform from diffraction plane to stress coordinates (Uchida matrix)
        Has shape (N,M,3,3) where N,M are the length of the phi,psi 1d arrays
        phi : 1d_array (radians)
        psi : 1d_array (radians)
        """
        #Compute sin and cosine
        cos_phi = np.cos(phi)
        sin_phi = np.sin(phi)
        cos_psi = np.cos(psi)
        sin_psi = np.sin(psi)

        #Create mesgrids
        cos_phi, cos_psi = np.meshgrid(cos_phi, cos_psi, indexing='ij')
        sin_phi, sin_psi = np.meshgrid(sin_phi, sin_psi, indexing='ij')
        
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
        return A

    def B_matrix(self, hkl):
        """
        Matrix to transform from diffraction plane to crystal coordinates
        """
        a = self.lattice_params.get("a_val")
        b = self.lattice_params.get("b_val")
        c = self.lattice_params.get("c_val")
        alpha = self.lattice_params.get("alpha")
        beta = self.lattice_params.get("beta")
        gamma = self.lattice_params.get("gamma")
        
        h, k, l = hkl
        if h == 0: h = 0.0000000001
        if k == 0: k = 0.0000000001
        if l == 0: l = 0.0000000001

        symmetry = self.symmetry
            
        if symmetry == "cubic":
            H = h / a
            K = k / a
            L = l / a
        elif symmetry in ["tetragonal_A", "tetragonal_B"]:
            H = h / a
            K = k / a
            L = l / c
        elif symmetry == "hexagonal":
            H = h / a
            K = (h+2*k) / (np.sqrt(3)*a)
            L = l / c
        elif symmetry == "orthorhombic":
            H = h / a
            K = k / b
            L = l / c
        elif symmetry == "trigonal_A":
            H = h / a
            K = (h+2*k) / (np.sqrt(3)*a)
            L = l / c
            
        # N and M from normalised hkls
        N = np.sqrt(K**2 + L**2)
        M = np.sqrt(H**2 + K**2 + L**2)
        
        B = np.array([
            [N/M, 0, H/M],
            [-H*K/(N*M), L/N, K/M],
            [-H*L/(N*M), -K/N, L/M]
        ])
        return B

    def X_matrix(self, alpha_deg, chi_deg):
        """
        Maps x-ray coordinates -> stress coordinates.
            alpha (deg) : rotation about z_S (=delta for chi=0)
            chi   (deg) : tilt of z_S from the x-ray z-axis
        Orthonormal, so inv(X) = X.T = the stress -> x-ray map
        """
        chi = np.radians(chi_deg)
        alpha = np.radians(alpha_deg)

        cos_chi, sin_chi = np.cos(chi), np.sin(chi)
        cos_alpha, sin_alpha = np.cos(alpha), np.sin(alpha)

        X = np.array([
            [cos_alpha,-1*sin_alpha*cos_chi,sin_alpha*sin_chi],
            [sin_alpha,cos_alpha*cos_chi,-1*cos_alpha*sin_chi],
            [0,sin_chi,cos_chi]
        ])
        return X

    def transform_stress_2_xray(self, X, vector):
        """Transform a vector specified in stress coordinates to x-ray coordinates"""
        vector = vector/np.linalg.norm(vector) #normlise vector
        return np.linalg.inv(X) @ vector

    def make_polar_vector(self, tilt, rot):
        """
        tilt = tilt from the frame z-axis (radians)
        rot  = azimuth measured from the y-axis (radians) 
        """
        return np.stack([
            np.sin(tilt) * np.sin(rot),
            np.sin(tilt) * np.cos(rot),
            np.cos(tilt)
        ], axis=-1)

    def build_preferred_directions(self):
        """
        Generates the preferred directions in the xray coordinate system
        """
        pref_dirs = []
        X = self.X_matrix(0, np.degrees(self.chi)) # x-ray -> stress, alpha=0
        for comp in self.components:
            tau   = np.radians(comp["tau"])
            omega = np.radians(comp["omega"])
            vec_S    = self.make_polar_vector(tau, omega)    # POA in stress coords
            vec_xray = self.transform_stress_2_xray(X, vec_S)  # applies inv(X) @ v to convert from stress to x-ray coords
            pref_dirs.append({"vector": vec_xray, "R": comp["R"], "weight": comp["weight"]})
        return pref_dirs

    def equal_area_projection(self, beta, gamma):
        # radius sqrt(2) at beta=pi/2 -> area 2*pi (equal to the hemispere surface area for unit sphere)
        r = 2 * np.sin(beta / 2)
        # gamma measured from the y-axis: gamma=0 -> +Y, increasing toward +X
        x = r * np.sin(gamma)
        y = r * np.cos(gamma)
        return x, y

    def spherical_to_vector(self, beta, gamma):
        """
        beta  = tilt from the z-axis (radians)
        gamma = azimuth measured from the y-axis (radians)
        """
        return np.stack([
            np.sin(beta) * np.sin(gamma),
            np.sin(beta) * np.cos(gamma),
            np.cos(beta)
        ], axis=-1)

    def MD_func(self, eta, R):
        """
        March-Dollase function.
        eta: array-like of angles in radians
        R: scalar or array-like broadcastable to alpha
        Returns: MD values elementwise
        """
        eta = np.asarray(eta)
        return ((np.sin(eta)**2)/R + (R**2)*(np.cos(eta)**2))**(-3/2)

    def multi_MD_PO_model(self, angle_array, R_array, weight_array):
        """
        Vectorized March-Dollase sum over preferred directions.
    
        angle_array: (..., n_pref) array of angles in radians
        R_array: (n_pref,) array of March-Dollase parameters
        weight_array: (n_pref,) array of weights
    
        Returns: (...,) array of intensities for each input vector
        """
        angle_array = np.asarray(angle_array)
        R_array = np.asarray(R_array)
        weight_array = np.asarray(weight_array)
        baseline = self.baseline
    
        # Normalize weights to account for baseline fraction
        tol = 1e-6
        if math.isclose(np.sum(weight_array), 0.0, abs_tol=tol):
            #Normalise baseline to 1 
            baseline = 1
            weights_normed = weight_array*0
        else:
            weights_normed = weight_array * (1-baseline) / np.sum(weight_array)
        
        # Evaluate MD function elementwise (broadcasting over last axis)
        P_eta = self.MD_func(angle_array, R_array)  # (..., n_pref)
    
        # Weighted sum over preferred directions (last axis)
        return baseline + np.sum(P_eta * weights_normed, axis=-1)

    def po_intensity_engine(self, hkl, phi_grid, psi_grid, alpha_grid):
        """
        PO intensity on matched orientation grids (phi, psi, alpha) — all in
        radians, identical shape. Shared engine for the delta-parametrised path
        (intensity_for_hkl) and the direct-orientation path used for the
        Funamori integration (intensity_from_orientation).
        """
        num_perms, all_permutations = self.get_permutations(hkl)

        # Uchida A(phi, psi) built element-wise on the grids
        cos_phi, sin_phi = np.cos(phi_grid), np.sin(phi_grid)
        cos_psi, sin_psi = np.cos(psi_grid), np.sin(psi_grid)
        A = np.empty(phi_grid.shape + (3, 3))
        A[..., 0, 0] = cos_phi;  A[..., 0, 1] = -sin_phi * cos_psi;  A[..., 0, 2] = sin_phi * spsi
        A[..., 1, 0] = sin_phi;  A[..., 1, 1] =  cos_phi * cos_psi;  A[..., 1, 2] = -cos_phi * sin_psi
        A[..., 2, 0] = 0.0;  A[..., 2, 1] =  sin_psi;  A[..., 2, 2] = cos_psi

        # Merkel alpha rotation:  A_full = A_Uchida @ R_z(-alpha)
        cos_alpha = np.cos(alpha_grid)[..., None]
        sin_alpha = np.sin(alpha_grid)[..., None]
        A_full = np.empty_like(A)
        A_full[..., 0] = A[..., 0] * cos_alpha + A[..., 1] * (-sin_alpha)
        A_full[..., 1] = A[..., 0] * sin_alpha + A[..., 1] * cos_alpha
        A_full[..., 2] = A[..., 2]
        A_full_T = np.swapaxes(A_full, -1, -2)  # diffraction -> stress matrix

        # fixed stress -> x-ray, pre-composed (same rotation the POA uses)
        X_s2x = self.X_matrix(0, np.degrees(self.chi)).T #Transpose the X-matrix which is the inverse
        M = X_s2x @ A_full_T #Now diffraction -> x-ray matrix

        POD_xtal = np.asarray(self.POD_xtal, dtype=float)
        POD_xtal = POD_xtal / np.linalg.norm(POD_xtal)

        I = np.zeros(phi_grid.shape, dtype=float)
        for hkl_perm in all_permutations:
            B   = self.B_matrix(hkl_perm)
            vec = B.T @ POD_xtal  # crystal -> diffraction transformation
            POD_xray = np.einsum('...ij,j->...i', M, vec) #diffraction -> x-ray transformation
            I += self.intensity_from_directions(POD_xray)
        return I / num_perms

    def intensity_for_hkl(self, hkl, phi, delta):
        """
        Computes the PO intensity over a (phi, delta) grid, averaged across hkl permutations.
        psi and alpha are derived from the azimuth delta; 
        POD is transformed from crystal -> diffraction (B^T) -> stress (A_full^T) -> x-ray (fixed X_s2x) 
        The PO engine used is po_intensity_engine.
        
        Parameters: 
        ---------------
        hkl : tuple
            (h,k,l) giving the miller indice of the unique reflection
        phi : 1d.array
            The phi values (degrees)
        delta : 1d.array
            The delta (azimuth) values (degrees)
        Returns:
        ---------------
        I : mesh_grid object (tuple of intensity value arrays of shape (phi, delta))
        """
        psi = self.get_psi(hkl, delta)               # degrees
        theta0 = self.get_theta(self.get_d0(hkl))

        phi = np.radians(phi)
        psi = np.radians(psi)
        delta = np.radians(delta)

        phi_grid, delta_grid = np.meshgrid(phi, delta, indexing="ij")
        _, psi_grid   = np.meshgrid(phi, psi, indexing="ij")   # psi rides the delta axis
        alpha_grid = compute_alpha(theta0, self.chi, delta_grid)

        I = self._po_intensity_engine(hkl, phi_grid, psi_grid, alpha_grid)
        return I, np.degrees(phi_grid), np.degrees(delta_grid)

    def intensity_from_orientation(self, hkl, phi_grid, psi_grid, alpha_grid):
        """
        PO intensity on an explicit (phi, psi, alpha) orientation grid (radians,
        matched shapes). Use when orientation is parametrised directly rather
        than via delta (e.g. the Funamori integration).
        """
        return self.po_intensity_engine(hkl, phi_grid, psi_grid, alpha_grid)
        
    def intensity_from_directions(self, vectors):
        """
        Vectorized intensity computation using multi_MD_PO_model.
    
        vectors: (..., 3)
        """
        vectors = vectors / np.linalg.norm(vectors, axis=-1, keepdims=True)
        pref_dirs = self.pref_directions
    
        # Extract arrays
        pref_vectors = np.array([d["vector"] for d in pref_dirs])  # (n_pref, 3)
        Rs = np.array([d["R"] for d in pref_dirs])         # (n_pref,)
        weights = np.array([d["weight"] for d in pref_dirs])       # (n_pref,)
    
        # Normalize preferred vectors
        pref_vectors = pref_vectors / np.linalg.norm(pref_vectors, axis=-1, keepdims=True)
    
        # Compute cos(angle) using einsum (broadcasting)
        cosang = np.einsum('...i,ji->...j', vectors, pref_vectors)
        cosang = np.clip(cosang, -1, 1)
        angles = np.arccos(cosang)  # (..., n_pref)
    
        # Compute intensity using vectorized multi_MD_PO_model
        I = self.multi_MD_PO_model(angles, Rs, weights)
    
        return I

    def compute_upper_lower_pole_data(self,
                                      n_psi=181,
                                      n_gamma=360
                                     ):
    
        gamma = np.linspace(0, 2*np.pi, n_gamma)
        # -------------------------
        # Upper hemisphere (north)
        # -------------------------
        beta_upper = np.linspace(0, np.pi/2, n_psi)
        BETA_u, GAMMA_u = np.meshgrid(beta_upper, gamma, indexing="ij")
    
        vectors_u = self.spherical_to_vector(BETA_u, GAMMA_u)
    
        intensity_u = self.intensity_from_directions(vectors_u)
    
        X_u, Y_u = self.equal_area_projection(BETA_u, GAMMA_u)
    
        # -------------------------
        # Lower hemisphere (south)
        # -------------------------
        beta_lower_geo = np.linspace(np.pi/2, np.pi, n_psi)
        BETA_l, GAMMA_l = np.meshgrid(beta_lower_geo, gamma, indexing="ij")
    
        vectors_l = self.spherical_to_vector(BETA_l, GAMMA_l)
    
        intensity_l = self.intensity_from_directions(vectors_l)
    
        # distance from SOUTH pole for projection
        beta_from_south = np.pi - BETA_l
        X_l, Y_l = self.equal_area_projection(beta_from_south, GAMMA_l)
    
        return (X_u, Y_u, intensity_u), (X_l, Y_l, intensity_l)

    def draw_polar_grid(self, 
                        ax,
                        beta_step_deg=15,
                        gamma_step_deg=30,
                        n_curve=400):
        # -----------------------
        # constant psi (circles)
        # -----------------------
        gamma = np.linspace(0, 2*np.pi, n_curve)
        beta_vals = np.deg2rad(np.arange(beta_step_deg, 90, beta_step_deg))
    
        for beta in beta_vals:
            BETA = np.full_like(gamma, beta)
            X, Y = self.equal_area_projection(BETA, gamma)
            ax.plot(X, Y, color="white", linewidth=0.6, alpha=0.9)
        # -----------------------
        # constant gamma (spokes)
        # -----------------------
        beta = np.linspace(0, np.pi/2, n_curve)
        gamma_vals = np.deg2rad(np.arange(0, 360, gamma_step_deg))
    
        for g in gamma_vals:
            GAMMA = np.full_like(beta, g)
            X, Y = self.equal_area_projection(beta, GAMMA)
            ax.plot(X, Y, color="white", linewidth=0.6, alpha=0.9)

    def make_intensity_pole_figure(self):
        (X_u, Y_u, I_u), (X_l, Y_l, I_l) = self.compute_upper_lower_pole_data()
    
        fig, axes = plt.subplots(1, 2, figsize=(8, 4), constrained_layout=True)
    
        radius = np.sqrt(2)
        # -------- upper hemisphere--------
        cf1 = axes[0].contourf(X_u, Y_u, I_u, levels=100, cmap="viridis", vmin=0)
        axes[0].add_artist(plt.Circle((0, 0), radius, fill=False, linewidth=1.5))
        axes[0].set_title("Upper hemisphere")
    
        # -------- lower hemisphere--------
        cf2 = axes[1].contourf(X_l, Y_l, I_l, levels=100, cmap="viridis", vmin=0)
        axes[1].add_artist(plt.Circle((0, 0), radius, fill=False, linewidth=1.5))
        axes[1].set_title("Lower hemisphere")
        
        #Format plots
        for ax in axes:
            self.draw_polar_grid(ax)
            ax.set_xlim(-radius, radius)
            ax.set_ylim(-radius, radius)
            ax.set_xlabel("X")
            ax.set_ylabel("Y")
            ax.set_aspect("equal")
        
        fig.colorbar(cf1, ax=axes, label="Intensity")
        fig.suptitle("Intensity Pole Figure")
        
