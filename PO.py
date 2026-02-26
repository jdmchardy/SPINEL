import numpy as np

class PO_Model:
    """
    Preferred Orientation Model
    Models the effects of preferred orientation on diffracted x-ray intensity
    """
    def __init__(self, po_model="MarchDollase",
                 components=[{"tau": 0,  "rho": 0,  "R": 1, "weight": 1}], #default of one component direction (R=1 is isotropic = no PO) aligned with stress z-axis
                 baseline=0, #A constant baseline value
                 chi_deg = 0
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
            "rho" : float (degrees)
                  The rotation angle around the stress axis
            "R" : float (Typically between 0 and 1)
                  The March-Dollase factor
            "weight" : float (Between 0 and 1)
                  The relative weight of the component.
        baseline : float
            A constant baseline value for the intensity. Between 0 and 1
        chi_deg : float (degrees)
            The chi angle between stress axis and x-ray axis
        """
        self.po_model = po_model
        self.components = components
        self.baseline = baseline
        self.chi = np.radians(chi_deg) #Convert to radians
        self.pref_directions = self.build_preferred_directions()

    def X_matrix(self, omega_deg, chi_deg):
        """Maps from xray coordinates to stress coordinates"""
    
        chi = np.radians(chi_deg)
        omega = np.radians(omega_deg)
    
        cos_chi = np.cos(chi)
        sin_chi = np.sin(chi)
        cos_omega = np.cos(omega)
        sin_omega = np.sin(omega)

        X = np.array([
            [cos_omega, sin_omega, 0],
            [-1*cos_chi*sin_omega, cos_chi*cos_omega, -1*sin_chi],
            [sin_chi*sin_omega, sin_chi*cos_omega, cos_chi]
        ])
        return X

    def transform_stress_2_xray(self, X, vector):
        """Transform a vector specified in stress coordinates to x-ray coordinates"""
        vector = vector/np.linalg.norm(vector) #normlise vector
        return np.linalg.inv(X) @ vector

    def make_polar_vector(self, tilt, rot):
        """
        tilt  = tilt from lab z-axis (radians)
        rot = rotation around z-axis (radians)
        """
        return np.stack([
            np.sin(tilt) * np.cos(rot),
            np.sin(tilt) * np.sin(rot),
            np.cos(tilt)
        ], axis=-1)

    def build_preferred_directions(self):
        """
        Generates the preferred directions in the xray coordinate system
        """
        pref_dirs = []

        chi_deg = np.degrees(self.chi)
        components = self.components
    
        #Compute X matrix
        X = self.X_matrix(0, chi_deg)
    
        for comp in components:
            tau = np.radians(comp["tau"])
            rho = np.radians(comp["rho"])
            vec = self.make_polar_vector(tau, rho) #Vector in the stress coordinates
            vec_xray = self.transform_stress_2_xray(X, vec) #vector transformed to the xray coordinates (rotation of chi)
            pref_dirs.append({
                "vector": vec_xray,
                "R": comp["R"],
                "weight": comp["weight"]
            })
        return pref_dirs

    def equal_area_projection(beta, gamma):
        r = 2 * np.sin(beta / 2) #With this scaling the circle radius is sqrt(2) giving area = 2pi (equal to the hemispere surface area for unit sphere)
        x = r * np.cos(gamma)
        y = r * np.sin(gamma)
        return x, y

    def spherical_to_vector(beta, gamma):
        return np.stack([
            np.sin(beta) * np.cos(gamma),
            np.sin(beta) * np.sin(gamma),
            np.cos(beta)
        ], axis=-1)

    def MD_func(self, alpha, R):
        """
        March-Dollase function.
        alpha: array-like of angles in radians
        R: scalar or array-like broadcastable to alpha
        Returns: MD values elementwise
        """
        alpha = np.asarray(alpha)
        return ((np.sin(alpha)**2)/R + (R**2)*(np.cos(alpha)**2))**(-3/2)

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
    
        # Normalize weights over preferred directions only
        weights_normed = weight_array / (np.sum(weight_array) + baseline)
    
        # Compute normalization factor for each preferred direction
        #This step is not required for the MD function
        #norm_factors = np.array([PO_normalization(R) for R in R_array])  # shape (n_pref,)
        
        # Evaluate MD function elementwise (broadcasting over last axis)
        P_alpha = self.MD_func(angle_array, R_array)  # (..., n_pref)
    
        # Weighted sum over preferred directions (last axis)
        return baseline + np.sum(P_alpha * weights_normed, axis=-1)

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
    
        intensity_u = self.intensity_from_directions(self,vectors_u)
    
        X_u, Y_u = self.equal_area_projection(BETA_u, GAMMA_u)
    
        # -------------------------
        # Lower hemisphere (south)
        # -------------------------
        beta_lower_geo = np.linspace(np.pi/2, np.pi, n_psi)
        BETA_l, GAMMA_l = np.meshgrid(beta_lower_geo, gamma, indexing="ij")
    
        vectors_l = self.spherical_to_vector(BETA_l, GAMMA_l)
    
        intensity_l = self.intensity_from_directions(self, vectors_l)
    
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

    def plot_intensity_pole_figure(self):
        upper, lower = self.compute_upper_lower_pole_data()
    
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
        
