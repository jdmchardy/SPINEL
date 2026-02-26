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

    def draw_polar_grid(ax,
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
        
