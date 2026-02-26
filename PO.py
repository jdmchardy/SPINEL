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

        (X_u, Y_u, I_u) = upper
        (X_l, Y_l, I_l) = lower
    
        fig, axes = plt.subplots(1, 2, figsize=(8, 4), constrained_layout=True)
    
        radius = np.sqrt(2)
    
        # -------- upper --------
        cf1 = axes[0].contourf(X_u, Y_u, I_u, levels=100, cmap="viridis", vmin=0)
        axes[0].add_artist(plt.Circle((0, 0), radius, fill=False, linewidth=1.5))
    
        self.draw_polar_grid(axes[0])
    
        axes[0].set_aspect("equal")
        axes[0].set_title("Upper hemisphere")
        axes[0].set_xlabel("X")
        axes[0].set_ylabel("Y")
    
        # -------- lower --------
        cf2 = axes[1].contourf(X_l, Y_l, I_l, levels=100, cmap="viridis", vmin=0)
        axes[1].add_artist(plt.Circle((0, 0), radius, fill=False, linewidth=1.5))
    
        self.draw_polar_grid(axes[1])
    
        axes[1].set_aspect("equal")
        axes[1].set_title("Lower hemisphere")
        axes[1].set_xlabel("X")
        axes[1].set_ylabel("Y")
    
        lim = radius
        axes[0].set_xlim(-lim, lim)
        axes[0].set_ylim(-lim, lim)
        axes[1].set_xlim(-lim, lim)
        axes[1].set_ylim(-lim, lim)
        
        fig.colorbar(cf1, ax=axes, label="Intensity")
        fig.suptitle("Intensity Pole Figure")
        
