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
