"""Simulation parameters"""


class SimulationParameters:
    """Simulation parameters"""

    def __init__(self, **kwargs):
        super(SimulationParameters, self).__init__()
        # Default parameters
        self.n_body_joints = 8
        self.n_legs_joints = 8
        self.duration = 30
        self.initial_phases = None
        self.drive = 0.0
        self.position_body_gain = 0.6  # default do not change
        self.position_limb_gain = 1  # default do not change
        self.phase_lag_body = None
        self.amplitude_gradient = None
        self.w_limb_body = 150
        self.limb_body_phase_offset = 0.0
        self.body_gain = 1.0
        self.limb_gain = 1.0
        self.update_drive = False
        self.drive_ramp = None
        # Feel free to add more parameters (ex: MLR drive)
        # self.drive_mlr = ...
        # ...

        # Update object with provided keyword arguments
        # NOTE: This overrides the previous declarations
        self.__dict__.update(kwargs)

