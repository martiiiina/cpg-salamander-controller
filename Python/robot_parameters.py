"""Robot parameters"""

import numpy as np
from farms_core import pylog


class RobotParameters(dict):
    """Robot parameters"""

    __getattr__ = dict.__getitem__
    __setattr__ = dict.__setitem__

    def __init__(self, parameters):
        super().__init__()

        # Initialise parameters
        self.n_body_joints = parameters.n_body_joints
        self.n_legs_joints = parameters.n_legs_joints
        self.initial_phases = parameters.initial_phases
        self.n_joints = self.n_body_joints + self.n_legs_joints
        self.n_oscillators_body = 2*self.n_body_joints
        self.n_oscillators_legs = 2*self.n_legs_joints
        self.n_oscillators = self.n_oscillators_body + self.n_oscillators_legs
        self.freqs = np.zeros(self.n_oscillators)
        self.coupling_weights = np.zeros([
            self.n_oscillators,
            self.n_oscillators,
        ])
        self.phase_bias = np.zeros([
            self.n_oscillators,
            self.n_oscillators,
        ])
        self.rates = np.zeros(self.n_oscillators)
        self.nominal_amplitudes = np.zeros(self.n_oscillators)
        self.amplitude_gradient = parameters.amplitude_gradient if parameters.amplitude_gradient is not None else np.ones(16)
        # self.feedback_gains_swim = np.zeros(self.n_oscillators)
        # self.feedback_gains_walk = np.zeros(self.n_oscillators)

        # # gains for final motor output
        # self.position_body_gain = parameters.position_body_gain
        # self.position_limb_gain = parameters.position_limb_gain

        self.update(parameters)

    def update(self, parameters):
        """Update network from parameters"""
        self.set_frequencies(parameters)  # f_i
        self.set_coupling_weights(parameters)  # w_ij
        self.set_phase_bias(parameters)  # psi_ij
        self.set_amplitudes_rate(parameters)  # a_i
        self.set_nominal_amplitudes(parameters)  # R_i

    def step(self, time, iteration, salamandra_data):
        """Step function called at each iteration

        Parameters
        ----------

        salamanra_data: salamandra_simulation/data.py::SalamandraData
            Contains the robot data, including network and sensors.

        gps (within the method): Numpy array of shape [9x3]
            Numpy array of size 9x3 representing the GPS positions of each link
            of the robot along the body. The first index [0-8] coressponds to
            the link number from head to tail, and the second index [0,1,2]
            coressponds to the XYZ axis in world coordinate.

        """
        # Example to get global coordinates of robot links
        gps = np.array(
            salamandra_data.sensors.links.urdf_positions()[iteration, :9],
        )
        # Example to update the drive
        # self.sim_parameters.drive = ...
        # self.set_frequencies(self.sim_parameters)  # f_i
        # self.set_nominal_amplitudes(self.sim_parameters)  # R_i
        # print("GPGS: {}".format(gps[4, 0]))
        # print("drive: {}".format(self.sim_parameters.drive))

    def set_frequencies(self, parameters):
        """
        Set frequencies
        According to Ijspeert (2007) the frequency is a piece-wise linear function 
        depending on the drive, with different thresholds for body and limbs
        """
        #pylog.error('Coupling weights must be set')
        d = parameters.drive
        d_low = 1
        d_high_body = 5
        d_high_limbs = 3

        # Parameters from Ijspeert (2007)
        c1_body = 0.2
        c0_body = 0.3
        c1_limbs = 0.2
        c0_limbs = 0.0

        if d_low <= d <= d_high_body:   
            v_i_body = c1_body * d + c0_body   
        else:   
            v_i_body = 0

        if d_low <= d <= d_high_limbs:   
            v_i_limbs = c1_limbs * d + c0_limbs  
        else:   
            v_i_limbs = 0
        
        # NOTE: check if ODEs want frequencies in Hz or rad/s, otherwise multiply by 2pi
        self.freqs[0:16] = 2*np.pi*v_i_body
        self.freqs[16:32] = 2*np.pi*v_i_limbs
        
        # self.freqs[0:2] = self.freqs[0:2] * (1 + 0.2)
        # self.freqs[16:24] = self.freqs[16:24] * (1 + 0.1)


    def set_coupling_weights(self, parameters):
        """
        Set coupling weights
        Populates a 32x32 coupling matrix where each element (i,j) defines 
        the coupling strength between oscillators i and j
        """
        #pylog.error('Coupling weights must be set')
        # NOTE: updated wrt prof's paper
        w_body = 10.0
        w_limb_body = 30.0
        w_limb = 10.0
        w_intralimb = 5

        # Body: ipsilateral nearest-neighbor (head↔tail)
        for k in range(self.n_body_joints - 1):
            self.coupling_weights[2*k+2, 2*k] = w_body      # left downward
            self.coupling_weights[2*k, 2*k+2] = w_body      # left upward
            self.coupling_weights[2*k+3, 2*k+1] = w_body    # right downward
            self.coupling_weights[2*k+1, 2*k+3] = w_body    # right upward

        # Body: contralateral (left↔right same segment)
        for k in range(self.n_body_joints):
            self.coupling_weights[2*k, 2*k+1] = w_body
            self.coupling_weights[2*k+1, 2*k] = w_body

        # Limb: antagonist pairs within each joint (e.g. 16↔17, 18↔19, ...)
        for k in [16,20,24,28]:
            self.coupling_weights[k, k+1] = w_body
            self.coupling_weights[k+1, k] = w_body
            self.coupling_weights[k+2, k+3] = w_body
            self.coupling_weights[k+3, k+2] = w_body

        # Limb: girdle↔knee within same limb (for circular motion)
        for leg in [16, 20, 24, 28]:
            self.coupling_weights[leg+2, leg] = w_intralimb
            self.coupling_weights[leg, leg+2] = w_intralimb
            self.coupling_weights[leg+3, leg+1] = w_intralimb
            self.coupling_weights[leg+1, leg+3] = w_intralimb

        # Between limbs: trot pattern
        # Diagonal pairs in phase (left-fore↔right-hind, right-fore↔left-hind), Contralateral pairs antiphase (left-fore↔right-fore, left-hind↔right-hind)
        for a, b in [(16, 20), (24,28), (20,28), (16,24)]:
            self.coupling_weights[a, b] = w_limb
            self.coupling_weights[b, a] = w_limb

        # Limb→body: only girdle oscillators, unidirectional, strong coupling
        for limb_osc in [16]:   # left foreleg girdle → segment 0
            for body_osc in [0, 2, 4, 6, 8]:
                self.coupling_weights[limb_osc, body_osc] = w_limb_body
        for limb_osc in [20]:   # right foreleg girdle → segment 1
            for body_osc in [1, 3, 5, 7, 9]:
                self.coupling_weights[limb_osc, body_osc] = w_limb_body
        for limb_osc in [24]:   # left hindleg girdle → segment 6
            for body_osc in [ 10, 12, 14]:
                self.coupling_weights[limb_osc, body_osc] = w_limb_body
        for limb_osc in [28]:   # right hindleg girdle → segment 7
            for body_osc in [ 11, 13, 15]:
                self.coupling_weights[limb_osc, body_osc] = w_limb_body
                
        # for limb_osc in [17]:   # left foreleg girdle → segment 0
        #     for body_osc in [ 2, 4, 6, 8]:
        #         self.coupling_weights[body_osc, limb_osc] = -w_limb_body
        # for limb_osc in [21]:   # right foreleg girdle → segment 1
        #     for body_osc in [ 3, 5, 7, 9]:
        #         self.coupling_weights[body_osc, limb_osc] = -w_limb_body
        # for limb_osc in [25]:   # left hindleg girdle → segment 6
        #     for body_osc in [ 10, 12, 14]:
        #         self.coupling_weights[body_osc, limb_osc] = -w_limb_body
        # for limb_osc in [29]:   # right hindleg girdle → segment 7
        #     for body_osc in [ 11, 13, 15]:
        #         self.coupling_weights[body_osc, limb_osc] = -w_limb_body


    def set_phase_bias(self, parameters):
        """
        Set phase bias
        These are the desired phase biases between oscillators i and j
        """
        #pylog.error('Phase bias must be set')
        phase_lag = parameters.phase_lag_body if parameters.phase_lag_body is not None else 2*np.pi/8
        
        # Body: ipsilateral nearest-neighbor
        for k in range(self.n_body_joints - 1):
            self.phase_bias[2*k+2, 2*k] = phase_lag     # left downward
            self.phase_bias[2*k, 2*k+2] = -phase_lag      # left upward
            self.phase_bias[2*k+3, 2*k+1] = phase_lag   # right downward
            self.phase_bias[2*k+1, 2*k+3] = -phase_lag    # right upward

        # Body: contralateral
        for k in range(self.n_body_joints):
            self.phase_bias[2*k, 2*k+1] = np.pi
            self.phase_bias[2*k+1, 2*k] = np.pi

        # Limb: antagonist pairs antiphase 
        for k in [16,20,24,28]:
            self.phase_bias[k, k+1] = np.pi
            self.phase_bias[k+1, k] = np.pi
            self.phase_bias[k+2, k+3] = np.pi
            self.phase_bias[k+3, k+2] = np.pi

        # Limb: girdle↔knee π/2 offset for circular motion
        for leg in [16, 20, 24, 28]:
            self.phase_bias[leg+2, leg] = -np.pi / 2
            self.phase_bias[leg, leg+2] = -np.pi / 2
            self.phase_bias[leg+3, leg+1] = -np.pi / 2
            self.phase_bias[leg+1, leg+3] = -np.pi / 2

        # Between limbs: trot
        for a, b in [(16, 20), (24,28), (20,28), (16,24)]:
            self.phase_bias[a, b] = -np.pi
            self.phase_bias[b, a] = -np.pi

        # Limb→body: only girdle oscillators
        for limb_osc in [16]:
            for body_osc in [0, 2, 4, 6, 8]:
                self.phase_bias[limb_osc, body_osc] = 0
        for limb_osc in [20]:
            for body_osc in [1, 3, 5, 7, 9]:
                self.phase_bias[limb_osc, body_osc] = 0
        for limb_osc in [24]:
            for body_osc in [ 10, 12, 14]:
                self.phase_bias[limb_osc, body_osc] = 0
        for limb_osc in [28]:
            for body_osc in [ 11, 13, 15]:
                self.phase_bias[limb_osc, body_osc] = 0


    def set_amplitudes_rate(self, parameters):
        """
        Set amplitude rates
        This parameter defines how fast the amplitude envelope converges
        Defined as a scalar ai = 20 both for limbs and body  
        """
        #pylog.error('Convergence rates must be set')
        self.rates[:] = 20.0

    def set_nominal_amplitudes(self, parameters):
        """
        Set nominal amplitudes
        According to Ijspeert (2007) the nominal amplitude is a piece-wise linear function 
        depending on the drive, with different thresholds for body and limbs
        """
        #pylog.error('Nominal amplitudes must be set')
        d = parameters.drive
        d_low = 1
        d_high_body = 5
        d_high_limbs = 3
        
        # Parameters from Ijspeert (2007)
        c1_body = 0.065
        c0_body = 0.196
        c1_limbs = 0.131
        c0_limbs = 0.131

        if d_low <= d <= d_high_body:   
            R_i_body = c1_body * d + c0_body   
        else:   
            R_i_body = 0

        if d_low <= d <= d_high_limbs:   
            R_i_limbs = c1_limbs * d + c0_limbs  
        else:   
            R_i_limbs = 0

        self.nominal_amplitudes[0:16] = R_i_body
        self.nominal_amplitudes[16:32] = R_i_limbs
