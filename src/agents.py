import random
import numpy as np


###########################################################################
# Swarm agents
###########################################################################

class swarmAgent:
    def __init__(self, pos, size_state, init_cell_state_value=None) -> None:

        self.pos = pos
        self.size_state = size_state
        self.init_cell_state_value = init_cell_state_value
        self.neighbors_NWES = None
        self.state = None

    #---------------------------------------------------
    
    def init_state(self):
        raise NotImplementedError("Subclasses must implement this method")
    
    #---------------------------------------------------

    def set_state(self, vector, with_noise_bool, noise_std):

        if with_noise_bool:
            state = vector.copy()
            noise = np.random.normal(0, noise_std, self.size_state) 
            vector = [self.clip(state[i]+noise[i]) for i in range(self.size_state)] # check  

        self.state = vector

    #---------------------------------------------------

    def clip(self, value):
        return max(0, min(value, 1))

    #---------------------------------------------------

    def get_state(self):
        return self.state
    
    #---------------------------------------------------

    def get_external_chemicals_to_spread(self):
        raise NotImplementedError("Subclasses must implement this method")
    
    #---------------------------------------------------
    
    def get_phenotype(self):
        raise NotImplementedError("Subclasses must implement this method")


###########################################################################

class agent1Output(swarmAgent):
    def __init__(self, pos, init_cell_state_value, agent_additional_weights=None):
        self.size_state = 1
        self.size_chemicals_to_spread = 1
        self.size_phenotype = 1
        super().__init__(pos=pos, size_state=self.size_state, init_cell_state_value=init_cell_state_value)
        self.init_state()

    #---------------------------------------------------

    def init_state(self, random_init_bool=False):
        
        if self.init_cell_state_value is None or random_init_bool:
            self.state = np.random.uniform(0, 1, self.size_state).tolist()
        else:
            self.state = [self.init_cell_state_value] * self.size_state

    #---------------------------------------------------

    def set_state(self, vector, with_noise_bool, noise_std):
        super().set_state(vector, with_noise_bool, noise_std)
    
    #---------------------------------------------------

    def get_external_chemicals_to_spread(self):
        state = super().get_state()
        return state # list of floats
    
    #---------------------------------------------------

    def get_phenotype(self):
        state = super().get_state()
        return (state[0] + 1) / 2 # (x+1)/2 to rescale (-1,1) in (0,1) keeping the scale. Issue with sigmoid(state[0]): sigmoid(x) with x in (-1, 1) returned inconvenient bounded result in [0.27, 0.73] and we need phenotype in [0,1]

###########################################################################

class agent2Outputs(swarmAgent):
    def __init__(self, pos, init_cell_state_value, agent_additional_weights=None):
        self.size_state = 2
        self.size_chemicals_to_spread = 1
        self.size_phenotype = 1
        super().__init__(pos=pos, size_state=self.size_state, init_cell_state_value=init_cell_state_value)
        self.init_state()

    #---------------------------------------------------

    def init_state(self, random_init_bool=False):

        if self.init_cell_state_value is None or random_init_bool:
            self.state = np.random.uniform(-1, 1, self.size_state).tolist()
            self.state[1] = np.abs(self.state[1])
        else:
            self.state = [self.init_cell_state_value] * self.size_state

    #---------------------------------------------------

    def set_state(self, vector, with_noise_bool, noise_std):
        super().set_state(vector, with_noise_bool, noise_std)
    
    #---------------------------------------------------

    def get_external_chemicals_to_spread(self):
        state = super().get_state()
        return [state[0]] # list of floats
    
    #---------------------------------------------------

    def get_phenotype(self):
        state = super().get_state()
        return (state[1] + 1) / 2 # (x+1)/2 to rescale (-1,1) in (0,1) keeping the scale. Issue with sigmoid(state[1]): sigmoid(x) with x in (-1, 1) returned inconvenient bounded result in [0.27, 0.73] and we need phenotype in [0,1]
    

###########################################################################

class agent3Outputs_Devert2011(swarmAgent):
    def __init__(self, pos, init_cell_state_value, agent_additional_weights=None):
        self.size_state = 3
        self.size_chemicals_to_spread = 2
        self.size_phenotype = 1
        self.agent_additional_weights = agent_additional_weights
        super().__init__(pos=pos, size_state=self.size_state, init_cell_state_value=init_cell_state_value)
        self.init_state()

    #---------------------------------------------------

    def init_state(self, random_init_bool=False):
        if self.init_cell_state_value is None or random_init_bool:
            self.state = np.random.uniform(-1, 1, self.size_state).tolist() # Devert, 2011. State in [-1,1] because phenotype is not included in the state
        else:
            self.state = [self.init_cell_state_value] * self.size_state

    #---------------------------------------------------

    def set_state(self, vector, with_noise_bool, noise_std):
        super().set_state(vector, with_noise_bool, noise_std)
    
    #---------------------------------------------------

    def get_external_chemicals_to_spread(self): # Devert, 2011. A state is 2 external chemicals + 1 internal chemicals
        state = super().get_state()
        return state[:self.size_chemicals_to_spread]

    #---------------------------------------------------

    def get_internal_chemicals(self): # Devert, 2011. A state is 2 external chemicals + 1 internal chemicals
        state = super().get_state()
        return state[self.size_chemicals_to_spread:] # list of floats
    
    #---------------------------------------------------

    def get_phenotype(self): # Devert, 2011. Expression function
        state = super().get_state()
        
        val = 0
        for i in range(len(state)):
            val += self.agent_additional_weights[i] * state[i]
        val += self.agent_additional_weights[-1] # bias neuron

        return (0.5 * (1 + np.tanh(val))) # float
    
    #---------------------------------------------------

    def get_agent_energy(self): # Devert, 2011. Expression function: energy is the square root of the sum of the squared values of chemicals u and v
        state = super().get_state()
        return (state[0]**2 + state[1]**2 + state[2]**2)**0.5


###########################################################################

class agentCoordinates_gradient(swarmAgent):
    def __init__(self, pos, init_cell_state_value, agent_additional_weights=None):
        self.size_state = 3
        self.size_chemicals_to_spread = 1
        self.size_phenotype = 2
        super().__init__(pos=pos, size_state=self.size_state, init_cell_state_value=init_cell_state_value)
        self.init_state()

    #---------------------------------------------------

    def init_state(self, random_init_bool=False):

        if self.init_cell_state_value is None or random_init_bool:
            self.state = np.random.uniform(-1, 1, self.size_state).tolist()
            self.state[1] = np.abs(self.state[1])
        else:
            self.state = [self.init_cell_state_value] * self.size_state

    #---------------------------------------------------

    def set_state(self, vector, with_noise_bool, noise_std):
        super().set_state(vector, with_noise_bool, noise_std)
    
    #---------------------------------------------------

    def get_external_chemicals_to_spread(self):
        state = super().get_state()
        return [state[0]] # list of floats
    
    #---------------------------------------------------

    def get_phenotype(self): # phenotype = [x,y]
        state = super().get_state()
        return [(state[1]+1)/2, (state[2]+1)/2] # (x+1)/2 to rescale (-1,1) in (0,1) keeping the scale


###########################################################################

class agentCoordinates_xy_map(swarmAgent):
    def __init__(self, pos, init_cell_state_value, agent_additional_weights=None):
        self.size_state = 2
        self.size_chemicals_to_spread = 1
        self.size_phenotype = 1
        super().__init__(pos=pos, size_state=self.size_state, init_cell_state_value=init_cell_state_value)
        self.init_state()

    #---------------------------------------------------

    def init_state(self, random_init_bool=False):

        if self.init_cell_state_value is None or random_init_bool:
            self.state = np.random.uniform(-1, 1, self.size_state).tolist()
            self.state[1] = np.abs(self.state[1])
        else:
            self.state = [self.init_cell_state_value] * self.size_state

    #---------------------------------------------------

    def set_state(self, vector, with_noise_bool, noise_std):
        super().set_state(vector, with_noise_bool, noise_std)
    
    #---------------------------------------------------

    def get_external_chemicals_to_spread(self):
        state = super().get_state()
        return [state[0]] # list of floats
    
    #---------------------------------------------------

    def get_phenotype(self):
        state = super().get_state()
        return (state[1] + 1) / 2