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
            state = np.array(vector, copy=True)  # safe conversion, even if vector is a list
            noise = np.random.normal(0, noise_std, self.size_state)
            vector = [self.clip(float(state[i] + noise[i])) for i in range(self.size_state)]

        self.state = list(vector)  # always convert to plain Python list

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
    def __init__(self, pos, init_cell_state_value):
        self.size_state = 1
        self.size_chemicals_to_spread = 1
        self.size_phenotype = 1
        super().__init__(pos=pos, size_state=self.size_state, init_cell_state_value=init_cell_state_value)
        self.init_state()

    #---------------------------------------------------

    def init_state(self, random_init_bool=False):
        
        if self.init_cell_state_value is None or random_init_bool:
            self.state = np.random.uniform(0, 1, self.size_state).tolist()  # Python list
        else:
            self.state = [self.init_cell_state_value] * self.size_state  # Python list

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
    def __init__(self, pos, init_cell_state_value):
        self.size_state = 2
        self.size_chemicals_to_spread = 1
        self.size_phenotype = 1
        super().__init__(pos=pos, size_state=self.size_state, init_cell_state_value=init_cell_state_value)
        self.init_state()

    #---------------------------------------------------

    def init_state(self, random_init_bool=False):

        if self.init_cell_state_value is None or random_init_bool:
            self.state = np.random.uniform(-1, 1, self.size_state).tolist()  # Python list
        else:
            self.state = [self.init_cell_state_value] * self.size_state  # Python list

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

class agent3Outputs(swarmAgent):
    def __init__(self, pos, init_cell_state_value):
        self.size_state = 3
        self.size_chemicals_to_spread = 2
        self.size_phenotype = 1
        super().__init__(pos=pos, size_state=self.size_state, init_cell_state_value=init_cell_state_value)
        self.init_state()

    #---------------------------------------------------

    def init_state(self, random_init_bool=False):

        if self.init_cell_state_value is None or random_init_bool:
            self.state = np.random.uniform(-1, 1, self.size_state).tolist()  # Python list
        else:
            self.state = [self.init_cell_state_value] * self.size_state  # Python list

    #---------------------------------------------------

    def set_state(self, vector, with_noise_bool, noise_std):
        super().set_state(vector, with_noise_bool, noise_std)
    
    #---------------------------------------------------

    def get_external_chemicals_to_spread(self):
        state = super().get_state()
        return state[:self.size_chemicals_to_spread] # list of floats
    
    #---------------------------------------------------

    def get_phenotype(self):
        state = super().get_state()
        return (state[2] + 1) / 2 # (x+1)/2 to rescale (-1,1) in (0,1) keeping the scale. Issue with sigmoid(state[1]): sigmoid(x) with x in (-1, 1) returned inconvenient bounded result in [0.27, 0.73] and we need phenotype in [0,1]
    

###########################################################################

class agentCoordinates_gradient(swarmAgent):
    def __init__(self, pos, init_cell_state_value):
        self.size_state = 3
        self.size_chemicals_to_spread = 1
        self.size_phenotype = 2
        super().__init__(pos=pos, size_state=self.size_state, init_cell_state_value=init_cell_state_value)
        self.init_state()

    #---------------------------------------------------

    def init_state(self, random_init_bool=False):

        if self.init_cell_state_value is None or random_init_bool:
            self.state = np.random.uniform(-1, 1, self.size_state).tolist()  # Python list
        else:
            self.state = [self.init_cell_state_value] * self.size_state  # Python list

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

class agent2Outputs_RGB(swarmAgent):
    def __init__(self, pos, init_cell_state_value):
        self.size_state = 4
        self.size_chemicals_to_spread = 1
        self.size_phenotype = 3
        super().__init__(pos=pos, size_state=self.size_state, init_cell_state_value=init_cell_state_value)
        self.init_state()

    #---------------------------------------------------

    def init_state(self, random_init_bool=False):

        if self.init_cell_state_value is None or random_init_bool:
            self.state = np.random.uniform(-1, 1, self.size_state).tolist()  # Python list
        else:
            self.state = [self.init_cell_state_value] * self.size_state  # Python list

    #---------------------------------------------------

    def set_state(self, vector, with_noise_bool, noise_std):
        super().set_state(vector, with_noise_bool, noise_std)
    
    #---------------------------------------------------

    def get_external_chemicals_to_spread(self):
        state = super().get_state()
        return [state[0]] # list of floats
    
    #---------------------------------------------------

    def get_phenotype(self): # phenotype = [r,g,b]
        state = super().get_state()
        return [(state[1]+1)/2, (state[2]+1)/2, (state[3]+1)/2] # (x+1)/2 to rescale (-1,1) in (0,1) keeping the scale


###########################################################################

class agent3Outputs_RGB(swarmAgent):
    def __init__(self, pos, init_cell_state_value):
        self.size_state = 5
        self.size_chemicals_to_spread = 2
        self.size_phenotype = 3
        super().__init__(pos=pos, size_state=self.size_state, init_cell_state_value=init_cell_state_value)
        self.init_state()

    #---------------------------------------------------

    def init_state(self, random_init_bool=False):

        if self.init_cell_state_value is None or random_init_bool:
            self.state = np.random.uniform(-1, 1, self.size_state).tolist()  # Python list
        else:
            self.state = [self.init_cell_state_value] * self.size_state  # Python list

    #---------------------------------------------------

    def set_state(self, vector, with_noise_bool, noise_std):
        super().set_state(vector, with_noise_bool, noise_std)
    
    #---------------------------------------------------

    def get_external_chemicals_to_spread(self):
        state = super().get_state()
        return state[:self.size_chemicals_to_spread] # list of floats
    
    #---------------------------------------------------

    def get_phenotype(self):
        state = super().get_state()
        return [(state[2]+1)/2, (state[3]+1)/2, (state[4]+1)/2] # (x+1)/2 to rescale (-1,1) in (0,1) keeping the scale    