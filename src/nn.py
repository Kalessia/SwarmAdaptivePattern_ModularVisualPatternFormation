import matplotlib.pyplot as plt
import numpy as np
import copy


###########################################################################
# Activation functions
###########################################################################

def tanh(x):
    return np.tanh(x)

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def relu(x):
    return np.maximum(0, x)


###########################################################################
# Vectorized Forward Neural Network
###########################################################################

class NeuralNetwork:
    def __init__(self, input_size, hidden_layers, output_size, activation_function='tanh'):
        self.input_size = input_size
        self.hidden_layers = hidden_layers
        self.output_size = output_size
        self.weights = None
        self.biases = None
        self.weights_biases_size = self.get_weights_biases_size()
        self.activation_function = activation_function

        # Choose activation function
        if activation_function == 'tanh':
            self.activation = tanh
        elif activation_function == 'sigmoid':
            self.activation = sigmoid
        elif activation_function == 'relu':
            self.activation = relu
        else:
            raise ValueError("Unsupported activation function.")

    #---------------------------------------------------

    def get_weights_biases_size(self):
        size = 0
        layers = [self.input_size] + self.hidden_layers + [self.output_size]
        for l in range(len(layers) - 1):
            size += (layers[l] + 1) * layers[l+1]
        
        return size

    #---------------------------------------------------

    def set_weights_biases_vectors_from_list(self, original_weights_list):

        weights = [] # list of weights matrix, one weights matrix per layer
        biases = [] # list of list of biases, one biases list per layer
        weights_list = copy.deepcopy(original_weights_list)

        # Input and hidden layers
        previous_layer_size = self.input_size
        for nb_layer in range(len(self.hidden_layers)):
            weights_matrix = []
            for _ in range(previous_layer_size):
                weights_matrix.append(weights_list[:self.hidden_layers[nb_layer]])
                weights_list = weights_list[self.hidden_layers[nb_layer]:] # pop values
            weights.append(weights_matrix)
            biases.append(weights_list[:self.hidden_layers[nb_layer]])
            weights_list = weights_list[self.hidden_layers[nb_layer]:] # pop values
            previous_layer_size = self.hidden_layers[nb_layer]
        
        # Output layer
        weights_matrix = []
        for _ in range(previous_layer_size):
            weights_matrix.append(weights_list[:self.output_size])
            weights_list = weights_list[self.output_size:] # pop values
        weights.append(weights_matrix)
        biases.append(weights_list[:self.output_size])
        weights_list = weights_list[self.output_size:] # pop values

        self.weights = weights
        self.biases = biases

        # print("\nself.weights", self.weights) # check if correct in C pogobots code
        # print("\nself.biases", self.biases)

    #---------------------------------------------------
    
    def forward(self, input_layer):

        previous_layer = input_layer
        for weight_matrix, bias_vector in zip(self.weights, self.biases):
            weighted_sum = np.dot(previous_layer, weight_matrix) + bias_vector # linear combination
            previous_layer = self.activation(weighted_sum)

        return previous_layer # the last previous_layer is the output layer of the NN

    #---------------------------------------------------

    def predict(self, input_layer):
        return self.forward(input_layer)

    #---------------------------------------------------

    def get_weights_biases_for_pogobots(self): # ameliorer ^____________^
        s_weights = ""
        s_biases = ""
        s_pointers_weights = ""
        s_pointers_biases = ""
        s_pointers_weights_sizes = ""
        s_pointers_biases_sizes = ""
        s_pointers = ""
     
        s_variables = f"uint16_t nb_layers = {len(self.weights)+1};\n"
        s_variables += f"uint16_t input_size = {self.input_size};\n"
        s_variables += f"uint16_t output_size = {self.output_size};\n"

        for nb_layer in range(len(self.weights)):
            layer_id = f"l{nb_layer}_l{nb_layer+1}"
            nb_neurons_prev_layer = len(self.weights[nb_layer])
            nb_neurons_next_layer = len(self.weights[nb_layer][0])
            s_variables += f"uint16_t nb_neurons_l{nb_layer} = {nb_neurons_prev_layer};\n"
            s_weights += f"const double weights_{layer_id}[{nb_neurons_prev_layer}][{nb_neurons_next_layer}] = {(str(self.weights[nb_layer]).replace('[','{')).replace(']','}')};\n"
            s_biases += f"const double biases_{layer_id}[{nb_neurons_next_layer}] = {(str(self.biases[nb_layer]).replace('[','{')).replace(']','}')};\n"
            
            if nb_layer > 0:
                s_pointers_weights += f", weights_{layer_id}"
                s_pointers_biases += f", biases_{layer_id}"
                s_pointers_weights_sizes += f", {{{nb_neurons_prev_layer}, {nb_neurons_next_layer}}}"
                s_pointers_biases_sizes += f", {nb_neurons_next_layer}"
            else:
                s_pointers_weights += f"weights_{layer_id}"
                s_pointers_biases += f"biases_{layer_id}"
                s_pointers_weights_sizes += f"{{{nb_neurons_prev_layer}, {nb_neurons_next_layer}}}"
                s_pointers_biases_sizes += f"{nb_neurons_next_layer}"

        s_variables += f"uint16_t nb_neurons_l{nb_layer+1} = {nb_neurons_next_layer};\n" # output layer

        s_pointers = f"const double weights[] = {{{s_pointers_weights}}};\n"
        s_pointers += f"const double biases[] = {{{s_pointers_biases}}};\n"
        s_pointers += f"const int weights_sizes[] = {{{s_pointers_weights_sizes}}};\n"
        s_pointers += f"const int biases_sizes[] = {{{s_pointers_biases_sizes}}};\n"

        s_define = f"\n#define ACTIVATION_FUNCTION {self.activation_function}_activation\n"

        return s_variables + s_weights + s_biases + s_pointers + s_define
    

###########################################################################
# Neural Network structure visualization
###########################################################################

def draw_neural_network(ax, input_size, hidden_layers, output_size):
    # Layer sizes include input, hidden, and output layers
    layer_sizes = [input_size] + hidden_layers + [output_size]
    num_layers = len(layer_sizes)

    # Vertical and horizontal spacing
    v_spacing = 1.0 / float(max(layer_sizes))
    h_spacing = 1.0 / float(num_layers - 1)

    # Draw neurons and edges
    for n, layer_size in enumerate(layer_sizes):
        # Position neurons
        layer_top = v_spacing * (layer_size - 1) / 2.0  # Center the layer vertically
        for m in range(layer_size):
            # Draw neuron as a circle
            neuron = plt.Circle((n * h_spacing, layer_top - m * v_spacing), v_spacing / 4.,
                                color='w', ec='k', zorder=4)
            ax.add_artist(neuron)
        
        # Draw bias neuron for hidden and output layers (not for the input layer)
        if n < num_layers - 1:  # No bias for the output layer
            bias = plt.Circle((n * h_spacing, layer_top + v_spacing), v_spacing / 4.,
                              color='w', ec='k', zorder=4, linestyle='--')
            ax.add_artist(bias)

    # Draw edges
    for n, (layer_size_a, layer_size_b) in enumerate(zip(layer_sizes[:-1], layer_sizes[1:])):
        layer_top_a = v_spacing * (layer_size_a - 1) / 2.0
        layer_top_b = v_spacing * (layer_size_b - 1) / 2.0
        for m in range(layer_size_a):
            for o in range(layer_size_b):
                # Draw connections between neurons of adjacent layers
                line = plt.Line2D([n * h_spacing, (n + 1) * h_spacing],
                                  [layer_top_a - m * v_spacing, layer_top_b - o * v_spacing], c='k')
                ax.add_artist(line)
        
        # Draw connections from bias neuron to all neurons in the next layer
        bias_x = n * h_spacing
        bias_y = layer_top_a + v_spacing
        for o in range(layer_size_b):
            line = plt.Line2D([bias_x, (n + 1) * h_spacing],
                              [bias_y, layer_top_b - o * v_spacing], c='k', linestyle='--')
            ax.add_artist(line)

    plt.show()


###########################################################################
# Exemple usage
###########################################################################

# Neural network
# input_size = 3   # 3 input features
# hidden_layers = [4, 5]  # 2 hidden layers with 4 and 5 neurons
# output_size = 2  # 2 output neurons

# nn = NeuralNetwork(input_size, hidden_layers, output_size, activation_function='tanh')
# X = np.random.randn(1, input_size)  # Example input
# output = nn.forward(X)
# print("Network output:", output)

# # Plot
# fig, ax = plt.subplots(figsize=(10, 6))
# ax.axis('off')
# draw_neural_network(ax, input_size, hidden_layers, output_size)


