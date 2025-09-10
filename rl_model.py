import numpy as np
import resources as r
from neural_network import NeuralNetwork
class VPGModel:
    def __init__(self, input_size = len(r.RAY_CAST_ANGLES)*2 + 3, output_size = 2, hidden_layer_sizes=[64, 64], seed=None):
        layer_sizes = [input_size] + hidden_layer_sizes + [output_size]
        activations = ['relu'] * len(hidden_layer_sizes) + ['linear']
        self.neural_net = NeuralNetwork(layer_sizes, activations, seed=seed)

        self.s = []
        self.a = []
        self.r = []
        self.gamma = 0.99

    def get_actions(self, state_vector):
        x = np.array(state_vector, np.float32)
        if x.ndim == 1:
            x = np.expand_dims(x, axis=0)

        mu, activation_outputs = self.neural_net.forward_propagation(x)
        sigma = np.exp(self.neural_net.log_std)
        noise = np.random.randn(*mu.shape) * sigma
        actions = np.tanh(mu + noise)

        self.s.append(x)
        self.a.append(mu)
        return actions





