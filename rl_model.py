import numpy as np
import resources as r
from neural_network import NeuralNetwork
class VPGModel:
    def __init__(self, input_size = 13, output_size = 2, hidden_layer_sizes=[64, 64], seed=None):
        layer_sizes = [input_size] + hidden_layer_sizes + [output_size]
        activations = ['relu'] * len(hidden_layer_sizes) + ['linear']
        self.neural_net = NeuralNetwork(layer_sizes, activations, seed=seed)

        self.states = []
        self.actions = []
        self.rewards = []
        self.gamma = 0.99

    def get_actions(self, state_vector):
        x = np.array(state_vector, np.float32)
        if x.ndim == 1:
            x = np.expand_dims(x, axis=0)

        mu, activation_outputs = self.neural_net.forward_propagation(x)
        sigma = np.exp(self.neural_net.log_std)
        noise = np.random.randn(*mu.shape) * sigma
        actions = mu + noise

        self.states.append(x)
        self.actions.append(actions)
        return actions

    def append_return(self, reward):
        self.rewards.append(reward)

    def compute_return(self):
        returns = []
        for i in range(len(self.rewards)):
            Gt = sum((self.rewards[i + j] * self.gamma ** j for j in range(len(self.rewards) - i)))
            returns.append(Gt)
        returns = np.array(returns, dtype= np.float32)
        # returns = (returns - returns.mean()) / (returns.std() + 1e-8)
        return returns

    def update_params(self):
        states = np.vstack(self.states)
        actions = np.vstack(self.actions)
        returns = self.compute_return()
        mu, _ = self.neural_net.forward_propagation(states)
        sigma = np.exp(self.neural_net.log_std)

        log_probs =  -0.5 * np.sum(((actions-mu) ** 2) / sigma ** 2 + 2 * self.neural_net.log_std + np.log(2 * np.pi), axis=1)
        objective = (log_probs * returns).mean()

        d_mu = (actions - mu) / (sigma ** 2) * returns[:, None]
        d_log_std = np.mean((((actions - mu) ** 2) / (sigma ** 2) - 1) * returns[:, None], axis = 0)

        self.neural_net.update_params(d_mu, d_log_std)
        self.states.clear()
        self.actions.clear()
        self.rewards.clear()
        return objective








