import numpy as np
import resources as r
from neural_network import NeuralNetwork
class VPGModel:
    def __init__(self, input_size, output_size = 2, hidden_layer_sizes=[128, 128, 128,128,128,128], seed=None):
        layer_sizes = [input_size] + hidden_layer_sizes + [output_size]
        activations = ['tanh'] * len(hidden_layer_sizes) + ['linear']
        self.neural_net = NeuralNetwork(layer_sizes, activations, seed=seed)

        self.states = []
        self.actions = []
        self.rewards = []
        self.gamma = 0.95

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
        n = len(self.rewards)
        returns = np.zeros(n, dtype=np.float32)
        G = 0.0
        self.rewards = np.clip(self.rewards, -10, 10)
        for i in reversed(range(n)):
            G = self.rewards[i] + self.gamma * G
            returns[i] = G
        if returns.std() > 0:
            returns = (returns - returns.mean()) / (returns.std() + 1e-8)
        else:
            returns = returns - returns.mean()

        returns = np.clip(returns, -5, 5)
        return returns

    def update_params(self):
        states = np.vstack(self.states)
        actions = np.vstack(self.actions)
        returns = self.compute_return()
        mu, _ = self.neural_net.forward_propagation(states)
        sigma = np.exp(self.neural_net.log_std)

        log_probs =  -0.5 * np.sum(((actions-mu) ** 2) / sigma ** 2 + 2 * self.neural_net.log_std + np.log(2 * np.pi), axis=1)
        log_probs -= np.sum(np.log(1 - np.tanh(actions) ** 2 + 1e-8), axis=-1)
        objective = (log_probs * returns).mean()

        advantage = returns - returns.mean()
        d_mu = (actions - mu) / (sigma ** 2) * advantage[:, None]
        d_log_std = np.mean((((actions - mu) ** 2) / (sigma ** 2) - 1) * advantage[:, None], axis = 0)
        # debug diagnostics (put before neural_net.update_params call)
        print("returns mean/std:", returns.mean(), returns.std())
        print("mu mean/std:", mu.mean(), mu.std())
        print("actions mean/std:", actions.mean(), actions.std())
        print("log_std (pre):", self.neural_net.log_std)

        self.neural_net.update_params(d_mu, d_log_std)
        self.states.clear()
        self.actions.clear()
        self.rewards = []
        return objective
