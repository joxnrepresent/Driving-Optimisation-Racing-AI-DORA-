import numpy as np
import resources as r
from neural_network import NeuralNetwork
from abc import ABC, abstractmethod

class RLModel(ABC):
    def __init__(self, layer_sizes, activations):
       # Recording Trajectory
       self.states = []
       self.actions = []
       self.rewards = []

       self.actor = NeuralNetwork(layer_sizes, activations)
       # Learning rate
       self.gamma = 0.99

    def get_actions(self, state_vector):
        # x: input vector
        x = np.array(state_vector, np.float32)
        if x.ndim == 1:
            x = np.expand_dims(x, axis=0)

        mu = self.actor.forward_propagation(x)
        sigma = np.exp(self.actor.log_std)
        noise = np.random.randn(*mu.shape) * sigma
        actions = mu + noise

        self.states.append(x)
        self.actions.append(actions)
        return actions

    def append_reward(self, reward):
        self.rewards.append(reward)

    @abstractmethod
    def compute_return(self):
        pass

    @abstractmethod
    def update_params(self):
        pass


class REINFORCEModel(RLModel):
    def __init__(self, input_size, output_size = 2, hidden_layer_sizes=[128, 128, 128,128,128,128], seed=None):
        # Initialising neural network (actor for A2C)
        layer_sizes = [input_size] + hidden_layer_sizes + [output_size]
        activations = ['tanh'] * len(hidden_layer_sizes) + ['tanh']
        super().__init__(layer_sizes, activations)

    def compute_return(self):
        n = len(self.rewards)
        returns = np.zeros(n, dtype=np.float32)
        G = 0.0
        rewards = np.clip(self.rewards, -10, 10)
        for i in reversed(range(n)):
            G = rewards[i] + self.gamma * G
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
        mu = self.actor.forward_propagation(states)
        sigma = np.exp(self.actor.log_std)
        advantage = returns - returns.mean()

        d_mu = (actions - mu) / (sigma ** 2) * advantage[:, None]
        d_log_std = np.mean((((actions - mu) ** 2) / (sigma ** 2) - 1) * advantage[:, None], axis = 0)

        self.actor.update_params(d_mu, d_log_std)
        self.states.clear()
        self.actions.clear()
        self.rewards.clear()

class A2CModel(RLModel):
    def __init__(self, input_size, output_size = 2, actor_hidden_layers = [64, 64],
                 critic_hidden_layers = [64, 64], seed=None):
        actor_layers = [input_size] + actor_hidden_layers + [output_size]
        actor_activations = ['tanh'] * len(actor_hidden_layers) + ['tanh']
        super().__init__(actor_layers, actor_activations)

        critic_layer_sizes = [input_size] + critic_hidden_layers + [1]
        critic_activations = ['tanh'] * len(critic_hidden_layers) + ['linear']
        self.critic = NeuralNetwork(critic_layer_sizes, critic_activations, seed=seed)

        self.actor_grad_scale = 1
        self.critic_grad_scale = 1

    def compute_return(self):
        n = len(self.rewards)
        returns = np.zeros(n, dtype=np.float32)
        G = 0.0
        rewards = np.clip(self.rewards, -10, 10)
        for i in reversed(range(n)):
            G = rewards[i] + self.gamma * G
            returns[i] = G
        return returns

    def update_params(self):
        states = np.vstack(self.states)
        actions = np.vstack(self.actions)
        returns = self.compute_return()

        values = self.critic.forward_propagation(states).reshape(-1)
        advantage = returns - values
        advantage = (advantage - advantage.mean()) / (advantage.std() + 1e-8)

        mu = self.actor.forward_propagation(states)
        sigma = np.exp(self.actor.log_std)
        d_mu = (actions - mu) / (sigma ** 2) * advantage[:, None] * self.actor_grad_scale
        d_log_std = np.mean((((actions - mu) ** 2) / (sigma ** 2) - 1) * advantage[:, None], axis=0) * self.actor_grad_scale
        self.actor.update_params(d_mu, d_log_std)

        d_value = -(values - returns).reshape(-1, 1) * self.critic_grad_scale
        self.critic.update_params(d_value)

        self.states.clear()
        self.actions.clear()
        self.rewards = []



        # log_probs =  -0.5 * np.sum(((actions-mu) ** 2) / sigma ** 2 + 2 * self.actor.log_std + np.log(2 * np.pi), axis=1)
        # log_probs -= np.sum(np.log(1 - np.tanh(actions) ** 2 + 1e-8), axis=-1)
        # objective = (log_probs * advantage).mean()
        # return objective