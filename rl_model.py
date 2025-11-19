import numpy as np
import resources as r
from neural_network import NeuralNetwork
from abc import ABC, abstractmethod
from resources import game_core

class RLModel(ABC):
    def __init__(self, layer_sizes, activations):
       # Recording Trajectory
       self.states = []
       self.pre_squash_actions = []
       self.rewards = []

       self.actor = NeuralNetwork(layer_sizes, activations)
       # Learning rate
       self.gamma = 0.99

    def get_stochastic_actions(self, state_vector):
        # x: input vector

        x = np.array(state_vector, np.float32)
        if x.ndim == 1:
            x = np.expand_dims(x, axis=0)

        mu = self.actor.forward_propagation(x)
        sigma = np.exp(self.actor.log_std)
        noise = np.random.randn(*mu.shape) * sigma
        pre_squash = mu + noise
        actions = np.clip(pre_squash, -1, 1)
        return actions, pre_squash

    def get_deterministic_actions(self, state_vector):
        x = np.array(state_vector, np.float32)
        if x.ndim == 1:
            x = np.expand_dims(x, axis=0)

        mu = self.actor.forward_propagation(x)
        actions = np.tanh(mu) # Use tanh to squash to [-1, 1]
        return actions


    def update_trajectory(self, state, action, reward):
        self.states.append(state)
        self.pre_squash_actions.append(action)
        self.rewards.append(reward)


    def compute_return(self):
        rewards = np.array(self.rewards, np.float32)

        time_steps, batch = rewards.shape
        returns = np.zeros(rewards.shape, np.float32)
        G = np.zeros(batch, np.float32)

        for i in reversed(range(time_steps)):
            G = rewards[i] + self.gamma * G
            returns[i] = G

        return returns.flatten()

    @abstractmethod
    def update_params(self):
        pass

    def update_actor(self, advantage, states, actions):
        mu = self.actor.forward_propagation(states)
        sigma = np.exp(self.actor.log_std)

        d_mu =  -((actions - mu) / (sigma ** 2 + game_core.eps)) * advantage[:, None]
        d_log_std = np.mean(-(((actions - mu) ** 2) / (sigma ** 2) - 1) * advantage[:, None], axis = 0) - 0.1
        self.actor.update_params(d_mu, d_log_std)

    def clear_trajectory(self):
        self.states.clear()
        self.pre_squash_actions.clear()
        self.rewards = []


class REINFORCEModel(RLModel):
    def __init__(self, input_size, output_size = 2, hidden_layer_sizes=[128, 128, 128,128,128,128], seed=None):
        # Initialising neural network (actor for A2C)
        layer_sizes = [input_size] + hidden_layer_sizes + [output_size]
        activations = ['tanh'] * len(hidden_layer_sizes) + ['custom linear']
        super().__init__(layer_sizes, activations)

    def update_params(self):
        returns = self.compute_return()
        advantage = returns - returns.mean()
        self.update_actor(advantage)

        self.clear_trajectory()

class A2CModel(RLModel):
    def __init__(self, input_size, output_size = 2, actor_hidden_layers = [64, 64],
                 critic_hidden_layers = [64, 64], num_of_steps = 100, seed=None):
        actor_layers = [input_size] + actor_hidden_layers + [output_size]
        actor_activations = ['tanh'] * len(actor_hidden_layers) + ['tanh']
        super().__init__(actor_layers, actor_activations)

        critic_layer_sizes = [input_size] + critic_hidden_layers + [1]
        critic_activations = ['tanh'] * len(critic_hidden_layers) + ['linear']
        self.critic = NeuralNetwork(critic_layer_sizes, critic_activations, seed=seed)

        self.entropy = 0.01
        self.gae_lambda = 0.95

    def compute_gae_advantages(self, values):
        rewards = np.array(self.rewards, np.float32)
        time_steps, batch_size = rewards.shape
        advantages = np.zeros_like(rewards)
        gae = np.zeros(batch_size)

        next_values = np.vstack([values[1:], np.zeros((1, batch_size))])

        deltas = rewards + self.gamma * next_values - values

        for t in reversed(range(time_steps)):
            advantages[t] = deltas[t] + self.gamma * self.gae_lambda * gae
            gae = advantages[t]

        return advantages

    def update_params(self):

        states = np.vstack(self.states)
        actions = np.vstack(self.pre_squash_actions)
        values = self.critic.forward_propagation(states).reshape(-1)
        returns = self.compute_return()
        advantage = returns - values
        advantage = np.where(advantage > advantage.mean() , advantage *2, advantage)
        advantage = (advantage - advantage.mean()) / (advantage.std() + 1e-8)

        self.update_actor(advantage, states, actions)
        d_value = (values - returns).reshape(-1, 1)
        self.critic.update_params(d_value)

        print(f"Returns:{returns.mean(axis=0)} Value:{values.mean(axis=0)} Critic Loss:{np.sqrt(np.mean((values - returns) ** 2))}")

        self.clear_trajectory()

# class DDPGModel(RLModel):
#     def __init__(self, input_size, output_size = 2, actor_hidden_layers = [64, 64],
#                  critic_hidden_layers = [64, 64], seed=None):
#         actor_layers = [input_size] + actor_hidden_layers + [output_size]
#         actor_activations = ['tanh'] * len(actor_hidden_layers) + ['linear']
#         super().__init__(actor_layers, actor_activations)
#
#         critic_layer_sizes = [input_size] + critic_hidden_layers + [1]
#         critic_activations = ['tanh'] * len(critic_hidden_layers) + ['linear']
#         self.critic = NeuralNetwork(critic_layer_sizes, critic_activations, seed=seed)
#
#         self.next_states = []
#         self.dones = []
#         self.exploration_std = 0.1
#
#     def compute_targets(self):
#         states = np.vstack(self.states)
#         actions = np.vstack(self.actions)
#         next_states = np.vstack(self.next_states)
#         rewards = np.array(self.rewards, np.float32).reshape(-1, 1)
#         dones = np.array(self.dones, np.float32).reshape(-1, 1)
#
#         next_actions = np.tanh(self.actor.forward_propagation(next_states))
#         target_Q = self.critic.forward_propagation(np.hstack([next_states, next_actions]))
#         y = rewards + self.gamma * (1 - dones) * target_Q
#         return np.vstack(states), np.vstack(actions), y
#
#     def update_params(self):
#         returns = self.compute_return()
#         states = np.vstack(self.states)
#         values = self.critic.forward_propagation(states).reshape(-1)
#         advantage = returns - values
#         advantage = (advantage - advantage.mean()) / (advantage.std() + 1e-8)
#
#         self.update_actor(advantage)
#
#         d_value = (values - returns).reshape(-1, 1)
#         self.critic.update_params(d_value)
#
#         print(f"Returns:{returns.mean(axis=0)} Value:{values.mean(axis=0)} Critic Loss:{np.sqrt(np.mean((values - returns) ** 2))} "
#               f"Pre-squash:{np.array(self.pre_squash_actions).mean(axis = 0).mean(axis=0)} std:{np.exp(self.actor.log_std)}")
#
#         self.clear_trajectory()

