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
       self.entropy = 0.02

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
        # x: input vector
        x = np.array(state_vector, np.float32)
        if x.ndim == 1:
            x = np.expand_dims(x, axis=0)
        mu = self.actor.forward_propagation(x)
        actions = np.clip(mu, -1, 1)
        return actions[0]

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

    def update_actor(self, advantage, states, pre_squash_actions):
        mu = self.actor.forward_propagation(states)
        sigma = np.exp(self.actor.log_std)

        d_mu = -((pre_squash_actions - mu) / (sigma ** 2 + game_core.eps)) * advantage[:, None]
        l2_lamda = 0.01
        d_mu += l2_lamda * mu
        d_mu = np.clip(d_mu, -1, 1)

        d_log_std = np.mean(-(((pre_squash_actions - mu) ** 2) / (sigma ** 2) - 1) * advantage[:, None], axis = 0) + self.entropy
        self.actor.update_params(d_mu, d_log_std)

    def clear_trajectory(self):
        self.states.clear()
        self.pre_squash_actions.clear()
        self.rewards = []


class REINFORCEModel(RLModel):
    def __init__(self, input_size, output_size = 2, hidden_layer_sizes= [ 32, 32, 64, 64, 64, 32, 32 ], seed=None):
        layer_sizes = [input_size] + hidden_layer_sizes + [output_size]
        activations = ['elu'] * len(hidden_layer_sizes) + ['linear']
        super().__init__(layer_sizes, activations)

    def update_params(self):
        states = np.vstack(self.states)
        pre_squash_actions = np.vstack(self.pre_squash_actions)
        returns = self.compute_return()
        advantage = (returns - returns.mean()) / (returns.std() + 1e-8)
        self.update_actor(advantage, states, pre_squash_actions)
        avg_rewards = np.array(self.rewards).mean()
        avg_returns = np.array(returns).mean()
        self.clear_trajectory()
        return (f"Reward:{avg_rewards:.3f} \n"
                f"Returns:{avg_returns:.3f} \n")

class MCACModel(RLModel):
    def __init__(self, input_size, output_size = 2, actor_hidden_layers = [ 32, 64, 64, 64, 32 ],
                 critic_hidden_layers = [16,16], num_of_steps = 100, seed=None):
        actor_layers = [input_size] + actor_hidden_layers + [output_size]
        actor_activations = ['elu'] * len(actor_hidden_layers) + ['linear']
        super().__init__(actor_layers, actor_activations)

        critic_layer_sizes = [input_size] + critic_hidden_layers + [1]
        critic_activations = ['tanh'] * len(critic_hidden_layers) + ['linear']
        self.critic = NeuralNetwork(critic_layer_sizes, critic_activations, seed=seed)

        self.num_of_steps = num_of_steps
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
        advantage = (advantage - advantage.mean()) / (advantage.std() + 1e-8)

        self.update_actor(advantage, states, actions)
        d_value = (values - returns).reshape(-1, 1)
        self.critic.update_params(d_value)

        avg_rewards = np.array(self.rewards).mean()
        avg_returns = np.array(returns).mean()
        avg_value = np.array(values).mean()
        avg_critic_loss = np.sqrt(np.mean((values - returns) ** 2))
        avg_std = np.exp(self.actor.log_std).mean()
        self.clear_trajectory()

        return (f"Reward:{avg_rewards:.3f} \n"
                f"Returns:{avg_returns:.3f} \n"
                f"Value:{avg_value:.3f} \n"
                f"Critic Loss:{avg_critic_loss:.3f} \n"
                f"Exploration (Std) :{avg_std:.3f} \n")

        # print(f"Value range: [{values.min():.2f}, {values.max():.2f}] | Return range: [{returns.min():.2f}, {returns.max():.2f}]")
