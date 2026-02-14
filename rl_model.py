import numpy as np
import resources as r
from neural_network import NeuralNetwork
from abc import ABC, abstractmethod
from resources import game_core


class RLModel(ABC):
    def __init__(self, l2_lambda, layer_sizes, activations, gamma, entropy,init_log_std, learning_rate, beta1, beta2):
       # Recording Trajectory
       self.states = []
       self.pre_squash_actions = []
       self.rewards = []

       self.actor = NeuralNetwork(layer_sizes, activations, init_log_std, learning_rate, beta1, beta2)
       self.gamma = gamma
       self.entropy = entropy
       self.l2_lambda = l2_lambda

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

    def update_trajectory(self, state, action, reward):
        self.states.append(state)
        self.pre_squash_actions.append(action)
        self.rewards.append(reward)


    def compute_monte_carlo_return(self):
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

        d_mu = -((pre_squash_actions - mu) / (sigma ** 2 + 1e-12)) * advantage[:, None]
        d_mu += self.l2_lambda * mu
        d_mu = np.clip(d_mu, -1, 1)

        d_log_std = (np.mean(-(((pre_squash_actions - mu) ** 2) / (sigma ** 2) - 1) * advantage[:, None], axis = 0) +
                     self.entropy)
        self.actor.update_params(d_mu, d_log_std)

    def clear_trajectory(self):
        self.states.clear()
        self.pre_squash_actions.clear()
        self.rewards = []


class REINFORCEModel(RLModel):
    def __init__(self, input_size, activations= None, output_size = 2, hidden_layer_sizes= [ 32, 32, 64, 64, 64, 32, 32 ],
                  gamma = 0.98, entropy_bonus = 0.02, l2_lambda = 0.02,
                 init_log_std = 0.3, learning_rate = 3e-3, beta1 = 0.9, beta2 = 0.999):
        layer_sizes = [input_size] + hidden_layer_sizes + [output_size]
        if activations is None:
            activations = ['elu'] * len(hidden_layer_sizes) + ['linear']
        super().__init__(
            l2_lambda,
            layer_sizes,
            activations,
            gamma,
            entropy_bonus,
            init_log_std,
            learning_rate,
            beta1,
            beta2
        )

    def update_params(self):
        states = np.vstack(self.states)
        pre_squash_actions = np.vstack(self.pre_squash_actions)
        returns = self.compute_monte_carlo_return()
        advantage = (returns - returns.mean()) / (returns.std() + 1e-8)
        self.update_actor(advantage, states, pre_squash_actions)
        avg_rewards = np.array(self.rewards).mean()
        avg_returns = np.array(returns).mean()
        avg_std = np.exp(self.actor.log_std).mean()
        self.clear_trajectory()
        return (f"Reward: {avg_rewards:.3f} \n"
                f"Returns: {avg_returns:.3f} \n"
                f"Exploration (Std): {avg_std:.3f} \n")

class MCACModel(RLModel):
    def __init__(self, input_size, actor_activations= None, critic_activations = None, output_size=2,
                 gamma=0.98, l2_lambda=0.02, entropy_bonus=0.02,
                 actor_layer_sizes=[32, 32, 64, 64, 64, 32, 32],  actor_init_log_std=0.3,
                 actor_learning_rate=3e-3, actor_beta1=0.9, actor_beta2=0.999,
                 critic_layer_sizes=[16, 16], critic_init_log_std=0.3,
                 critic_learning_rate=3e-3, critic_beta1=0.9, critic_beta2=0.999
                 ):
        actor_layers = [input_size] + actor_layer_sizes + [output_size]
        if actor_activations is None:
            actor_activations = ['elu'] * len(actor_layer_sizes) + ['linear']
        if critic_activations is None:
            critic_activations = ['tanh'] * len(critic_layer_sizes) + ['linear']
        super().__init__(l2_lambda, actor_layers, actor_activations, gamma, entropy_bonus,
                         actor_init_log_std, actor_learning_rate, actor_beta1, actor_beta2)

        critic_layers = [input_size] + critic_layer_sizes + [1]

        self.critic = NeuralNetwork(critic_layers, critic_activations, critic_init_log_std, critic_learning_rate,
                                    critic_beta1, critic_beta2)

    def update_params(self):

        states = np.vstack(self.states)
        actions = np.vstack(self.pre_squash_actions)
        values = self.critic.forward_propagation(states).reshape(-1)
        returns = self.compute_monte_carlo_return()
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

        return (f"Reward: {avg_rewards:.3f} \n"
                f"Returns: {avg_returns:.3f} \n"
                f"Value: {avg_value:.3f} \n"
                f"Critic Loss: {avg_critic_loss:.3f} \n"
                f"Exploration (Std): {avg_std:.3f} \n")