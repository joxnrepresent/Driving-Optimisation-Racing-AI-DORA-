def draw(screen, elements):
    for element in elements:
        if isinstance(element[0], pygame.Rect):
            pygame.draw.rect(screen, element[1], element[0])
        elif isinstance(element[0], pygame.Surface):
            screen.blit(element[0], element[1])
        elif element[0] == "semicircle":
            draw_semicircle(screen, * element[1:])
        elif element[0] == "line":
            pygame.draw.line(screen, * element[1:])

def draw_semicircle(screen, center, radius, fill_colour, start_angle = 0, border_width = 5, border_colour = "Black", resolution = 5 ):
    points = [center]
    for i in range(180*resolution + 1 ):
        angle = start_angle + i/resolution
        x = center[0] + radius * math.cos(math.radians(angle))
        y = center[1] - radius * math.sin(math.radians(angle))
        points.append((x, y))
    pygame.draw.polygon(screen, fill_colour, points)
    pygame.draw.polygon(screen, border_colour, points, border_width)


    DRAG_COEFFICIENT = 0.4257
    ROLLING_RESISTANCE_COEFFICIENT = 12.8


#
# class Meter:
#     def __init__(self, min_value, max_value):
#         self.min_value = min_value
#         self.max_value = max_value
#         self.value = 0
#         self.range_of_values = self.max_value - self.min_value
#         self.dragging = False
#
#     def update_value(self, value):
#         self.value = resources.clamp_value(value, self.min_value, self.max_value)
#         return (self.value - self.min_value) / self.range_of_values
#
# class HorizontalMeter(Meter):
#     def __init__(self, x, y, width, height, base_colour, knob_colour, min_value, max_value):
#         super().__init__(min_value, max_value)
#         self.knob_width = 10
#         self.base = pygame.Rect(x, y, width, height)
#         self.knob = pygame.Rect((x + (width - self.knob_width) // 2), y, self.knob_width, height)
#         self.elements = ((self.base, base_colour), (self.knob, knob_colour))
#
#     def update_value(self, value):
#         relative_value = super().update_value(value)
#         self.knob.x = relative_value * self.base.width + self.base.left

# class Slider(HorizontalMeter):
#     def event_handle(self, event):
#         if self.knob.collidepoint(event.pos):
#             if event.type == pygame.MOUSEBUTTONDOWN and event.button == 1:
#                self.dragging = True
#             if event.type == pygame.MOUSEBUTTONUP and event.button == 1:
#                 self.dragging = False
#
#         if self.dragging:
#             click_position = resources.clamp_value(event.pos[0], self.base.left, self.base.right)
#             relative_position = click_position - self.base.left
#
#             percentage_increment = relative_position / (self.base.right - self.base.left)
#             value = round(percentage_increment * self.range_of_values, 2) + self.min_value
#             super().update_value(value)
#


import numpy as np
from neural_network import NeuralNetwork  # your custom NN class
import resources as r


class VPGAgent:
    def __init__(self, state_size, action_size, hidden_sizes=[64, 64], seed=None):
        # Policy network (outputs mean of Gaussian for actions)
        layer_sizes = [state_size] + hidden_sizes + [action_size]
        activations = ['relu'] * len(hidden_sizes) + ['linear']
        self.policy_net = NeuralNetwork(layer_sizes, activations, seed=seed)

        # Experience storage
        self.states = []
        self.actions = []
        self.rewards = []
        self.gamma = 0.99  # discount factor

    def get_action(self, state):
        state = np.array(state, dtype=np.float32).reshape(1, -1)
        mean, _ = self.policy_net.forward_propagation(state)
        std = np.exp(self.policy_net.log_std)
        action = mean + std * np.random.randn(*mean.shape)
        return action.flatten()  # 1D array: [steer, throttle]

    def store_transition(self, state, action, reward):
        self.states.append(state)
        self.actions.append(action)
        self.rewards.append(reward)

    def compute_returns(self):
        """Compute discounted returns and normalize"""
        returns = []
        G = 0
        for rwd in reversed(self.rewards):
            G = rwd + self.gamma * G
            returns.insert(0, G)
        returns = np.array(returns)
        returns = (returns - returns.mean()) / (returns.std() + 1e-8)
        return returns

    def update_policy(self):
        """Vanilla policy gradient update"""
        states = np.array(self.states)
        actions = np.array(self.actions)
        returns = self.compute_returns()

        # Forward pass to get mean of policy
        mean, _ = self.policy_net.forward_propagation(states)
        std = np.exp(self.policy_net.log_std)

        # Compute log probability of taken actions
        log_probs = -0.5 * np.sum(((actions - mean) ** 2) / (std ** 2) + 2 * np.log(std) + np.log(2 * np.pi), axis=1)

        # Policy gradient: grad = log_prob * return
        d_logp = (log_probs * returns)[:, None]  # simple scaling, batch dimension

        # Backpropagate through your NN
        self.policy_net.compute_gradients(d_logp)
        self.policy_net.update_parameters(np.zeros_like(self.policy_net.log_std))  # no update for log_std yet

        # Clear episode memory
        self.states, self.actions, self.rewards = [], [], []
