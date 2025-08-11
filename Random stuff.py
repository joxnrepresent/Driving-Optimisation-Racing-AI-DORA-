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

# import numpy as np
# import copy
#
# # ----------------------------
# # Activation function classes
# # ----------------------------
# class Activation:
#     def forward(self, z): raise NotImplementedError
#     def derivative(self, z): raise NotImplementedError  # derivative wrt pre-activation z
#
# class ReLU(Activation):
#     def forward(self, z): return np.maximum(0.0, z)
#     def derivative(self, z): return (z > 0).astype(float)
#
# class Tanh(Activation):
#     def forward(self, z): return np.tanh(z)
#     def derivative(self, z):
#         t = np.tanh(z)
#         return 1.0 - t*t
#
# class Sigmoid(Activation):
#     def forward(self, z):
#         return 1.0 / (1.0 + np.exp(-z))
#     def derivative(self, z):
#         s = self.forward(z)
#         return s * (1.0 - s)
#
# class Identity(Activation):
#     def forward(self, z): return z
#     def derivative(self, z): return np.ones_like(z)
#
# # activation factory
# def get_activation(name):
#     name = name.lower()
#     if name in ('relu',): return ReLU()
#     if name in ('tanh',): return Tanh()
#     if name in ('sigmoid','logistic'): return Sigmoid()
#     if name in ('identity','linear','none'): return Identity()
#     raise ValueError(f"Unknown activation '{name}'")
#
# # ----------------------------
# # Weight initialization helper
# # ----------------------------
# def init_weights(in_dim, out_dim, activation):
#     """
#     - For relu: He normal
#     - For tanh/sigmoid/identity: Glorot (Xavier) uniform
#     Returns shape (in_dim, out_dim)
#     """
#     act_name = activation.__class__.__name__.lower()
#     if 'relu' in act_name:
#         std = np.sqrt(2.0 / in_dim)
#         return np.random.randn(in_dim, out_dim) * std
#     else:
#         limit = np.sqrt(6.0 / (in_dim + out_dim))
#         return np.random.uniform(-limit, limit, size=(in_dim, out_dim))
#
# # ----------------------------
# # Layer class (linear + activation)
# # ----------------------------
# class Layer:
#     def __init__(self, in_dim, out_dim, activation='relu'):
#         self.in_dim = in_dim
#         self.out_dim = out_dim
#         self.activation = get_activation(activation)
#         self.weights = init_weights(in_dim, out_dim, self.activation)  # shape (in, out)
#         self.biases = np.zeros(out_dim)  # shape (out,)
#
#         # caches for backprop
#         self._last_input = None    # shape (batch, in)
#         self._last_z = None        # shape (batch, out)
#         self._last_a = None        # shape (batch, out)
#
#     def forward(self, x):
#         """
#         x: (batch, in_dim)
#         returns a: (batch, out_dim)
#         """
#         z = x.dot(self.weights) + self.biases  # broadcasting biases (out,)
#         a = self.activation.forward(z)
#         # cache
#         self._last_input = x
#         self._last_z = z
#         self._last_a = a
#         return a
#
#     def backward(self, dA):
#         """
#         dA: dL/dA for this layer (batch, out_dim)
#         returns:
#           - dX: dL/dX to pass to previous layer (batch, in_dim)
#           - dW: gradient for weights (in_dim, out_dim)
#           - db: gradient for biases (out_dim,)
#         Notes: we compute average gradient over batch (i.e., divide by batch_size)
#         """
#         batch = dA.shape[0]
#         z = self._last_z
#         x = self._last_input
#         dZ = dA * self.activation.derivative(z)   # (batch, out)
#         dW = x.T.dot(dZ) / batch                   # (in, out)
#         db = dZ.mean(axis=0)                       # (out,)
#         dX = dZ.dot(self.weights.T)                # (batch, in)
#         return dX, dW, db
#
#     def get_params(self):
#         return {'w': self.weights.copy(), 'b': self.biases.copy()}
#
#     def set_params(self, params):
#         self.weights = params['w'].copy()
#         self.biases = params['b'].copy()
#
#
# # ----------------------------
# # NeuralNetwork class
# # ----------------------------
# class NeuralNetwork:
#     def __init__(self, layer_sizes, activations=None, learn_log_std=False, init_log_std=-1.0, seed=None):
#         """
#         layer_sizes: list like [input_dim, hidden1, ..., output_dim]
#         activations: list of activation names for layers (len = len(layer_sizes)-1) or None:
#                      if None -> hidden: 'relu', output: 'tanh' (common for continuous [-1,1])
#         learn_log_std: if True, network has a learnable log_std vector for output dims (useful for VPG)
#         init_log_std: initial value for log_std (scalar or array-like)
#         """
#         if seed is not None:
#             np.random.seed(seed)
#         L = len(layer_sizes)-1
#         if activations is None:
#             activations = ['relu']*(L-1) + ['tanh']
#         assert len(activations) == L
#
#         self.layers = []
#         for i in range(L):
#             self.layers.append(Layer(layer_sizes[i], layer_sizes[i+1], activation=activations[i]))
#
#         self.learn_log_std = bool(learn_log_std)
#         if self.learn_log_std:
#             output_dim = layer_sizes[-1]
#             if np.isscalar(init_log_std):
#                 self.log_std = np.full(output_dim, init_log_std, dtype=float)
#             else:
#                 self.log_std = np.array(init_log_std, dtype=float).copy()
#         else:
#             self.log_std = None
#
#         # optimizer state for Adam
#         self._opt_state = {'m_w': [np.zeros_like(l.weights) for l in self.layers],
#                            'v_w': [np.zeros_like(l.weights) for l in self.layers],
#                            'm_b': [np.zeros_like(l.biases) for l in self.layers],
#                            'v_b': [np.zeros_like(l.biases) for l in self.layers],
#                            't': 0}
#         if self.learn_log_std:
#             self._opt_state.update({'m_logstd': np.zeros_like(self.log_std),
#                                     'v_logstd': np.zeros_like(self.log_std)})
#
#     # ----------------------------
#     # Forward and prediction
#     # ----------------------------
#     def forward(self, x, return_all=False):
#         """
#         x: (batch, input_dim)
#         return_all: if True returns list of activations (including input) for inspection
#         returns: output (batch, output_dim) or (output, activations_list)
#         """
#         a = x
#         activations = [a]
#         for layer in self.layers:
#             a = layer.forward(a)
#             activations.append(a)
#         if return_all:
#             return a, activations
#         return a
#
#     def predict(self, x):
#         return self.forward(x, return_all=False)
#
#     # ----------------------------
#     # Backprop (compute gradients)
#     # ----------------------------
#     def compute_gradients(self, dLoss_dOutput):
#         """
#         dLoss_dOutput: (batch, output_dim) gradient of loss w.r.t network output (the last activation).
#         This method will run a backward pass using cached forward activations. It returns a `grads` dict:
#           grads = {'dW': [..], 'dB': [..], 'dLogStd': gradient_or_None}
#         Note: this function DOES NOT update weights; it only returns gradients.
#         """
#         grads_w = [None] * len(self.layers)
#         grads_b = [None] * len(self.layers)
#         dA = dLoss_dOutput
#         # iterate backward
#         for i in reversed(range(len(self.layers))):
#             layer = self.layers[i]
#             dA, dW, db = layer.backward(dA)
#             grads_w[i] = dW
#             grads_b[i] = db
#
#         grads = {'dW': grads_w, 'dB': grads_b}
#         # we don't produce dLogStd here — user must compute and pass if learn_log_std is used
#         return grads
#
#     # ----------------------------
#     # Apply gradients (SGD or Adam)
#     # ----------------------------
#     def apply_gradients(self, grads, lr=1e-3, optimizer='sgd', beta1=0.9, beta2=0.999, eps=1e-8, grad_logstd=None):
#         """
#         grads: dict returned by compute_gradients
#         lr: learning rate
#         optimizer: 'sgd' or 'adam'
#         grad_logstd: if learn_log_std is True, pass gradient vector for log_std (shape = output_dim)
#         """
#         if optimizer.lower() == 'sgd':
#             for i, layer in enumerate(self.layers):
#                 layer.weights -= lr * grads['dW'][i]
#                 layer.biases -= lr * grads['dB'][i]
#             if self.learn_log_std and (grad_logstd is not None):
#                 self.log_std -= lr * grad_logstd
#         elif optimizer.lower() == 'adam':
#             st = self._opt_state
#             st['t'] += 1
#             t = st['t']
#             for i, layer in enumerate(self.layers):
#                 g_w = grads['dW'][i
#                 g_b = grads['dB'][i]
#                 st['m_w'][i] = beta1 * st['m_w'][i] + (1 - beta1) * g_w
#                 st['v_w'][i] = beta2 * st['v_w'][i] + (1 - beta2) * (g_w * g_w)
#                 mhat_w = st['m_w'][i] / (1 - beta1**t)
#                 vhat_w = st['v_w'][i] / (1 - beta2**t)
#                 layer.weights -= lr * mhat_w / (np.sqrt(vhat_w) + eps)
#
#                 st['m_b'][i] = beta1 * st['m_b'][i] + (1 - beta1) * g_b
#                 st['v_b'][i] = beta2 * st['v_b'][i] + (1 - beta2) * (g_b * g_b)
#                 mhat_b = st['m_b'][i] / (1 - beta1**t)
#                 vhat_b = st['v_b'][i] / (1 - beta2**t)
#                 layer.biases -= lr * mhat_b / (np.sqrt(vhat_b) + eps)
#
#             if self.learn_log_std and (grad_logstd is not None):
#                 st['m_logstd'] = beta1 * st['m_logstd'] + (1 - beta1) * grad_logstd
#                 st['v_logstd'] = beta2 * st['v_logstd'] + (1 - beta2) * (grad_logstd * grad_logstd)
#                 mhat = st['m_logstd'] / (1 - beta1**t)
#                 vhat = st['v_logstd'] / (1 - beta2**t)
#                 self.log_std -= lr * mhat / (np.sqrt(vhat) + eps)
#         else:
#             raise ValueError("optimizer must be 'sgd' or 'adam'")



    # ----------------------------
    # Parameter vectorization for EA
    # ----------------------------
    def get_params_vector(self):
        """
        Flatten all layer weights and biases into a single 1D numpy vector.
        Format: [layer0_weights.ravel(), layer0_biases, layer1_weights.ravel(), layer1_biases, ... , log_std(if learnable)]
        """
        parts = []
        shapes = []
        for l in self.layers:
            w = l.weights
            b = l.biases
            shapes.append(('w', w.shape))
            parts.append(w.ravel())
            shapes.append(('b', b.shape))
            parts.append(b.ravel())
        if self.learn_log_std:
            shapes.append(('logstd', self.log_std.shape))
            parts.append(self.log_std.ravel())
        vec = np.concatenate(parts).astype(float)
        return vec

    def set_params_vector(self, vec):
        """
        Unpack vector into layer weights/biases and optional log_std.
        """
        idx = 0
        for l in self.layers:
            n_w = l.in_dim * l.out_dim
            w_flat = vec[idx: idx + n_w]; idx += n_w
            l.weights = w_flat.reshape((l.in_dim, l.out_dim)).copy()

            n_b = l.out_dim
            b_flat = vec[idx: idx + n_b]; idx += n_b
            l.biases = b_flat.copy()

        if self.learn_log_std:
            out_dim = self.layers[-1].out_dim
            n_ls = out_dim
            self.log_std = vec[idx: idx + n_ls].copy()
            idx += n_ls

        if idx != len(vec):
            raise ValueError("Parameter vector size mismatch when setting params")

    def perturb_params_vector(self, std, seed=None):
        """
        Return a new vector = current_params + normal_noise(0, std)
        Does not change the network itself.
        """
        vec = self.get_params_vector()
        if seed is not None:
            rng = np.random.RandomState(seed)
            noise = rng.randn(*vec.shape) * std
        else:
            noise = np.random.randn(*vec.shape) * std
        return vec + noise

    def add_noise_to_params(self, std, seed=None):
        """
        Add Gaussian noise in place to current params.
        """
        new_vec = self.perturb_params_vector(std, seed)
        self.set_params_vector(new_vec)

    # ----------------------------
    # Misc helpers
    # ----------------------------
    def copy(self):
        return copy.deepcopy(self)

    def zero_grad_state(self):
        # resets Adam state (rarely needed)
        self._opt_state = {'m_w': [np.zeros_like(l.weights) for l in self.layers],
                           'v_w': [np.zeros_like(l.weights) for l in self.layers],
                           'm_b': [np.zeros_like(l.biases) for l in self.layers],
                           'v_b': [np.zeros_like(l.biases) for l in self.layers],
                           't': 0}
        if self.learn_log_std:
            self._opt_state.update({'m_logstd': np.zeros_like(self.log_std),
                                    'v_logstd': np.zeros_like(self.log_std)})

# ----------------------------
# End of neural network module
# ----------------------------
