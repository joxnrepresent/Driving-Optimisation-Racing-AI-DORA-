import numpy as np
from statistics import variance

class Layer:
    def __init__(self, input_size, output_size, activation):
        self.biases = np.zeros(output_size)
        self.activation_name = activation

        self.input_size = input_size
        self.output_size = output_size

        self.weights = self.initialise_weights()
        self.last_x = None
        self.last_z = None
        self.last_a = None

    def initialise_weights(self):
        if self.activation_name in ['relu', 'elu']:
            sigma = np.sqrt(2/self.input_size)
            return np.random.randn(self.input_size, self.output_size) * sigma
        else:
            limit = np.sqrt(6/(self.input_size +self.output_size))
            return np.random.uniform(-limit, limit, (self.input_size, self.output_size))

    def activation(self, z):
        if self.activation_name == 'relu':
            return np.maximum(0, z)
        if self.activation_name == 'elu':
            return np.where(z > 0, z, np.exp(z) - 1)
        elif self.activation_name == 'linear':
            return z
        elif self.activation_name == 'tanh':
            return np.tanh(z)

    def activation_derivative(self, z):
        if self.activation_name == 'relu':
            return (z>0).astype(float)
        elif self.activation_name == 'elu':
            return np.where(z > 0, 1, np.exp(z))
        elif self.activation_name == 'linear':
            return np.ones_like(z)
        elif self.activation_name == "tanh":
            return 1 - np.tanh(z) ** 2

    def layer_forward_pass(self, x):
        z = x.dot(self.weights) + self.biases
        a = self.activation(z)
        self.last_x = x
        self.last_z = z
        self.last_a = a
        return a

    def layer_backward_pass(self, d_a):
        batch_size = d_a.shape[0]
        d_z = d_a * self.activation_derivative(self.last_z)
        d_W = self.last_x.T.dot(d_z) / batch_size
        """x.t has shape (input_size, batch_size)
        d_z has shape (batch_size, output_size) 
        So x.T.dot(d_z) will have shape (input_size, output_size), same as weights"""
        d_b = d_z.mean(axis=0)
        d_x = d_z.dot(self.weights.T)

        return d_W, d_b, d_x

    def get_params(self):
        return self.weights.copy(), self.biases.copy()

    def set_params(self, params):
        self.weights = params[0].copy()
        self.biases = params[1].copy()


class NeuralNetwork:
    def __init__(self, layer_sizes, activations, init_log_std, alpha, beta1, beta2, seed=None):
        if seed is not None:
            np.random.seed(seed)

        self.layers = []
        assert len(activations) == len(layer_sizes)-1
        for i in range(len(layer_sizes)-1):
            self.layers.append(Layer(layer_sizes[i], layer_sizes[i+1], activations[i]))

        self.log_std = np.full(layer_sizes[-1], init_log_std, dtype=np.float64)
        self.alpha = alpha
        self.beta1 = beta1
        self.beta2 = beta2
        #Adaptive Moment (Adam) Optimiser
        """
        1st moment - mean
        2nd moment - variance
        """
        self.adam_moments = {'w':[], 'b':[],
                             'log_std':(np.zeros_like(self.log_std), np.zeros_like(self.log_std)), 't':0}
        for layer in self.layers:
            self.adam_moments['w'].append((np.zeros_like(layer.weights), np.zeros_like(layer.weights)))
            self.adam_moments['b'].append((np.zeros_like(layer.biases), np.zeros_like(layer.biases)))

    def get_params(self):
        params = {
            'weights': [layer.weights.copy() for layer in self.layers],
            'biases': [layer.biases.copy() for layer in self.layers],
            'log_std': self.log_std.copy(),
            'adam_moments': {
                'w': [(m.copy(), v.copy()) for m, v in self.adam_moments['w']],
                'b': [(m.copy(), v.copy()) for m, v in self.adam_moments['b']],
                'log_std': (self.adam_moments['log_std'][0].copy(),
                            self.adam_moments['log_std'][1].copy()),
                't': self.adam_moments['t']
            }
        }
        return params

    def set_params(self, params):
        for i, layer in enumerate(self.layers):
            layer.weights = params['weights'][i].copy()
            layer.biases = params['biases'][i].copy()

        self.log_std = params['log_std'].copy()
        self.adam_moments['w'] = [(m.copy(), v.copy()) for m, v in params['adam_moments']['w']]
        self.adam_moments['b'] = [(m.copy(), v.copy()) for m, v in params['adam_moments']['b']]
        self.adam_moments['log_std'] = (params['adam_moments']['log_std'][0].copy(),
                                        params['adam_moments']['log_std'][1].copy())
        # self.adam_moments['t'] = params['adam_moments']['t']
        self.adam_moments['t'] = 0

    def forward_propagation(self, a):
        # a -> input to next layer
        for layer in self.layers:
            a = layer.layer_forward_pass(a)
        return a

    def compute_gradients(self, d_mu):
        weight_grads = []
        bias_grads = []
        for layer in reversed(self.layers):
            d_W, d_b, d_mu = layer.layer_backward_pass(d_mu)
            weight_grads.insert(0, d_W)
            bias_grads.insert(0, d_b)
        return weight_grads, bias_grads

    def update_params(self, d_mu, d_log_std = None):
        # Adam optimisation
        # w_moments, b_moments, log_stds_moments, t = self.adam_moments.values()
        weight_grads, bias_grads = self.compute_gradients(d_mu)

        self.adam_moments['t'] += 1
        for i, layer in enumerate(self.layers):
            self.adam_moments['w'][i], weight_update = self.adam_estimation(self.adam_moments['w'][i][0],
                                                                            self.adam_moments['w'][i][1],
                                                                            weight_grads[i],
                                                                            self.adam_moments['t'],
                                                                            self.alpha,
                                                                            self.beta1,
                                                                            self.beta2)

            self.adam_moments['b'][i], bias_update = self.adam_estimation(self.adam_moments['b'][i][0],
                                                                          self.adam_moments['b'][i][1],
                                                                          bias_grads[i],
                                                                          self.adam_moments['t'],
                                                                            self.alpha,
                                                                            self.beta1,
                                                                            self.beta2)
            layer.weights += weight_update
            layer.biases += bias_update

        if d_log_std is not None:
            self.adam_moments['log_std'], log_std_update = self.adam_estimation(self.adam_moments['log_std'][0],
                                                                            self.adam_moments['log_std'][1],
                                                                            d_log_std,
                                                                            self.adam_moments['t'],
                                                                            self.alpha,
                                                                            self.beta1,
                                                                            self.beta2)

        if d_log_std is not None:
            self.log_std += log_std_update
            self.log_std = np.clip(self.log_std, -3, 1)

    @staticmethod
    def adam_estimation(m, v, grad, t, alpha, beta1, beta2):
        m = beta1 * m + (1 - beta1) * grad
        v = beta2 * v + (1 - beta2) * grad * grad
        m_hat = m / (1 - beta1 ** t)
        v_hat = v / (1 - beta2 ** t)
        update_value = -alpha * m_hat / (np.sqrt(v_hat) + 1e-8)
        return (m, v), update_value
