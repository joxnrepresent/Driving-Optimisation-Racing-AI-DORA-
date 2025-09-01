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
        if self.activation_name == 'relu':
            sigma = np.sqrt(2/self.input_size)
            return np.random.randn(self.input_size, self.output_size) * sigma
        else:
            limit = np.sqrt(6/(self.input_size +self.output_size))
            return np.random.uniform(-limit, limit, (self.input_size, self.output_size))

    def activation(self, z):
        if self.activation_name == 'relu':
            return np.maximum(0, z)
        elif self.activation_name == 'linear':
            return z

    def activation_derivative(self, z):
        if self.activation_name == 'relu':
            return (z>0).astype(float)
        elif self.activation_name == 'linear':
            return np.ones_like(z)


    def layer_forward_pass(self, x):
        z = x.dot(self.weights) + self.biases
        a = self.activation(z)

        self.last_x = x
        self.last_z = z
        self.last_a = a

        return a

    def layer_backward_pass(self, d_a):
        batch_size = d_a.shape[0]                           #d_a is dL/d_a
        d_z =  d_a * self.activation_derivative(self.last_z) #dL/d_z = dL/d_a * d_a/d_z
        d_W = self.last_x.T.dot(d_z) / batch_size                      #dL/d_W = x.T * dL/d_z
        """x.t has shape (input_size, batch_size)
        d_z has shape (batch_size, output_size) 
        So x.T.dot(d_z) will have shape (input_size, output_size), same as weights"""
        d_b = d_z.mean(axis=0)                               #dL/d_b = mean of dL/d_z
        d_x = d_z.dot(self.weights.T)                        #dL/d_x = dL/d_z * W.t

        return d_W, d_b, d_x

    def get_params(self):
        return self.weights.copy(), self.biases.copy()

    def set_params(self, params):
        self.weights = params[0].copy()
        self.biases = params[1].copy()


class NeuralNetwork:
    def __init__(self, layer_sizes, activations, init_log_std=-1.0, seed=None):
        if seed is not None:
            np.random.seed(seed)

        self.layers = []
        assert len(activations) == len(layer_sizes)-1
        for i in range(len(layer_sizes)-1):
            self.layers.append(Layer(layer_sizes[i], layer_sizes[i+1], activations[i]))

        self.log_std = np.full(layer_sizes[-1], init_log_std)

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

    def forward_propagation(self, a):
        # a -> output to next layer
        activation_outputs = [a]
        for layer in self.layers:
            a = layer.layer_forward_pass(a)
            activation_outputs.append(a)
        return a, activation_outputs

    def computing_log_probabilities(self, actions, mu):
        sigma = np.exp(self.log_std)
        return -0.5 * np.sum(((actions-mu) ** 2) / sigma ** 2 + 2 * self.log_std + np.log(2 * np.pi), axis=1)

    def compute_gradients(self, d_a):
        weight_grads = []
        bias_grads = []
        for layer in reversed(self.layers):
            d_W, d_b, d_a = layer.layer_backward_pass(d_a)
            weight_grads.insert(0, d_W)
            bias_grads.insert(0, d_b)

    def update_parameters(self, d_log_std):
        # Adam optimisation
        # w_moments, b_moments, log_stds_moments, t = self.adam_moments.values()
        self.adam_moments['t'] += 1
        for i, layer in enumerate(self.layers):
            self.adam_moments['w'][i], weight_update = self.adam_estimation(self.adam_moments['w'][i][0],
                                                                            self.adam_moments['w'][i][1],
                                                                            weight_grads[i],
                                                                            self.adam_moments['t'])

            self.adam_moments['b'][i], bias_update = self.adam_estimation(self.adam_moments['b'][i][0],
                                                                          self.adam_moments['b'][i][1],
                                                                          bias_grads[i],
                                                                          self.adam_moments['t'])
            layer.weights -= weight_update
            layer.biases -= bias_update

        self.adam_moments['log_std'], log_std_update = self.adam_estimation(self.adam_moments['log_std'][0],
                                                                            self.adam_moments['log_std'][1],
                                                                            d_log_std,
                                                                            self.adam_moments['t'])
        self.log_std -= log_std_update
        
    @staticmethod
    def adam_estimation(m, v, grad, t, alpha = 1e-3, beta1 = 0.9, beta2 = 0.999, eps = 1e-8):
        m = beta1 * m + (1 - beta1) * grad
        v = beta2 * v + (1 - beta2) * grad * grad
        m_hat = m / (1 - beta1 ** t)
        v_hat = v / (1 - beta2 ** t)
        update_value = alpha * m_hat / (np.sqrt(v_hat) + eps)
        return (m, v), update_value