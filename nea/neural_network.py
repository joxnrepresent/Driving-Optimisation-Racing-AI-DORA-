import numpy as np
"""
Neural Network and neural network Layer objects

Layer objects handle forward and backward propagation using the weights and biases. Operations are handled as layers of
neurons instead of individual neurons to enable optimised matrix operations for computational speedup

Multi-layer perceptron neural network object that feedforwards input data thorugh array of layers to produce output
log_std parameter of neural network is updated as training progresses
Neural network implements Adam optimiser for updating weights, biases and log_std. 

Classes:
    Layer: Layer of neurons 
    NeuralNetwork: Array of layers with additional methods for updates
"""
class Layer:
    """
    Layer of neurons. Handles operations for forward and backward pass

    Attributes:
        input_size: size of input vector
        output_size: size of output vector (input to next layer)
        biases (np.array(output_size)): Array of biases
        weights (np.array(input_size, output_size)): Matrix of weights
        last_x, last_z, last_a (np.array(input_size, output_size)): cached values for gradient computation
    """

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
        """
        Apply random intialisation type depending on the layer activation function

        Returns:
            np.array(input_size, output_size): randomised weights
        """
        if self.activation_name in ['relu', 'elu']:
            # He initialisation
            sigma = np.sqrt(2/self.input_size)
            return np.random.randn(self.input_size, self.output_size) * sigma
        else:
            # Xavier initialisation
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
        """
        Compute output vector from the input by linear operations followed by non-linear activation
        Args:
            x (np.array(input_size, output_size)): input vector
        Returns:
            np.array(output_size, output_size): output vector
        """
        z = x.dot(self.weights) + self.biases
        a = self.activation(z)
        self.last_x = x
        self.last_z = z
        self.last_a = a
        return a

    def layer_backward_pass(self, d_a):
        """
        Computes gradients of the output gradient with respect to weight and bias parameters

        Args:
            d_a (np.array(output_size, output_size)): gradient of output vector

        Returns:
            d_w (np.array(output_size, output_size)): gradient w.r.t weights
            d_b (np.array(output_size, output_size)): gradient w.r.t biases
            d_x (np.array(output_size, output_size)): gradient w.r.t input vector (output vector of previous layer)
        """
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
    """
    MLP (multilayer perceptron) neural network consisting of array of layers.
    Includes trainable log_std and non-trainable hyperparameters

    Attributes:
        layers (list(Layer)): array of layers
        log_std (float): log standard deviation (trainable)
        alpha (float): learning rate (non-trainable)
        beta1 (float): first moment coefficient (non-trainable)
        beta2 (float): second moment coefficient (non-trainable)
        adam_moments (dict): cached adam moments
    """
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
        self.adam_moments['t'] = 0

    def forward_propagation(self, a):
        """
        Forward propagate input vector through network to produce output
        'a' -> output of 1 layer and input to next layer

        Args:
            a (np.array): input vector
        Returns:
            np.array: output vector
        """
        for layer in self.layers:
            a = layer.layer_forward_pass(a)
        return a

    def compute_gradients(self, d_a):
        """
        Backpropagation to compute gradients of advantage w.r.t individual parameters
        'd_a' -> gradient w.r.t input of layer and w.r.t output of previous layer

        Args:
            d_a (np.array): gradient w.r.t network output
        Returns:
            np.array(), np.array(): list of gradients w.r.t weight and biases
        """
        weight_grads = []
        bias_grads = []
        for layer in reversed(self.layers):
            d_W, d_b, d_a = layer.layer_backward_pass(d_a)
            weight_grads.insert(0, d_W)
            bias_grads.insert(0, d_b)
        return weight_grads, bias_grads

    def update_params(self, d_mu, d_log_std = None):
        """
        Update parameters by calculating gradients and with adam optimiser

        Args:
            d_mu (np.array): gradient w.r.t network output
            d_log_std (np.array): gradient w.r.t log standard deviation
        """
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
        """
        Determines update value using adam estimation formula.
        """
        m = beta1 * m + (1 - beta1) * grad
        v = beta2 * v + (1 - beta2) * grad ** 2
        m_hat = m / (1 - beta1 ** t)
        v_hat = v / (1 - beta2 ** t)
        update_value = -alpha * m_hat / (np.sqrt(v_hat) + 1e-8)
        return (m, v), update_value


