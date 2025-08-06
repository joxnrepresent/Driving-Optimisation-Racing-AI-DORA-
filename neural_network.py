import numpy

class Neural_Network:
    def __init__(self, learning_rate, discount_factor, standard_deviation, episode_size):
        self.eta = learning_rate
        self.gamma = discount_factor
        self.sigma = standard_deviation
        self.T = episode_size

        self.tau = []       #Trajectory
        self.L = []         #Loss


class Layer:
    def __init__(self, activation, learning_rate):
        self.phi = activation
        self.eta = learning_rate

        self.x = []
        self.y = []
        self.W = []
        self.b = []

class Output_Layer(Layer):
