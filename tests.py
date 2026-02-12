from neural_network import  NeuralNetwork
import numpy as np
from numpy import e
import matplotlib
matplotlib.use("TkAgg")

import matplotlib.pyplot as plt
import matplotlib.cm as cm


def sin_x(x):
    return np.sin(x)

def trig_f_1(x):
    return np.sin(x)+0.2*np.sin(5*x)

def goldilocks(x):
    return np.sin(x) + 0.1* x* np.sin(2*x)

def fourier_hard(x):
    return 2 * (
        np.sin(x)
        + 0.5*np.sin(3*x)
        + 0.25*np.sin(7*x)
        + 0.125*np.sin(15*x)
    )

def fourier_modulated(x):
    return (
        np.sin(x)
        * (1 + 0.3*np.sin(3*x) + 0.2*np.sin(7*x))
    )
def hierarchical_fourier(x):
    return (
        np.sin(x)
        * (1 + 0.3*np.sin(3*x))                 # coarse modulation
        + 0.15*np.sin(7*x) * np.sin(0.5*x)     # medium detail
        + 0.05*np.sin(15*x) * np.sin(0.2*x)    # fine detail
    )
test_layer_configs = [[1, 4, 1],
                      [1, 32, 1],
                      [1, 4, 4, 4, 1],
                      [1, 8, 16, 8, 1],
                      [1,16,16,16,1],
                      [1, 16, 32, 32, 16, 1]]

activation_options = ["relu", "elu", "tanh"]

for activation in activation_options:
    for layer_config in test_layer_configs:
        neural_net = NeuralNetwork(
            layer_sizes=layer_config,
            activations=[activation] * (len(layer_config)-2) + ["linear"],
            init_log_std=0.0,
            alpha=0.01,
            beta1=0.9,
            beta2=0.999,
            seed=42
        )

        epochs_list = [10, 1000, 10000, 50000]
        max_epochs = max(epochs_list)

        batch_size = 10
        x_min, x_max = -20, 20

        saved_predictions = {}

        for epoch in range(1, max_epochs + 1):

            # Sample random inputs
            x = np.random.uniform(x_min, x_max, (batch_size, 1))
            y_true = fourier_hard(x)

            # Forward pass
            y_pred = neural_net.forward_propagation(x)

            # MSE loss
            loss = np.mean((y_pred - y_true) ** 2)

            # Gradient of MSE wrt output
            d_mu = (2 / batch_size) * (y_pred - y_true)

            # Backprop + Adam update
            neural_net.update_params(d_mu)

            # Save predictions at checkpoints
            if epoch in epochs_list:
                x_plot = np.linspace(x_min, x_max, 400).reshape(-1, 1)
                y_plot = neural_net.forward_propagation(x_plot)
                saved_predictions[epoch] = (x_plot.copy(), y_plot.copy())

            if epoch % 100 == 0:
                print(f"Epoch {epoch}, Loss = {loss:.6f}")


        x_true = np.linspace(x_min, x_max, 400)
        y_true = fourier_hard(x_true)
        plt.figure(figsize=(10, 6))
        plt.plot(
            x_true, y_true,
            linestyle="-.",
            color="grey",
            linewidth=2,
            label="f(x)"
        )

        colors = cm.viridis(np.linspace(0, 1, len(saved_predictions)))
        for i, (epoch, (x_pred, y_pred)) in enumerate(saved_predictions.items()):
            plt.plot(
                x_pred.flatten(),
                y_pred.flatten(),
                marker='o',
                markersize=3,
                linewidth=1.2,
                alpha=0.7,
                color=colors[i],
                label=f"NN @ {epoch} epochs",
                markevery=3
            )

        plt.legend()
        plt.xlabel("x")
        plt.ylabel("y")
        plt.title(f"Architecture: {layer_config}")
        plt.grid(True)
        plt.ylim(-5, 5)
        plt.savefig(f"Neural Network Tests/{activation}, {layer_config}.png")


# grid = SpatialHashGrid()
#
# walls = [
#     ((0, 0), (200, 0)),
#     ((0, 0), (0, 200)),
#     ((0, 0), (200, 200)),
#     ((230, 20), (120, 175)),
# ]
#
# for i, seg in enumerate(walls):
#     grid.hash_segment(seg, i)
#
# print()
# print("Testing ray intersection")
# print()
#
# print("Test 1: dx > dy")
# ray1 = ((100, 50), (250, 100))
# hit1 = grid.return_collision_point(ray1, walls)
# print("Collision point:", hit1)
# print()
#
# print("Test 2: dy > dx")
# ray2 = ((100, 20), (130, 250))
# hit2 = grid.return_collision_point(ray2, walls)
# print("Collision point:", hit2)
# print()
#
# print("Test 3: no collision")
# ray3 = ((250, 250), (400, 400))
# hit3 = grid.return_collision_point(ray3, walls)
# print(" Collision point:", hit3)
