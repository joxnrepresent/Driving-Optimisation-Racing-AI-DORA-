import numpy as np
import matplotlib
from nea.neural_network import NeuralNetwork
matplotlib.use("TkAgg")
import matplotlib.pyplot as plt
import matplotlib.cm as cm


# Import your NeuralNetwork class here
# from neural_network import NeuralNetwork, Layer


# ============================================================================
# TEST FUNCTIONS
# ============================================================================

def test_9_2_network_shape(layer_config, activation):
    """
    Test 9.2: Network shape test
    Validates that network structure matches configuration
    """
    print(f"\n{'=' * 60}")
    print(f"TEST 9.2: Network Shape Test")
    print(f"Configuration: {layer_config}, Activation: {activation}")
    print(f"{'=' * 60}")

    neural_net = NeuralNetwork(
        layer_sizes=layer_config,
        activations=[activation] * (len(layer_config) - 2) + ["linear"],
        init_log_std=0.0,
        alpha=0.01,
        beta1=0.9,
        beta2=0.999,
        seed=42
    )

    # BREAKPOINT HERE - inspect neural_net.layers
    print(f"\nNumber of layers: {len(neural_net.layers)}")

    all_valid = True
    for i, layer in enumerate(neural_net.layers):
        expected_input = layer_config[i]
        expected_output = layer_config[i + 1]
        actual_weight_shape = layer.weights.shape
        actual_bias_shape = layer.biases.shape

        print(f"\nLayer {i}:")
        print(f"  Expected: input={expected_input}, output={expected_output}")
        print(f"  Weights shape: {actual_weight_shape} (expected: ({expected_input}, {expected_output}))")
        print(f"  Biases shape: {actual_bias_shape} (expected: ({expected_output},))")
        print(f"  Activation: {layer.activation_name}")

        # Check shapes
        weight_shape_valid = actual_weight_shape == (expected_input, expected_output)
        bias_shape_valid = actual_bias_shape == (expected_output,)

        # Check for numerical values (no NaN or Inf)
        weights_numerical = not (np.isnan(layer.weights).any() or np.isinf(layer.weights).any())
        biases_numerical = not (np.isnan(layer.biases).any() or np.isinf(layer.biases).any())

        print(f"  Weight shape valid: {weight_shape_valid}")
        print(f"  Bias shape valid: {bias_shape_valid}")
        print(f"  Weights numerical (no NaN/Inf): {weights_numerical}")
        print(f"  Biases numerical (no NaN/Inf): {biases_numerical}")

        if not (weight_shape_valid and bias_shape_valid and weights_numerical and biases_numerical):
            all_valid = False

    print(f"\n{'=' * 60}")
    print(f"TEST 9.2 RESULT: {'PASS' if all_valid else 'FAIL'}")
    print(f"{'=' * 60}\n")

    return neural_net, all_valid


def test_9_3_gradient_flow(neural_net, num_episodes=5):
    """
    Test 9.3: Gradient flow test
    Validates that gradients are calculated correctly and parameters update
    """
    print(f"\n{'=' * 60}")
    print(f"TEST 9.3: Gradient Flow Test")
    print(f"{'=' * 60}")

    batch_size = 10
    x_min, x_max = -10, 10

    # Store initial parameters
    initial_params = neural_net.get_params()

    gradient_history = []
    param_update_history = []

    for episode in range(num_episodes):
        print(f"\nEpisode {episode + 1}/{num_episodes}")

        # Generate batch
        x = np.random.uniform(x_min, x_max, (batch_size, neural_net.layers[0].input_size))
        y_true = np.sin(x)

        # Forward pass
        y_pred = neural_net.forward_propagation(x)

        # MSE loss
        loss = np.mean((y_pred - y_true) ** 2)
        print(f"  Loss: {loss:.6f}")

        # Gradient of MSE wrt output
        d_mu = (2 / batch_size) * (y_pred - y_true)

        # Store params before update
        params_before = neural_net.get_params()

        # Compute gradients (this happens inside update_params)
        weight_grads, bias_grads = neural_net.compute_gradients(d_mu)

        # BREAKPOINT HERE - inspect weight_grads and bias_grads

        # Check gradients for NaN/Inf
        gradients_valid = True
        print(f"  Gradient checks:")
        for i, (w_grad, b_grad) in enumerate(zip(weight_grads, bias_grads)):
            w_has_nan = np.isnan(w_grad).any()
            w_has_inf = np.isinf(w_grad).any()
            b_has_nan = np.isnan(b_grad).any()
            b_has_inf = np.isinf(b_grad).any()

            w_grad_magnitude = np.abs(w_grad).mean()
            b_grad_magnitude = np.abs(b_grad).mean()

            print(f"    Layer {i}: W_grad mean_abs={w_grad_magnitude:.6f}, "
                  f"B_grad mean_abs={b_grad_magnitude:.6f}")
            print(f"             NaN in W_grad: {w_has_nan}, Inf in W_grad: {w_has_inf}")
            print(f"             NaN in B_grad: {b_has_nan}, Inf in B_grad: {b_has_inf}")

            if w_has_nan or w_has_inf or b_has_nan or b_has_inf:
                gradients_valid = False

        gradient_history.append({
            'episode': episode,
            'weight_grads': [g.copy() for g in weight_grads],
            'bias_grads': [g.copy() for g in bias_grads],
            'valid': gradients_valid
        })

        # Update parameters
        neural_net.update_params(d_mu)

        # BREAKPOINT HERE - inspect updated parameters

        # Check if parameters actually changed
        params_after = neural_net.get_params()
        params_changed = False
        print(f"  Parameter update checks:")
        for i in range(len(neural_net.layers)):
            w_change = np.abs(params_after['weights'][i] - params_before['weights'][i]).mean()
            b_change = np.abs(params_after['biases'][i] - params_before['biases'][i]).mean()
            print(f"    Layer {i}: W change mean_abs={w_change:.8f}, B change mean_abs={b_change:.8f}")
            if w_change > 1e-10 or b_change > 1e-10:
                params_changed = True

        param_update_history.append({
            'episode': episode,
            'params_changed': params_changed
        })

    # Final validation
    all_gradients_valid = all(g['valid'] for g in gradient_history)
    all_params_updated = all(p['params_changed'] for p in param_update_history)

    print(f"\n{'=' * 60}")
    print(f"TEST 9.3 SUMMARY:")
    print(f"  All gradients valid (no NaN/Inf): {all_gradients_valid}")
    print(f"  Parameters updated each episode: {all_params_updated}")
    print(f"TEST 9.3 RESULT: {'PASS' if (all_gradients_valid and all_params_updated) else 'FAIL'}")
    print(f"{'=' * 60}\n")

    return gradient_history, all_gradients_valid and all_params_updated


def test_9_4_batch_processing(layer_config, activation, batch_sizes=[1, 5, 12, 50]):
    """
    Test 9.4: Batch processing test
    Validates parallel processing of multiple inputs
    """
    print(f"\n{'=' * 60}")
    print(f"TEST 9.4: Batch Processing Test")
    print(f"Configuration: {layer_config}, Activation: {activation}")
    print(f"{'=' * 60}")

    neural_net = NeuralNetwork(
        layer_sizes=layer_config,
        activations=[activation] * (len(layer_config) - 2) + ["linear"],
        init_log_std=0.0,
        alpha=0.01,
        beta1=0.9,
        beta2=0.999,
        seed=42
    )

    all_valid = True

    for batch_size in batch_sizes:
        print(f"\nTesting batch_size={batch_size}")

        # Generate batch input
        x = np.random.randn(batch_size, layer_config[0])

        # Forward pass
        y_pred = neural_net.forward_propagation(x)

        # BREAKPOINT HERE - inspect y_pred shape

        expected_output_shape = (batch_size, layer_config[-1])
        actual_output_shape = y_pred.shape

        print(f"  Input shape: {x.shape}")
        print(f"  Output shape: {actual_output_shape} (expected: {expected_output_shape})")

        shape_valid = actual_output_shape == expected_output_shape

        # Generate dummy target and compute gradients
        y_true = np.random.randn(batch_size, layer_config[-1])
        d_mu = (2 / batch_size) * (y_pred - y_true)

        weight_grads, bias_grads = neural_net.compute_gradients(d_mu)

        # BREAKPOINT HERE - inspect gradient shapes

        # Check that gradients have correct shapes
        gradients_valid = True
        for i, (w_grad, b_grad) in enumerate(zip(weight_grads, bias_grads)):
            expected_w_shape = (layer_config[i], layer_config[i + 1])
            expected_b_shape = (layer_config[i + 1],)

            w_shape_valid = w_grad.shape == expected_w_shape
            b_shape_valid = b_grad.shape == expected_b_shape

            # Check that gradients are averaged over batch
            # (individual gradient components should not scale with batch size)
            w_grad_magnitude = np.abs(w_grad).max()

            print(f"  Layer {i} gradients:")
            print(f"    W_grad shape: {w_grad.shape} (expected: {expected_w_shape}) - Valid: {w_shape_valid}")
            print(f"    B_grad shape: {b_grad.shape} (expected: {expected_b_shape}) - Valid: {b_shape_valid}")
            print(f"    W_grad max magnitude: {w_grad_magnitude:.6f}")

            if not (w_shape_valid and b_shape_valid):
                gradients_valid = False

        batch_valid = shape_valid and gradients_valid
        print(f"  Batch {batch_size} result: {'PASS' if batch_valid else 'FAIL'}")

        if not batch_valid:
            all_valid = False

    print(f"\n{'=' * 60}")
    print(f"TEST 9.4 RESULT: {'PASS' if all_valid else 'FAIL'}")
    print(f"{'=' * 60}\n")

    return all_valid


def test_activation_functions():
    """
    Additional test: Activation function correctness
    Validates that activation functions compute correct values
    """
    print(f"\n{'=' * 60}")
    print(f"ADDITIONAL TEST: Activation Function Correctness")
    print(f"{'=' * 60}")

    test_input = np.array([-2.0, -1.0, 0.0, 1.0, 2.0])

    activations_to_test = ['relu', 'elu', 'tanh', 'linear']

    all_valid = True

    for activation_name in activations_to_test:
        print(f"\nTesting {activation_name}:")

        # Create a simple network just to access activation function
        net = NeuralNetwork(
            layer_sizes=[1, 1],
            activations=[activation_name],
            init_log_std=0.0,
            alpha=0.01,
            beta1=0.9,
            beta2=0.999
        )

        layer = net.layers[0]
        output = layer.activation(test_input)
        derivative = layer.activation_derivative(test_input)

        print(f"  Input:      {test_input}")
        print(f"  Output:     {output}")
        print(f"  Derivative: {derivative}")

        # Verify known values
        if activation_name == 'relu':
            expected = np.maximum(0, test_input)
            valid = np.allclose(output, expected)
        elif activation_name == 'elu':
            expected = np.where(test_input > 0, test_input, np.exp(test_input) - 1)
            valid = np.allclose(output, expected)
        elif activation_name == 'tanh':
            expected = np.tanh(test_input)
            valid = np.allclose(output, expected)
        elif activation_name == 'linear':
            expected = test_input
            valid = np.allclose(output, expected)

        print(f"  Expected:   {expected}")
        print(f"  Valid:      {valid}")

        if not valid:
            all_valid = False

    print(f"\n{'=' * 60}")
    print(f"ACTIVATION TEST RESULT: {'PASS' if all_valid else 'FAIL'}")
    print(f"{'=' * 60}\n")

    return all_valid


def test_weight_initialisation():
    """
    Additional test: Weight initialisation
    Validates that weights are initialised with correct distributions
    """
    print(f"\n{'=' * 60}")
    print(f"ADDITIONAL TEST: Weight Initialisation")
    print(f"{'=' * 60}")

    layer_config = [15, 64, 64, 2]
    all_valid = True

    for activation in ['relu', 'tanh']:
        print(f"\nTesting {activation} initialisation:")

        net = NeuralNetwork(
            layer_sizes=layer_config,
            activations=[activation] * (len(layer_config) - 1),
            init_log_std=0.0,
            alpha=0.01,
            beta1=0.9,
            beta2=0.999,
            seed=42
        )

        for i, layer in enumerate(net.layers):
            w_mean = layer.weights.mean()
            w_std = layer.weights.std()

            if activation in ['relu', 'elu']:
                # He initialisation: std should be approximately sqrt(2/input_size)
                expected_std = np.sqrt(2 / layer.input_size)
                print(f"  Layer {i} (He init):")
            else:
                # Xavier initialisation: std should be approximately sqrt(2/(input_size + output_size))
                expected_std = np.sqrt(2 / (layer.input_size + layer.output_size))
                print(f"  Layer {i} (Xavier init):")

            print(f"    Mean: {w_mean:.6f} (should be ≈ 0)")
            print(f"    Std:  {w_std:.6f} (expected ≈ {expected_std:.6f})")

            # Check if mean is close to 0 and std is in reasonable range
            mean_valid = abs(w_mean) < 0.1
            std_valid = abs(w_std - expected_std) < expected_std * 0.5  # Within 50% of expected

            print(f"    Valid: {mean_valid and std_valid}")

            if not (mean_valid and std_valid):
                all_valid = False

    print(f"\n{'=' * 60}")
    print(f"INITIALISATION TEST RESULT: {'PASS' if all_valid else 'FAIL'}")
    print(f"{'=' * 60}\n")

    return all_valid


# ============================================================================
# MAIN TEST RUNNER
# ============================================================================

if __name__ == "__main__":
    # Test configurations
    test_layer_configs = [
        [15, 32, 32, 2],  # Typical RL config
        [15, 64, 64, 2],  # Larger RL config
        [1, 4, 1],  # Simple network
        [1, 16, 32, 16, 1],  # Deep network
    ]

    activation_options = ["relu", "elu", "tanh"]

    print("=" * 60)
    print("NEURAL NETWORK WHITE BOX TESTING")
    print("=" * 60)

    # Run Test 9.2 for one configuration
    print("\n" + "=" * 60)
    print("RUNNING TEST 9.2: Network Shape Test")
    print("=" * 60)
    neural_net, test_9_2_pass = test_9_2_network_shape([15, 32, 32, 2], "tanh")

    # Run Test 9.3 using the network from 9.2
    print("\n" + "=" * 60)
    print("RUNNING TEST 9.3: Gradient Flow Test")
    print("=" * 60)
    gradient_history, test_9_3_pass = test_9_3_gradient_flow(neural_net, num_episodes=5)

    # Run Test 9.4 for one configuration
    print("\n" + "=" * 60)
    print("RUNNING TEST 9.4: Batch Processing Test")
    print("=" * 60)
    test_9_4_pass = test_9_4_batch_processing([15, 32, 32, 2], "tanh")

    # Run additional tests
    print("\n" + "=" * 60)
    print("RUNNING ADDITIONAL TESTS")
    print("=" * 60)
    activation_test_pass = test_activation_functions()
    init_test_pass = test_weight_initialisation()

    # Final summary
    print("\n" + "=" * 60)
    print("FINAL TEST SUMMARY")
    print("=" * 60)
    print(f"Test 9.2 (Network Shape):        {'PASS' if test_9_2_pass else 'FAIL'}")
    print(f"Test 9.3 (Gradient Flow):        {'PASS' if test_9_3_pass else 'FAIL'}")
    print(f"Test 9.4 (Batch Processing):     {'PASS' if test_9_4_pass else 'FAIL'}")
    print(f"Additional (Activation Functions): {'PASS' if activation_test_pass else 'FAIL'}")
    print(f"Additional (Weight Initialisation): {'PASS' if init_test_pass else 'FAIL'}")
    print("=" * 60)

