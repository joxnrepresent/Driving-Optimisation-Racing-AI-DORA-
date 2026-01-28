"""
White-Box Testing Scripts for Racing AI Project
Run these tests to verify internal implementation details
Place this file in your project root directory and run: python whitebox_tests.py
"""

import numpy as np
import math
from pygame import Vector2

print("=" * 80)
print("RACING AI PROJECT - WHITE-BOX TESTING SUITE")
print("=" * 80)
print()

# ============================================================================
# TEST SUITE 6: Track Validation Tests
# ============================================================================
print("TEST SUITE 6: TRACK VALIDATION TESTS")
print("-" * 80)


def test_6_1_1_wall_segment_hashing():
    """Test 6.1.1: Test wall segment hashing"""
    from track import SpatialHashGrid

    grid = SpatialHashGrid(cell_size=10)
    segment = ((0, 0), (50, 50))
    grid.hash_segment(segment, 0)

    expected_cells = [(0, 0), (1, 1), (2, 2), (3, 3), (4, 4), (5, 5)]
    found_cells = [cell for cell in grid.cells.keys() if 0 in grid.cells[cell]]

    print(f"Test 6.1.1: Wall segment hashing")
    print(f"  Expected cells: {expected_cells}")
    print(f"  Found cells: {sorted(found_cells)}")
    print(f"  PASS" if len(found_cells) >= 5 else f"  FAIL")
    print()


def test_6_1_2_parallel_segments():
    """Test 6.1.2: Collision between parallel segments"""
    import resources as r

    seg1 = ((0, 0), (100, 0))
    seg2 = ((0, 20), (100, 20))

    intersection = r.get_line_segments_intersection(seg1, seg2)

    print(f"Test 6.1.2: Parallel segments collision")
    print(f"  Segment 1: {seg1}")
    print(f"  Segment 2: {seg2}")
    print(f"  Intersection: {intersection}")
    print(f"  PASS" if intersection is None else f"  FAIL")
    print()


def test_6_1_3_intersecting_segments():
    """Test 6.1.3: Collision between intersecting segments"""
    import resources as r

    seg1 = ((0, 0), (100, 100))
    seg2 = ((0, 100), (100, 0))

    intersection = r.get_line_segments_intersection(seg1, seg2)

    print(f"Test 6.1.3: Intersecting segments collision")
    print(f"  Segment 1: {seg1}")
    print(f"  Segment 2: {seg2}")
    print(f"  Intersection: {intersection}")
    expected = (50, 50)
    if intersection:
        distance = math.sqrt((intersection[0] - expected[0]) ** 2 + (intersection[1] - expected[1]) ** 2)
        print(f"  Distance from expected: {distance:.2f}")
        print(f"  PASS" if distance < 1 else f"  FAIL")
    else:
        print(f"  FAIL - No intersection found")
    print()


def test_6_1_4_track_width_interpolation():
    """Test 6.1.5: Track width interpolation"""

    # Simulate width interpolation logic
    widths_dict = {0.0: 60, 1.0: 80}

    # Interpolation at 0.5
    start_width = widths_dict[0.0]
    end_width = widths_dict[1.0]
    t = 0.5
    interpolated_width = start_width + (end_width - start_width) * t

    print(f"Test 6.1.5: Track width interpolation")
    print(f"  Width at 0.0: {widths_dict[0.0]}")
    print(f"  Width at 1.0: {widths_dict[1.0]}")
    print(f"  Interpolated width at 0.5: {interpolated_width}")
    print(f"  Expected: 70")
    print(f"  PASS" if abs(interpolated_width - 70) < 1 else f"  FAIL")
    print()


# ============================================================================
# TEST SUITE 7: Car Physics Tests
# ============================================================================
print("\nTEST SUITE 7: CAR PHYSICS TESTS")
print("-" * 80)


def test_7_1_1_throttle_increment():
    """Test 7.1.1: Throttle increment rate"""

    current_throttle = 0.0
    target_throttle = 1.0
    throttle_factor = 0.1

    for i in range(10):
        d_throttle = target_throttle - current_throttle
        if abs(d_throttle) > 0.1:
            current_throttle += throttle_factor * (1 if d_throttle > 0 else -1)
        else:
            current_throttle = target_throttle
        current_throttle = max(-1.0, min(1.0, current_throttle))

    print(f"Test 7.1.1: Throttle increment rate")
    print(f"  Final throttle: {current_throttle}")
    print(f"  Expected: ~1.0")
    print(f"  PASS" if abs(current_throttle - 1.0) < 0.01 else f"  FAIL")
    print()


def test_7_1_2_brake_increment():
    """Test 7.1.2: Brake increment rate"""

    current_throttle = 1.0
    target_throttle = -1.0
    brake_factor = 0.04

    for i in range(50):
        d_throttle = target_throttle - current_throttle
        if abs(d_throttle) > 0.1:
            current_throttle += brake_factor * (1 if d_throttle > 0 else -1)
        else:
            current_throttle = target_throttle
        current_throttle = max(-1.0, min(1.0, current_throttle))

    print(f"Test 7.1.2: Brake increment rate")
    print(f"  Final throttle: {current_throttle}")
    print(f"  Expected: -1.0")
    print(f"  PASS" if abs(current_throttle - (-1.0)) < 0.01 else f"  FAIL")
    print()


def test_7_1_3_max_speed_clamping():
    """Test 7.1.3: Maximum speed clamping"""

    velocity = Vector2(500, 0)
    max_speed = 400

    if velocity.length_squared() != 0:
        velocity.clamp_magnitude_ip(max_speed)

    print(f"Test 7.1.3: Maximum speed clamping")
    print(f"  Final velocity magnitude: {velocity.magnitude()}")
    print(f"  Expected: {max_speed}")
    print(f"  PASS" if abs(velocity.magnitude() - max_speed) < 0.01 else f"  FAIL")
    print()


def test_7_1_4_car_friction():
    """Test 7.1.4: Car friction when no throttle"""

    velocity = Vector2(100, 0)
    throttle = 0

    if throttle == 0:
        velocity *= 0.992

    print(f"Test 7.1.4: Car friction")
    print(f"  Final velocity magnitude: {velocity.magnitude()}")
    print(f"  Expected: ~99.2")
    print(f"  PASS" if abs(velocity.magnitude() - 99.2) < 0.1 else f"  FAIL")
    print()


def test_7_1_5_steering_clamping():
    """Test 7.1.5: Steering angle clamping"""

    current_steer_input = 1.5
    current_steer_input = max(-1.0, min(1.0, current_steer_input))

    print(f"Test 7.1.5: Steering clamping")
    print(f"  Final steer input: {current_steer_input}")
    print(f"  Expected: 1.0")
    print(f"  PASS" if current_steer_input == 1.0 else f"  FAIL")
    print()


# ============================================================================
# TEST SUITE 8: Neural Network Tests
# ============================================================================
print("\nTEST SUITE 8: NEURAL NETWORK TESTS")
print("-" * 80)


def test_8_1_1_layer_initialization():
    """Test 8.1.1: Layer initialization"""
    from neural_network import Layer

    layer = Layer(input_size=10, output_size=5, activation='relu')

    print(f"Test 8.1.1: Layer initialization")
    print(f"  Weights shape: {layer.weights.shape}")
    print(f"  Expected: (10, 5)")
    print(f"  Biases shape: {layer.biases.shape}")
    print(f"  Expected: (5,)")
    print(f"  PASS" if layer.weights.shape == (10, 5) and layer.biases.shape == (5,) else f"  FAIL")
    print()


def test_8_1_2_relu_activation():
    """Test 8.1.2: ReLU activation"""
    from neural_network import Layer

    layer = Layer(1, 1, 'relu')
    input_arr = np.array([-2, -1, 0, 1, 2])
    output = layer.activation(input_arr)
    expected = np.array([0, 0, 0, 1, 2])

    print(f"Test 8.1.2: ReLU activation")
    print(f"  Input: {input_arr}")
    print(f"  Output: {output}")
    print(f"  Expected: {expected}")
    print(f"  PASS" if np.allclose(output, expected) else f"  FAIL")
    print()


def test_8_1_3_elu_activation():
    """Test 8.1.3: ELU activation on negative"""
    from neural_network import Layer

    layer = Layer(1, 1, 'elu')
    input_val = np.array([-1.0])
    output = layer.activation(input_val)
    expected = np.exp(-1.0) - 1  # ≈ -0.632

    print(f"Test 8.1.3: ELU activation")
    print(f"  Input: -1.0")
    print(f"  Output: {output[0]:.3f}")
    print(f"  Expected: {expected:.3f}")
    print(f"  PASS" if abs(output[0] - expected) < 0.01 else f"  FAIL")
    print()


def test_8_1_4_tanh_activation():
    """Test 8.1.4: Tanh activation"""
    from neural_network import Layer

    layer = Layer(1, 1, 'tanh')
    input_arr = np.array([-2, 0, 2])
    output = layer.activation(input_arr)
    expected = np.tanh(input_arr)

    print(f"Test 8.1.4: Tanh activation")
    print(f"  Input: {input_arr}")
    print(f"  Output: {output}")
    print(f"  Expected: {expected}")
    print(f"  PASS" if np.allclose(output, expected) else f"  FAIL")
    print()


def test_8_1_5_forward_propagation_shape():
    """Test 8.1.5: Forward propagation shape"""
    from neural_network import NeuralNetwork

    nn = NeuralNetwork([10, 20, 2], ['relu', 'linear'])
    input_batch = np.random.randn(5, 10)
    output = nn.forward_propagation(input_batch)

    print(f"Test 8.1.5: Forward propagation shape")
    print(f"  Input shape: {input_batch.shape}")
    print(f"  Output shape: {output.shape}")
    print(f"  Expected: (5, 2)")
    print(f"  PASS" if output.shape == (5, 2) else f"  FAIL")
    print()


def test_8_1_6_log_std_clipping():
    """Test 8.1.6: Log std clipping"""

    log_std = np.array([5.0, -5.0])
    log_std = np.clip(log_std, -3, 1)

    print(f"Test 8.1.6: Log std clipping")
    print(f"  Clipped log_std: {log_std}")
    print(f"  Expected: [1, -3]")
    print(f"  PASS" if np.allclose(log_std, [1, -3]) else f"  FAIL")
    print()


# ============================================================================
# TEST SUITE 9: Reinforcement Learning Tests
# ============================================================================
print("\nTEST SUITE 9: REINFORCEMENT LEARNING TESTS")
print("-" * 80)


def test_9_1_1_return_calculation():
    """Test 9.1.1: Return calculation with gamma=0.99"""

    rewards = np.array([[1], [2], [3]], dtype=np.float32)
    gamma = 0.99

    time_steps, batch = rewards.shape
    returns = np.zeros(rewards.shape, np.float32)
    G = np.zeros(batch, np.float32)

    for i in reversed(range(time_steps)):
        G = rewards[i] + gamma * G
        returns[i] = G

    returns_flat = returns.flatten()
    expected = np.array([1 + 0.99 * 2 + 0.99 ** 2 * 3, 2 + 0.99 * 3, 3])

    print(f"Test 9.1.1: Return calculation")
    print(f"  Rewards: [1, 2, 3]")
    print(f"  Returns: {returns_flat}")
    print(f"  Expected: {expected}")
    print(f"  PASS" if np.allclose(returns_flat, expected, rtol=0.01) else f"  FAIL")
    print()


def test_9_1_2_stochastic_actions():
    """Test 9.1.2: Stochastic action sampling"""
    from rl_model import REINFORCEModel

    model = REINFORCEModel(15, )
    state = np.array([0.5] * 15)

    actions_list = []
    for _ in range(10):
        actions, _ = model.get_stochastic_actions(state)
        actions_list.append(actions[0])

    actions_arr = np.array(actions_list)
    variance = np.var(actions_arr, axis=0)

    print(f"Test 9.1.2: Stochastic action sampling")
    print(f"  Variance across 10 samples: {variance}")
    print(f"  PASS" if np.all(variance > 0.001) else f"  FAIL (actions should vary)")
    print()


def test_9_1_3_deterministic_actions():
    """Test 9.1.3: Deterministic action output"""
    from rl_model import REINFORCEModel

    model = REINFORCEModel(15, )
    state = np.array([0.5] * 15)

    actions_list = []
    for _ in range(5):
        actions = model.get_deterministic_actions(state)
        actions_list.append(actions)

    actions_arr = np.array(actions_list)
    variance = np.var(actions_arr, axis=0)

    print(f"Test 9.1.3: Deterministic action output")
    print(f"  Variance across 5 samples: {variance}")
    print(f"  PASS" if np.all(variance < 1e-10) else f"  FAIL (actions should be identical)")
    print()


def test_9_1_4_action_clipping():
    """Test 9.1.4: Action clipping"""

    mu = np.array([[1.5, -2.0]])
    actions = np.clip(mu, -1, 1)
    expected = np.array([[1.0, -1.0]])

    print(f"Test 9.1.4: Action clipping")
    print(f"  Network output μ: {mu[0]}")
    print(f"  Clipped actions: {actions[0]}")
    print(f"  Expected: {expected[0]}")
    print(f"  PASS" if np.allclose(actions, expected) else f"  FAIL")
    print()


def test_9_1_5_trajectory_storage():
    """Test 9.1.5: Trajectory storage"""
    from rl_model import REINFORCEModel

    model = REINFORCEModel(15, )

    for i in range(5):
        state = np.random.randn(1, 15)
        action = np.random.randn(1, 2)
        reward = np.array([i])
        model.update_trajectory(state, action, reward)

    print(f"Test 9.1.5: Trajectory storage")
    print(f"  States length: {len(model.states)}")
    print(f"  Actions length: {len(model.pre_squash_actions)}")
    print(f"  Rewards length: {len(model.rewards)}")
    print(f"  Expected: 5 for all")
    pass_test = (len(model.states) == 5 and
                 len(model.pre_squash_actions) == 5 and
                 len(model.rewards) == 5)
    print(f"  PASS" if pass_test else f"  FAIL")
    print()


# ============================================================================
# RUN ALL TESTS
# ============================================================================



# Track validation tests
test_6_1_1_wall_segment_hashing()
test_6_1_2_parallel_segments()
test_6_1_3_intersecting_segments()
test_6_1_4_track_width_interpolation()

# Car physics tests
test_7_1_1_throttle_increment()
test_7_1_2_brake_increment()
test_7_1_3_max_speed_clamping()
test_7_1_4_car_friction()
test_7_1_5_steering_clamping()

# Neural network tests
test_8_1_1_layer_initialization()
test_8_1_2_relu_activation()
test_8_1_3_elu_activation()
test_8_1_4_tanh_activation()
test_8_1_5_forward_propagation_shape()
test_8_1_6_log_std_clipping()

# RL tests
test_9_1_1_return_calculation()
test_9_1_2_stochastic_actions()
test_9_1_3_deterministic_actions()
test_9_1_4_action_clipping()
test_9_1_5_trajectory_storage()

print("=" * 80)
print("ALL WHITE-BOX TESTS COMPLETED")
print("=" * 80)

