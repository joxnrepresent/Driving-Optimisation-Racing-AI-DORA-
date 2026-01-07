import  numpy as np

# mydict = {1:[0,1,2,3], 5:[6,7,8,9]}
# mydict[90] = "alphabet"
# if mydict.__contains__('c'):
#     print(True)
# else:
#     print(False)
#
# mydict.pop(90)
# print(mydict)
my_list = [1,2,3,5,6]
my_list = my_list[3:]
mean = np.mean(my_list)
print(max(my_list))

print(65.3436 % 1)





















def compute_reward(self, prev_progress, max_progress, sensors, actions):
    reward = 0
    rewards_breakdown = []

    # Crash penalty
    if self.is_crashed:
        rewards_breakdown = [-60, 0, 0 ,0 ,0 ,0 ,0,0, 0]
        return -60, rewards_breakdown
    else:
        rewards_breakdown.append(0)
    # --------------------------------------------------- #

    # Progress reward
    distance_moved = self.progress - prev_progress
    if distance_moved < -0.5:
        distance_moved += 1.0
    if distance_moved > 1e-5:
        rewards_breakdown.append(distance_moved * 400)
        reward += distance_moved * 400
    else:
        rewards_breakdown.append(-0.02)
        reward-= 0.02
    # --------------------------------------------------- #

    # Imbalance reward
    left_sensors = [sensors[2], sensors[4], sensors[6], sensors[8], sensors[10]]
    avg_left = sum(left_sensors) / 5
    right_sensors = [sensors[1], sensors[3], sensors[5], sensors[7], sensors[9]]
    avg_right = sum(right_sensors) / 5
    imbalance = abs(avg_left - avg_right)
    reward -= imbalance * 0.9
    rewards_breakdown.append(-imbalance * 0.9)
    # --------------------------------------------------- #

    # Speed reward
    speed = self.velocity.magnitude()
    normalized_speed = speed / self.max_speed
    if sensors[0] > 0.04:
        rewards_breakdown.append(normalized_speed * 0.2)
        reward += normalized_speed * 0.2
    else:
        reward -= normalized_speed * 1
        rewards_breakdown.append(-normalized_speed * 1)
    # --------------------------------------------------- #

    # Max progress reward/penalty
    progress_change = self.progress - max_progress
    reward += 0 * progress_change
    rewards_breakdown.append(0 * progress_change)
    # --------------------------------------------------- #

    # Wall hugging penalty
    min_sensor = min(sensors) if sensors else 0
    if min_sensor < 0.03:
        reward -= (0.03 - min_sensor) * 70.0
        rewards_breakdown.append(-(0.03 - min_sensor) * 70.0)
    else:
        rewards_breakdown.append(0)
    # --------------------------------------------------- #
    steer, throttle = actions

    # Jittery steering penalty
    steer_change = abs(self.current_steer_input - steer)
    reward -= steer_change * 0.015
    rewards_breakdown.append(-steer_change * 0.015)

    # --------------------------------------------------- #
    # Jittery acceleration penalty
    throttle_change = abs(self.current_throttle_input - throttle)
    reward -= throttle_change * 0.015
    rewards_breakdown.append(-throttle_change * 0.015)
    # --------------------------------------------------- #

    # Time penalty
    reward += 0.005
    rewards_breakdown.append(0.05)
    # --------------------------------------------------- #

    return reward, rewards_breakdown