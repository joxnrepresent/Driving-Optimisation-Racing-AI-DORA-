import math

import numpy as np
from pygame import Vector2
from pygame_gui.core.colour_parser import is_float_str

import resources as r
import pygame
from abc import ABC, abstractmethod
from pygame_gui import elements
from resources import game_core
from gui_custom_elements import UIGaugeMeter


class Car(pygame.sprite.Sprite, ABC):
    """
    This class is the Car sprite and extends the Sprite class. The movement of the car is controlled by user input.
    The throttle and steer values are used to control the car's movement. The car's acceleration, velocity and position
    are calculated using vector math and arbitrary constants. Movement is not modelling in terms of forces in this
    version.In later versions, forces will be used to model car movement which would allow lesser bugs, more accurate
    steering and implementing drifting physics, however for present testing, this level of abstraction is sufficient
    """

    def __init__(self, starting_position=(550, 130), starting_orientation=270,
                 car_proportions=pygame.Vector2(2.718, 4.287) * game_core.meter_pixel_conversion):
        super().__init__()
        self.is_crashed = False
        self.position = pygame.Vector2(starting_position)
        self.velocity = pygame.Vector2(0, 0)
        self.acceleration = pygame.Vector2(0, 0)
        self.direction = starting_orientation
        self.progress = 0

        self.current_throttle_input = 0.0
        self.current_steer_input = 0.0
        self.throttle = 0.0
        self.steer = 0.0

        self.driving_force = 40
        self.braking_force = 80
        self.max_speed = 200
        self.max_steer = math.radians(20)
        self.car_proportions = car_proportions

        self.steer_response = 0.3
        self.throttle_factor = 0.08
        self.brake_factor = 0.06
        self.wheelbase = 3.6
        self.max_lateral_accel = 10.0

        """
        The '_original_car' variable is necessary for rotating the sprite properly since the car sprite is rotated
        to an angle with respect to the natural orientation of the sprite on the screen.
        """
        self._original_car = r.set_image("Mclaren")
        self._original_car = pygame.transform.scale(self._original_car, car_proportions)
        self._original_car = pygame.transform.rotate(self._original_car, 180)
        self._update_car_sprite_position()

        """
        hitbox -> Stores coordinates of 4 corners of hitbox
        """
        self.hitbox = []
        self._update_hitbox()

    @abstractmethod
    def update(self, *args, **kwargs):
        pass

    def collision_detection(self, track):
        self._update_hitbox()
        if not self.is_crashed:
            self.is_crashed = track.hitbox_collision_detection(self.hitbox)

    def _car_movement(self, target_steer, target_throttle):
        self._update_controls(target_steer, target_throttle)
        self._update_steer_and_throttle()
        self._update_car_vector_values()
        self._update_car_sprite_position()

    def _update_steer_and_throttle(self):
        target_steer = self.max_steer * self.current_steer_input
        self.steer += (target_steer - self.steer) * self.steer_response

        if self.current_throttle_input > 0:
            self.throttle = self.driving_force * self.current_throttle_input
        else:
            self.throttle = self.braking_force * self.current_throttle_input

    def _update_controls(self, target_steer, target_throttle):
        d_steer_input = target_steer - self.current_steer_input
        d_throttle_input = target_throttle - self.current_throttle_input
        if abs(d_steer_input) > 0.05:
            self.current_steer_input += 0.15 * r.sign(d_steer_input)
        else:
            self.current_steer_input = target_steer
        self.current_steer_input = max(-1.0, min(1.0, self.current_steer_input))
        if abs(d_throttle_input) > 0.1:
            throttle_update_factor = self.throttle_factor if target_throttle >= 0 else self.brake_factor
            self.current_throttle_input += throttle_update_factor * r.sign(d_throttle_input)
        else:
            self.current_throttle_input = target_throttle
        self.current_throttle_input = max(-1.0, min(1.0, self.current_throttle_input))

    def _update_car_vector_values(self):
        dt = 1 / game_core.frame_rate
        direction_unit_vector = pygame.Vector2(0, 1).rotate(self.direction)
        self.acceleration = self.throttle * direction_unit_vector
        if self.throttle == 0:
            self.velocity *= 0.995
        else:
            self.velocity += self.acceleration * dt

        if self.velocity.length_squared() != 0:
            self.velocity.clamp_magnitude_ip(self.max_speed)
        speed = self.velocity.magnitude()
        grip = 0.5
        self.velocity = self.velocity.lerp(direction_unit_vector * speed, grip)
        self.position += self.velocity * dt
        speed_m = speed / game_core.meter_pixel_conversion
        if speed_m > 0.1:
            max_angular_velocity = self.max_lateral_accel / speed_m
            angular_velocity = max(-max_angular_velocity,
                                   min(max_angular_velocity,
                                                    (math.tan(self.steer) * speed_m) / self.wheelbase))
            self.direction += math.degrees(angular_velocity) * dt

    def _update_hitbox(self):
        center_x, center_y = self.rect.center
        width, length = self.car_proportions / 2
        width -= 5
        corners = [(-width, length), (width, length), (width, -length), (-width, -length)]

        for i in range(4):
            x, y = corners[i]
            angle = math.radians(self.direction)
            corners[i] = (center_x + x * math.cos(angle) - y * math.sin(angle),
                          center_y + x * math.sin(angle) + y * math.cos(angle))
        self.hitbox[:] = corners

    def update_and_get_progress(self, track_spine):
        center = Vector2(self.rect.center)
        closest_point = min(track_spine, key=lambda point: (center - point).length_squared())
        current_progress = track_spine.index(closest_point) / (len(track_spine) - 1)

        change_in_progress = current_progress - self.progress
        if change_in_progress > 0.5:
            change_in_progress -= 1.0
        elif change_in_progress <= -0.5:
            change_in_progress += 1.0

        normalised_progress = self.progress + change_in_progress
        is_lap_finished = False
        if normalised_progress > self.progress:
            if normalised_progress >= 1:
                # print("lap completed!")
                self.progress = normalised_progress
                is_lap_finished = True
            else:
                self.progress = normalised_progress
        else:
            if normalised_progress <= -0.5:
                print("Wrong way! Please reset")
                self.is_crashed = True
            else:
                self.progress = normalised_progress
        return self.progress, is_lap_finished

    def _update_car_sprite_position(self):
        self.image = pygame.transform.rotate(self._original_car, -self.direction)
        self.rect = self.image.get_rect(center=(int(self.position.x), int(self.position.y)))

class PlayerCar(Car):
    def __init__(self, starting_position = None):
        super().__init__(starting_position)

        self.steering_slider = elements.UIHorizontalSlider(
            relative_rect=pygame.Rect((game_core.screen_dimensions[0] / 2, 700), (600, 30)),
            start_value=0,
            value_range=(-100, 100),
            manager=game_core.gui_manager
        )

        self.throttle_and_braking_meter = elements.UIProgressBar(
            relative_rect=pygame.Rect((game_core.screen_dimensions[0] / 4 - 300, 700), (500, 30)),
            manager=game_core.gui_manager
        )
        self.speedometer = UIGaugeMeter(
            relative_rect=pygame.Rect((game_core.screen_dimensions[0] / 4 - 300, 600), (200, 100)),
            manager=game_core.gui_manager
        )

    def update(self):
        self._car_movement(self._handle_steering(), self._handle_throttle())
        self.throttle_and_braking_meter.set_current_progress((self.current_throttle_input+1)*50)
        self.speedometer.update_value(self.velocity.magnitude())

    def _handle_steering(self):
        target_steering = self.steering_slider.get_current_value()/100
        return target_steering

    def _handle_throttle(self):
        target_throttle = 0
        if pygame.K_w in game_core.pressed_keys:
            target_throttle = 1
        elif pygame.K_s in game_core.pressed_keys:
            target_throttle = -1
        return target_throttle

class AICar(Car):
    def __init__(self, starting_position = None):
        super().__init__(starting_position)
        self.ray_cast_angles = [15, 30, 45, 60, 75, 60]
        self.car_max_ray_cast = game_core.screen_dimensions[0] * 0.8
        self.prev_steer_input = 0

    def update(self, target_actions):
        self._car_movement(*target_actions)

    # Calls ray cast method of track to get point of collision and normalised distance (w.r.t max ray length)
    # Stores distances as sensor data
    # Adds coordinates of start and end point of each ray to debugger
    def ray_cast(self, track, index):
        center = pygame.Vector2(self.rect.center)
        rays = []

        forward_ray = pygame.Vector2(0, 1).rotate(self.direction) * self.car_max_ray_cast
        rays.append((center, center + forward_ray))
        for angle in self.ray_cast_angles:
            rays.append((center, center + forward_ray.rotate(angle)))
            rays.append((center, center + forward_ray.rotate(-angle)))

        collided_rays = []
        sensors = []
        for ray in rays:
            hit_point, normalised_collision_distance = track.ray_cast(ray)
            collided_rays.append((center, hit_point))
            sensors.append(normalised_collision_distance)
        game_core.game_mode.debug_elements["rays"][index] = collided_rays
        return sensors

    def compute_reward(self, is_stuck, is_lap_finished, prev_progress, mean_progress, sensors):
        reward = 0
        rewards_breakdown = []

        # 1) Stopping penalty
        if self.is_crashed:
            if is_stuck:
                stopped_penalty = -3.0
            else:
                stopped_penalty = -7.0
            reward += stopped_penalty
            # 2) Max progress bonus
            current_progress = self.progress
            progress_change = current_progress - mean_progress
            exploration_coef = 140 if progress_change >  0  else 40
            exploration_reward = exploration_coef * progress_change
            reward += exploration_reward
            rewards_breakdown.append(exploration_reward)

            rewards_breakdown = [stopped_penalty, exploration_reward, 0, 0 , 0 , 0 , 0 , 0, 0]
            return reward, rewards_breakdown
        else:
            rewards_breakdown.append(0)
            rewards_breakdown.append(0)
        # --------------------------------------------------- #

        # 3) Efficiency reward
        distance_moved = self.progress - prev_progress

        if distance_moved < -0.5:
            distance_moved += 1.0

        if distance_moved > 1e-4:
            speed = self.velocity.magnitude()
            normalized_speed = speed / self.max_speed
            speed_reward = 0.025 * normalized_speed**2
            # efficiency_reward = 0.15 *  np.sqrt(distance_moved * 70 + speed_reward)
            efficiency_reward = distance_moved * 45 + speed_reward

            rewards_breakdown.append(efficiency_reward)
            reward += efficiency_reward
        else:
            rewards_breakdown.append(-0.001)
            reward-= 0.001
        # --------------------------------------------------- #

        # 4) Imbalance reward
        left_sensors = [sensors[2], sensors[4], sensors[6], sensors[8]]
        avg_left = sum(left_sensors) / 4
        right_sensors = [sensors[1], sensors[3], sensors[5], sensors[7]]
        avg_right = sum(right_sensors) / 4
        imbalance = avg_right - avg_left
        # imbalance_reward = (0.009 -imbalance * 0.15)
        # imbalance_reward = - 0.08 * (imbalance ** 1.5)
        # aligned_steering_reward = imbalance * self.current_steer_input * 0.8

        left_clearance = sum(left_sensors)
        right_clearance = sum(right_sensors)

        aligned_steering_reward = ((right_clearance - left_clearance) * self.current_steer_input ) * 0.6
        if distance_moved > 1e-4:


            reward += aligned_steering_reward
            rewards_breakdown.append(aligned_steering_reward)
        else:
            rewards_breakdown.append(0)
        # --------------------------------------------------- #

        # 5) Wall hugging penalty
        min_sensor = min(sensors) if sensors else 0
        if min_sensor < 0.02:
            reward -= (0.02 - min_sensor) * 14
            rewards_breakdown.append(-(0.02 - min_sensor) * 14)
        else:
            rewards_breakdown.append(0)
        # --------------------------------------------------- #

        # 6) Steer anticipation reward
        front_sensor = sensors[0]  # assuming 0 is straight ahead
        corner_strength = max(0.0, 0.5 - front_sensor) * abs(imbalance)
        steer_anticipation_reward = corner_strength * self.current_steer_input  * 0
        if distance_moved > 1e-4:
            reward += steer_anticipation_reward
            rewards_breakdown.append(steer_anticipation_reward)
        else:
            rewards_breakdown.append(0)

        # --------------------------------------------------- #
        # 7) No steer penalty

        if min(sensors) < 0.03 and distance_moved > 1e-4:
            no_steering_penalty = -1.5 * (1- abs(self.current_steer_input))
            reward += no_steering_penalty
            rewards_breakdown.append(no_steering_penalty)
        else:
            rewards_breakdown.append(0)
        # --------------------------------------------------- #

        # 8) Survival bonus
        reward += 0.004
        rewards_breakdown.append(0.004)
        # ---------------------------------------------------

        # 9) Finish lap bonus
        if is_lap_finished:
            reward += 10
            rewards_breakdown.append(10)
        else:
            rewards_breakdown.append(0)


        reward = np.clip(reward, -5, 30)
        return reward, rewards_breakdown
