import math
import numpy as np
import resources as r
import pygame
from resources import game_core
from gui_custom_elements import UIGaugeMeter
from pygame import Vector2
from abc import ABC, abstractmethod
from pygame_gui.elements import UIPanel, UIProgressBar, UIHorizontalSlider

class Car(pygame.sprite.Sprite, ABC):

    def __init__(self, starting_position=(game_core.screen_dimensions[0]/2, game_core.screen_dimensions[1]/2),
            starting_orientation =270, car_proportions=pygame.Vector2(272, 429) / (1.1*game_core.meter_pixel_conversion) ,
                 image = "Mclaren"):
        super().__init__()
        self.is_crashed = False
        self.starting_position = starting_position
        self.position = pygame.Vector2(starting_position)
        self.velocity = pygame.Vector2(0, 0)
        self.acceleration = pygame.Vector2(0, 0)
        self.direction = starting_orientation
        self.progress = 0

        self.current_throttle_input = 0.0
        self.current_steer_input = 0.0
        self.throttle = 0.0
        self.steer = 0.0

        self.driving_force = 50
        self.braking_force = 90
        self.max_speed = 400
        self.max_steer = math.radians(30)
        self.car_proportions = car_proportions

        self.steer_response = 0.3
        self.throttle_factor = 0.1
        self.brake_factor = 0.04
        self.wheelbase = 3.6
        self.max_lateral_accel = 20.0

        """
        The '_original_car' variable is necessary for rotating the sprite properly since the car sprite is rotated
        to an angle with respect to the natural orientation of the sprite on the screen.
        """
        self._original_car = r.set_image(image)
        self._original_car = pygame.transform.scale(self._original_car, car_proportions)
        self._original_car = pygame.transform.rotate(self._original_car, 180)
        self._update_car_sprite_position()

        """
        hitbox -> Stores coordinates of 4 corners of hitbox
        """
        self.hitbox = []
        self._update_hitbox()

    def set_car_position(self, position, orientation):
        self.position = position
        self.direction = orientation

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
            self.velocity *= 0.992
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

    def set_progress(self, progress):
        self.progress = progress

    def update_and_get_progress(self, track_spine):
        center = Vector2(self.rect.center)
        closest_point = min(track_spine, key=lambda point: (center - point).length_squared())
        current_progress = track_spine.index(closest_point) / (len(track_spine) - 1)

        change_in_progress = current_progress - (self.progress % 1)
        if change_in_progress > 0.5:
            change_in_progress -= 1.0
        elif change_in_progress <= -0.5:
            change_in_progress += 1.0

        new_progress = self.progress + change_in_progress
        is_lap_finished = False
        if new_progress > self.progress:
            if math.floor(new_progress) - math.floor(self.progress) > 0:
                print("lap completed!")
                self.progress = new_progress
                is_lap_finished = True
            else:
                self.progress = new_progress
        else:
            if new_progress <= -0.5:
                print("Wrong way! Please reset")
                self.is_crashed = True
                self.progress = 0
            else:
                self.progress = new_progress
        return self.progress, is_lap_finished

    def _update_car_sprite_position(self):
        self.image = pygame.transform.rotate(self._original_car, -self.direction)
        self.rect = self.image.get_rect(center=(int(self.position.x), int(self.position.y)))

    @abstractmethod
    def reset(self):
        self.is_crashed = False
        self.position = pygame.Vector2(self.starting_position)
        self.velocity = pygame.Vector2(0, 0)
        self.acceleration = pygame.Vector2(0, 0)
        self.current_throttle_input = 0
        self.current_steer_input = 0
        self.throttle = 0
        self.steer = 0
        self.direction = 270
        self.progress = math.floor(self.progress)

        self._update_car_sprite_position()
        self._update_hitbox()


class PlayerCar(Car):
    def __init__(self, starting_position = None, image = "purple_car",
                 car_proportions=pygame.Vector2(272, 429) / (0.8*game_core.meter_pixel_conversion)):
        super().__init__(starting_position, image= image, car_proportions=  car_proportions)

        self.hud_panel = UIPanel(
            relative_rect=pygame.Rect(
                (0, game_core.screen_dimensions[1] - 200,
                 (game_core.screen_dimensions[0]), 200)
            ),
            manager=game_core.gui_manager,
            starting_height=1000,
            object_id='#HUD_panel'
        )

        self.throttle_and_braking_meter =UIProgressBar(
            relative_rect=pygame.Rect((game_core.screen_dimensions[0] / 4 - 200, 100), (400, 50)),
            manager=game_core.gui_manager,
            container= self.hud_panel,
            object_id = '#UIProgressBar'
        )

        self.steering_slider = UIHorizontalSlider(
            relative_rect=pygame.Rect((game_core.screen_dimensions[0] *3/4 - 200, 100), (400, 50)),
            start_value=0,
            value_range=(-100, 100),
            manager=game_core.gui_manager,
            container= self.hud_panel,
            object_id = '#UIHorizontalSlider'
        )
        self.is_slider_enabled = False
        self.steering_slider.disable()

        self.speedometer = UIGaugeMeter(
            relative_rect=pygame.Rect((game_core.screen_dimensions[0]/2  - 125 , 20), (250, 125)),
            manager=game_core.gui_manager,
            container= self.hud_panel
        )

    def toggle_slider(self):
        self.is_slider_enabled = not self.is_slider_enabled
        if self.is_slider_enabled:
            self.steering_slider.enable()
        else:
            self.steering_slider.disable()

    def delete_control_panel(self):
        self.steering_slider.kill()
        self.throttle_and_braking_meter.kill()
        self.speedometer.kill()

    def update(self):
        self._car_movement(self._handle_steering(), self._handle_throttle())
        self.throttle_and_braking_meter.set_current_progress((self.current_throttle_input+1)*50)
        self.speedometer.update_value(self.velocity.magnitude())

    def _handle_steering(self):
        if self.is_slider_enabled:
            target_steering = self.steering_slider.get_current_value()/100
        else:
            if pygame.K_a in game_core.pressed_keys:
                target_steering = -1
            elif pygame.K_d in game_core.pressed_keys:
                target_steering = 1
            else:
                target_steering = 0
        return target_steering

    def _handle_throttle(self):
        target_throttle = 0
        if pygame.K_w in game_core.pressed_keys:
            target_throttle = 1
        elif pygame.K_s in game_core.pressed_keys:
            target_throttle = -1
        return target_throttle

    def reset(self):
        super().reset()
        self.steering_slider.kill()
        self.throttle_and_braking_meter.kill()
        self.speedometer.kill()
        self.throttle_and_braking_meter =UIProgressBar(
            relative_rect=pygame.Rect((game_core.screen_dimensions[0] / 4 - 200, 100), (400, 50)),
            manager=game_core.gui_manager,
            container= self.hud_panel,
            object_id = '#UIProgressBar'
        )

        self.steering_slider = UIHorizontalSlider(
            relative_rect=pygame.Rect((game_core.screen_dimensions[0] *3/4 - 200, 100), (400, 50)),
            start_value=0,
            value_range=(-100, 100),
            manager=game_core.gui_manager,
            container= self.hud_panel,
            object_id = '#UIHorizontalSlider'
        )
        self.is_slider_enabled = False
        self.steering_slider.disable()

        self.speedometer = UIGaugeMeter(
            relative_rect=pygame.Rect((game_core.screen_dimensions[0]/2  - 125 , 20), (250, 125)),
            manager=game_core.gui_manager,
            container= self.hud_panel
        )



class AICar(Car):
    def __init__(self, starting_position = None, image = "black_car",
                 car_proportions = pygame.Vector2(272, 429) / (1*game_core.meter_pixel_conversion)):
        super().__init__(starting_position, car_proportions=car_proportions, image=image)
        self.ray_cast_angles = [15, 30, 45, 60, 75, 60]
        self.car_max_ray_cast = game_core.screen_dimensions[0] * 0.8

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

    def reset(self):
        super().reset()

    def compute_reward(self, is_stuck, is_lap_finished, prev_progress, mean_progress, sensors):
        reward = 0
        rewards_breakdown = []

        # 1) Stopping penalty
        if self.is_crashed:
            if is_stuck:
                stopped_penalty = -10.0
            else:
                stopped_penalty = -50.0
            reward += stopped_penalty
            # 2) Max progress bonus
            current_progress = self.progress
            progress_change = current_progress - mean_progress
            exploration_coef = 140 if progress_change >  0  else 40
            exploration_reward = r.sign(progress_change) + 2.5* math.log(abs(exploration_coef * progress_change) + 1) * 0
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
            efficiency_reward = distance_moved * 40 + speed_reward

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

        aligned_steering_reward = ((right_clearance - left_clearance) * self.current_steer_input ) * 0.55
        if distance_moved > 1e-4:


            reward += aligned_steering_reward
            rewards_breakdown.append(aligned_steering_reward)
        else:
            rewards_breakdown.append(0)
        # --------------------------------------------------- #

        # 5) Wall hugging penalty
        min_sensor = min(sensors) if sensors else 0
        if min_sensor < 0.01:
            reward -= (0.02 - min_sensor) * 14
            rewards_breakdown.append(-(0.02 - min_sensor) * 14)
        else:
            rewards_breakdown.append(0)
        # --------------------------------------------------- #

        # 6) Steer anticipation reward
        front_sensor = sensors[0]  # assuming 0 is straight ahead
        corner_strength = max(0.0, 0.5 - front_sensor) * abs(imbalance)
        steer_anticipation_reward = corner_strength * self.current_steer_input  * 2
        if distance_moved > 1e-4:
            reward += steer_anticipation_reward
            rewards_breakdown.append(steer_anticipation_reward)
        else:
            rewards_breakdown.append(0)

        # --------------------------------------------------- #
        # 7) No steer penalty

        if min(sensors) < 0.03 and distance_moved > 1e-4:
            no_steering_penalty = -1 * (1- abs(self.current_steer_input))
            reward += no_steering_penalty
            rewards_breakdown.append(no_steering_penalty)
        else:
            rewards_breakdown.append(0)
        # --------------------------------------------------- #

        # 8) Survival bonus
        reward += 0.003
        rewards_breakdown.append(0.003)
        # ---------------------------------------------------

        # 9) Finish lap bonus
        if is_lap_finished:
            reward += 70
            rewards_breakdown.append(70)
        else:
            rewards_breakdown.append(0)


        return reward, rewards_breakdown