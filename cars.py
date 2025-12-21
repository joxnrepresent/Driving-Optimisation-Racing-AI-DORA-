import math
from pygame import Vector2
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

    def __init__(self, starting_position = (550, 130), starting_orientation = 270, car_proportions = pygame.Vector2(2.718, 4.287) * game_core.meter_pixel_conversion):
        super().__init__()
        self.is_crashed = False
        self.position = pygame.Vector2(starting_position)
        self.velocity = pygame.Vector2(0,0)
        self.acceleration = pygame.Vector2(0,0)
        self.direction = starting_orientation
        self.progress = 0

        self.current_throttle_input = 0.0
        self.current_steer_input = 0.0
        self.throttle = 0.0
        self.steer = 0.0

        self.driving_force = 60
        self.braking_force = 150
        self.max_speed = 500
        self.max_steer = 1.7
        self.car_proportions = car_proportions

        self.steer_factor = self.max_steer / 10
        self.throttle_factor = 0.08
        self.brake_factor = 0.06
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
        self._update_hitboxes()


    @abstractmethod
    def update(self, *args, **kwargs):
        pass

    # Changes the state of the car to indicate it has crashed and stop its movement
    def collision_detection(self, track):
        self._update_hitboxes()
        if not self.is_crashed:
            self.is_crashed = track.hitbox_collision_detection(self.hitbox)


    # Calls procedures which handle the movement of the car sprite and update its parameters
    def _car_movement(self, target_steer, target_throttle):
        self._update_controls(target_steer, target_throttle)
        self._update_steer_and_throttle()
        self._update_car_vector_values()
        self._update_car_sprite_position()

    # Increments the steer value.
    # Steer percentage is the value on the steer slider (determines the limit to which the steer can be incremented).
    # The steer_factor is the value by which steer is incremented (gradual changes rather than abrupt updates).
    # Direction of car updated.

    def _update_steer_and_throttle(self):
        steer_limit = self.max_steer * abs(self.current_steer_input) - self.steer_factor
        self.steer += self.steer_factor * r.sign(self.current_steer_input)
        self.steer = r.clamp_value(self.steer, -steer_limit, steer_limit)
        self.direction += self.steer
        self.direction = r.wrap_value(self.direction, 0, 360)

        if self.current_throttle_input > 0:
            self.throttle = self.driving_force * self.current_throttle_input
        else:
            self.throttle = self.braking_force * self.current_throttle_input


    def _update_controls(self, target_steer, target_throttle):
        d_steer_input = target_steer- self.current_steer_input
        d_throttle_input = target_throttle - self.current_throttle_input
        if abs(d_steer_input) > 0.05:
            self.current_steer_input += self.steer_factor * r.sign(d_steer_input)
        elif target_steer == 0:
            self.current_steer_input = 0
        if abs(d_throttle_input) > 0.1:
            throttle_update_factor = self.throttle_factor if target_throttle >= 0 else self.brake_factor
            self.current_throttle_input += throttle_update_factor * r.sign(d_throttle_input)
        elif target_throttle == 0:
            self.current_throttle_input =0



    # Updates the acceleration, velocity and position vectors based on the throttle and direction of the car.
    # Velocity gradually decays when throttle = 0 (roughly emulates friction).
    # Grip variable and lerp function are used to smoothly update car position each frame (arbitrary values).
    """This method would be different when modelling forces."""
    def _update_car_vector_values(self):
        dt = 1/ game_core.frame_rate
        direction_unit_vector = pygame.Vector2(0, 1).rotate(self.direction)
        self.acceleration = self.throttle * direction_unit_vector
        if self.throttle == 0:
            self.velocity *= 0.995
        else:
            self.velocity += self.acceleration * dt
        if self.velocity.length_squared() != 0:
            self.velocity.clamp_magnitude_ip(self.max_speed)
        grip = 0.5
        speed = self.velocity.magnitude()
        self.velocity = self.velocity.lerp(direction_unit_vector * speed, grip)
        self.position += self.velocity * dt

    # Updates the coordinates of the corners of the hitbox w.r.t the position and orientation of the car
    def _update_hitboxes(self):
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
        current_progress = track_spine.index(closest_point)/(len(track_spine)-1)

        change_in_progress = current_progress - self.progress
        if change_in_progress > 0.5:
            change_in_progress -= 1.0
        elif change_in_progress <= -0.5:
            change_in_progress += 1.0

        normalised_progress = self.progress + change_in_progress
        if normalised_progress > self.progress:
            if normalised_progress >= 1:
                print("lap completed!")
                self.progress = normalised_progress - 1
            else:
                self.progress = normalised_progress
        else:
            if normalised_progress <= -0.5:
                print("Wrong way! Please reset")
                self.is_crashed = True
            else:
                self.progress = normalised_progress
        return self.progress

    # Updates the rotation of the car sprite with respect to its direction and the original orientation
    # Updates position (rect) of the car sprite
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


    # Updates the steer value when the steering slider is moved.
    # def _handle_steering(self):
    #     self.steering_wheel_amount = max_steer * self.steering_slider.get_current_value()/100

    def _handle_steering(self):
        target_steering = self.steering_slider.get_current_value()/100
        return target_steering

    # Increments the throttle or braking based on user input.
    # Value decays when there is no input to emulate release of throttle/brake.
    """
    For accuracy, the throttle would have to be handled in a similar way as the steering using a slider of some sort
    to allow more control over the magnitude, since by using keys to control throttle, there is no way to steadily hold 
    the throttle partially pressed down. However for testing purposes, this is ideal since its easier to control than
    having 2 separate sliders
    """
    # def _handle_throttle(self):
    #     if pygame.K_w in game_core.pressed_keys:
    #         self.throttle_and_braking_pedal_amount = r.clamp_value(self.throttle_and_braking_pedal_amount
    #                                                                + throttle_factor, 0, 1)
    #     elif pygame.K_s in game_core.pressed_keys:
    #         self.throttle_and_braking_pedal_amount = r.clamp_value(self.throttle_and_braking_pedal_amount
    #                                                                - brake_factor, -1, 0)
    #     else:
    #         self.throttle_and_braking_pedal_amount *= 0.9
    #         if -0.01 < self.throttle_and_braking_pedal_amount < 0.01:
    #             self.throttle_and_braking_pedal_amount = 0
    #     throttle_normalised_value = (self.throttle_and_braking_pedal_amount + 1) * 50
    #     self.throttle_and_braking_meter.set_current_progress(throttle_normalised_value)
    #     self.speedometer.update_value(self.velocity.magnitude())

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
        self.ray_cast_angles = [10, 20, 40, 60, 75]
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

    def compute_reward(self, prev_progress, max_progress, sensors, actions):

        reward = 0

        rewards_breakdown = []

        # Crash penalty

        if self.is_crashed:

            rewards_breakdown = [-20*(2-self.progress), 0, 0 ,0 ,0 ,0 ,0,0, 0]

            return -20*(2-self.progress), rewards_breakdown

        else:

            rewards_breakdown.append(0)

        # --------------------------------------------------- #

        # Progress reward

        distance_moved = self.progress - prev_progress

        if distance_moved < -0.5:

            distance_moved += 1.0

        if distance_moved > 1e-5:

            rewards_breakdown.append(distance_moved * 100)

            reward += distance_moved * 100

        else:

            rewards_breakdown.append(-0.01)

            reward-= 0.01

        # --------------------------------------------------- #

        # Imbalance reward

        left_sensors = [sensors[2], sensors[4], sensors[6], sensors[8], sensors[10]]

        avg_left = sum(left_sensors) / 5

        right_sensors = [sensors[1], sensors[3], sensors[5], sensors[7], sensors[9]]

        avg_right = sum(right_sensors) / 5

        imbalance = abs(avg_left - avg_right)

        reward -= imbalance * 0.2

        rewards_breakdown.append(-imbalance * 0.2)

        # --------------------------------------------------- #

        # Speed reward

        speed = self.velocity.magnitude()

        normalized_speed = speed / self.max_speed

        rewards_breakdown.append(normalized_speed * 0.4)

        reward += normalized_speed * 0.4

        # --------------------------------------------------- #

        # Max progress reward/penalty

        progress_change = self.progress - max_progress

        reward += 0.1 * progress_change

        rewards_breakdown.append(0.1 * progress_change)

        # --------------------------------------------------- #

        # Wall hugging penalty

        min_sensor = min(sensors) if sensors else 0

        if min_sensor < 0.02:

            reward -= (0.02 - min_sensor) * 10.0

            rewards_breakdown.append(-(0.02 - min_sensor) * 10.0)

        else:

            rewards_breakdown.append(0)

        # --------------------------------------------------- #

        steer, throttle = actions

        # Jittery steering penalty

        steer_change = abs(self.current_steer_input - steer)

        reward -= steer_change * 0.005

        rewards_breakdown.append(-steer_change * 0.005)

        # --------------------------------------------------- #

        # Jittery acceleration penalty

        throttle_change = abs(self.current_throttle_input - throttle)

        reward -= throttle_change * 0.015

        rewards_breakdown.append(-throttle_change * 0.015)

        # --------------------------------------------------- #

        # Time penalty

        reward += 0.005

        rewards_breakdown.append(0.005)

        # --------------------------------------------------- #

        return reward, rewards_breakdown



