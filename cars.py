"""
Defines base Car class that handles physics simulation, collision detection and controlling of throttle and steering.
PlayerCar extends Car and implements methods and GUI elements to allow the player to control the car movement.
AICar implements methods for ray casting, that returns sensor data for the RL model. It also implements method for
computing rewards for each action.

Classes:
    Car: Abstract base class for both car types with physics simulation and collision detection
    PlayerCar: Human-controlled car with GUI elements (HUD, controls)
    AICar: AI-controlled car with sensors and reward computation
"""

import math
from abc import ABC, abstractmethod

import numpy as np
import pygame
from pygame import Vector2
from pygame_gui.elements import UIPanel, UIProgressBar, UIHorizontalSlider

import resources as r
from resources import game_core
from gui_custom_elements import UIGaugeMeter

class Car(pygame.sprite.Sprite, ABC):
    """
    Implements car physics including throttle/brake, steering, velocity, and angular motion. Subclasses must implement 
    update() and reset() for centralised control.

    Attributes:
        is_crashed (bool): Whether car has collided with track walls
        position (Vector2): Current position in pixels
        velocity (Vector2): Current velocity vector in pixels/second
        acceleration (Vector2): Current acceleration vector
        direction (float): Heading angle in degrees (0 = up)
        progress (float): Cumulative laps completed (e.g., 2.5 = 2.5 laps)
        current_throttle_input (float): Target throttle [-1, 1]
        current_steer_input (float): Target steering [-1, 1]
        throttle (float): Actual throttle force being applied
        steer (float): Actual steering angle in radians
        hitbox (list[tuple]): Four corner coordinates for collision detection
    """
    DRIVING_FORCE = 50
    BRAKING_FORCE = 90
    MAX_SPEED = 300
    MAX_STEER_RAD = math.radians(30)
    WHEELBASE = 3.6
    MAX_LATERAL_ACCELERATION = 20.00

    STEER_RESPONSE = 0.3
    THROTTLE_RESPONSE = 0.1
    BRAKE_RESPONSE = 0.04
    FRICTION_DECAY = 0.992
    GRIP = 0.5
    
    def __init__(self,
                 starting_position=(game_core.screen_dimensions[0]/2,
                                    game_core.screen_dimensions[1]/2),
                 starting_orientation =270,
                 car_proportions=pygame.Vector2(272, 429) / (1.1 * game_core.METER_PIXEL_CONVERSION),
                 image = "Mclaren"):
        """initialise car with starting state.

        Args:
            starting_position (tuple): (x, y) coordinates of start line
            starting_orientation (float): Initial direction to face (degrees).
            car_proportions (Vector2): (width, height) in pixels
            image (str): Car sprite filename
        """
        super().__init__()

        # Default parameters
        self.starting_position = starting_position
        self.starting_orientation = starting_orientation
        self.direction = starting_orientation
        self.car_proportions = car_proportions
        """
        The '_original_car' variable is necessary for rotating the sprite properly since the car sprite is rotated
        to an angle with respect to the natural orientation of the sprite on the screen.
        """
        self._original_car = r.set_image(image)
        self._original_car = pygame.transform.scale(self._original_car, car_proportions)
        self._original_car = pygame.transform.rotate(self._original_car, 180)

        # State variables
        self.is_crashed = False
        self.position = pygame.Vector2(starting_position)
        self.velocity = pygame.Vector2(0, 0)
        self.acceleration = pygame.Vector2(0, 0)
        self.progress = 0
        self.hitbox = []

        # Control inputs
        self.current_throttle_input = 0.0
        self.current_steer_input = 0.0

        # Controls states
        self.throttle = 0.0
        self.steer = 0.0

        self._update_car_sprite_position()
        self._update_hitbox()

    @abstractmethod
    def update(self, *args, **kwargs):
        """
        Update car state for one timestep.
        Subclasses must implement this.
        """
        pass

    def collision_detection(self, track):
        """
        Check if car's hitbox collides with track walls. Updates is_crashed state if collision detected.

        Args:
            track (Track): Track object with collision detection methods
        """
        self._update_hitbox()
        if not self.is_crashed:
            self.is_crashed = track.hitbox_collision_detection(self.hitbox)

    def drive_car(self, target_steer, target_throttle):
        """
        Update car state based on inputs.

        Smoothly increments throttle and steer inputs, calculates forces, updates velocity and position, then updates
        sprite and hitbox.

        Args:
            target_steer (float): Desired steering input [-1, 1]
            target_throttle (float): Desired throttle input [-1, 1]
        """
        self._update_controls(target_steer, target_throttle)
        self._update_steer_and_throttle()
        self._update_car_vector_values()
        self._update_car_sprite_position()


    def _update_controls(self, target_steer, target_throttle):
        """
        Smooth control inputs toward target values. Limits rate of increase with response factors.

        Args:
            target_steer (float): Desired steering [-1, 1]
            target_throttle (float): Desired throttle [-1, 1]
        """

        # Increment steer input
        d_steer_input = target_steer - self.current_steer_input
        if abs(d_steer_input) > 0.05:
            self.current_steer_input += 0.15 * r.sign(d_steer_input)
        else:
            self.current_steer_input = target_steer
        self.current_steer_input = max(-1.0, min(1.0, self.current_steer_input))

        # Increment throttle input
        d_throttle_input = target_throttle - self.current_throttle_input
        if abs(d_throttle_input) > 0.05:
            throttle_update_factor = self.THROTTLE_RESPONSE if target_throttle >= 0 else self.BRAKE_RESPONSE
            self.current_throttle_input += throttle_update_factor * r.sign(d_throttle_input)
        else:
            self.current_throttle_input = target_throttle
        self.current_throttle_input = max(-1.0, min(1.0, self.current_throttle_input))

    def _update_steer_and_throttle(self):
        """Convert control inputs to steering angle and throttle force."""

        target_steer = self.MAX_STEER_RAD * self.current_steer_input
        self.steer += (target_steer - self.steer) * self.STEER_RESPONSE

        if self.current_throttle_input > 0:
            self.throttle = self.DRIVING_FORCE * self.current_throttle_input
        else:
            self.throttle = self.BRAKING_FORCE * self.current_throttle_input

    def _update_car_vector_values(self):
        """
        Update velocity, position, and heading based on physics.

        Implements simplified car physics with:
        - Velocity decay when no throttle
        - Grip (smoothly interpolates velocity)
        - Bicycle model for turning
        """

        dt = game_core.dt_ms/1000

        # Update acceleration in heading direction
        direction_unit_vector = pygame.Vector2(0, 1).rotate(self.direction)
        self.acceleration = self.throttle * direction_unit_vector

        # Update velocity
        if self.throttle == 0:
            self.velocity *= self.FRICTION_DECAY
        else:
            self.velocity += self.acceleration * dt
        if self.velocity.length_squared() != 0:
            self.velocity.clamp_magnitude_ip(self.MAX_SPEED)

        # Apply grip
        speed = self.velocity.magnitude()
        self.velocity = self.velocity.lerp(direction_unit_vector * speed, self.GRIP)

        # Update position
        self.position += self.velocity * dt

        # Update direction based on bicycle model
        speed_m = speed / game_core.METER_PIXEL_CONVERSION      # Speed in m/s
        if speed_m > 0.1:
            # Calculate clamped angular velocity
            max_angular_velocity = self.MAX_LATERAL_ACCELERATION / speed_m
            angular_velocity = max(-max_angular_velocity,
                                   min(max_angular_velocity,
                                                    (math.tan(self.steer) * speed_m) / self.WHEELBASE))

            # Update direction
            self.direction += math.degrees(angular_velocity) * dt

    def _update_hitbox(self):
        """
        Recalculate hitbox corners based on current position and rotation.
        """
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
        """
        Calculate car's progress along track and detect lap completion.

        Finds nearest point on track spine to calculates fractional progress (0-1). Handles wraparound at track
        boundaries, and detects lap completion or wrong-way driving.

        Args:
            track_spine (list[Vector2]): Ordered list of track centerline points

        Returns:
            tuple(float, bool): (cumulative_progress, is_lap_finished)
                - cumulative_progress: Total laps + fractional progress (e.g., 2.5)
                - is_lap_finished: True if lap boundary was just crossed
        """

        # Calculate fractional progress by finding the nearest track spine point
        center = Vector2(self.rect.center)
        closest_point = min(track_spine, key=lambda point: (center - point).length_squared())
        current_progress = track_spine.index(closest_point) / (len(track_spine) - 1)

        # Handle wrap-around by checking change in progress
        change_in_progress = current_progress - (self.progress % 1)
        if change_in_progress > 0.5:
            change_in_progress -= 1.0
        elif change_in_progress <= -0.5:
            change_in_progress += 1.0

        # Update progress
        new_progress = self.progress + change_in_progress


        is_lap_finished = False
        if new_progress > self.progress:
            # Check if lap was completed
            if math.floor(new_progress) - math.floor(self.progress) > 0:
                print("lap completed!")
                self.progress = new_progress
                is_lap_finished = True
            else:
                self.progress = new_progress
        else:
            # Check if car is travelling wrong way
            if new_progress <= -0.5:
                print("Wrong way! Please reset")
                self.is_crashed = True
                self.progress = 0
            else:
                self.progress = new_progress

        # print(self.progress)
        return self.progress, is_lap_finished

    def _update_car_sprite_position(self):
        """Rotate sprite and update position."""
        self.image = pygame.transform.rotate(self._original_car, -self.direction)
        self.rect = self.image.get_rect(center=(int(self.position.x), int(self.position.y)))

    @abstractmethod
    def reset(self):
        """
        Reset car to starting state.
        Child classes must call this and then reset their specific attributes.
        """
        self.is_crashed = False
        self.position = pygame.Vector2(self.starting_position)
        self.direction = self.starting_orientation
        self.velocity = pygame.Vector2(0, 0)
        self.acceleration = pygame.Vector2(0, 0)
        self.current_throttle_input = 0
        self.current_steer_input = 0
        self.throttle = 0
        self.steer = 0
        self.progress = math.floor(self.progress)

        self._update_car_sprite_position()
        self._update_hitbox()


class PlayerCar(Car):
    """
    Car that is controlled by player input.
    Displays HUD panel with speedometer, throttle meter, and steering slider. Controlled with keyboard keys by default.

    Attributes:
        hud_panel (UIPanel): Container of all HUD elements
        speedometer (UIGaugeMeter): Gauge speed display
        throttle_and_braking_meter (UIProgressBar): Indicates current throttle/brake input
        steering_slider (UIHorizontalSlider): Optional slider to control steering
        is_slider_enabled (bool): Indicates whether user steers with keyboard or slider.
    """
    def __init__(self,
                 starting_position = None,
                 starting_orientation = 270,
                 image = "purple_car",
                 car_proportions=pygame.Vector2(272, 429) / (0.8 * game_core.METER_PIXEL_CONVERSION)):
        """
        Initialise player cor with player control GUI elements

        Args:
           starting_position (tuple): (x, y) coordinates of start line
            starting_orientation (float): Initial direction to face (degrees).
            image (str): Car sprite filename
            car_proportions (Vector2): Car size
        """
        super().__init__(starting_position, starting_orientation = starting_orientation,
                         image= image, car_proportions=  car_proportions)

        # HUB Panel controls
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
        """
        Toggles the steering controls between keyboard controls and steering slider
        """
        self.is_slider_enabled = not self.is_slider_enabled
        if self.is_slider_enabled:
            self.steering_slider.enable()
        else:
            self.steering_slider.disable()

    def delete_control_panel(self):
        """
        Removes all HUD control panel elements when resetting cars/simulation
        """
        self.hud_panel.kill()
        self.steering_slider.kill()
        self.throttle_and_braking_meter.kill()
        self.speedometer.kill()

    def update(self):
        """
        Handles car movement and control panel updates
        """
        self.drive_car(self._handle_steering(), self._handle_throttle())
        self.throttle_and_braking_meter.set_current_progress((self.current_throttle_input+1)*50)
        self.speedometer.update_value(self.velocity.magnitude())

    def _handle_steering(self):
        """
        Determines target steer value based on user input.
        Controlled by steering slider value if slider is enabled. Else controlled with keystrokes:
        'A' -> turn left, 'D' -> turn right

        Returns:
            float: [-1, 1]
                -1 -> max left turn
                1 -> max right turn
        """
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
        """
        Determines target throttle value based on user input.
        'W' -> Full throttle
        'S' -> Full brake

        Returns:
            float: [-1, 1]
                -1 -> max braking
                1 -> max throttle
        """
        target_throttle = 0
        if pygame.K_w in game_core.pressed_keys:
            target_throttle = 1
        elif pygame.K_s in game_core.pressed_keys:
            target_throttle = -1
        return target_throttle

    def reset(self):
        """
        Resets car to starting state.
        Re-initialises the HUD control panel
        """
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
    """
    AI-controlled car with sensors and reward computation.
    Updated on target actions determined by AI model. ray_cast() returns sensor data. compute_reward() returns reward
    signals for AI behaviour.
    """

    RAY_CAST_ANGLES = [15, 30, 45, 60, 75, 60]
    MAX_RAY_CAST = game_core.screen_dimensions[0] * 0.8

    CRASH_PENALTY = -40.0
    STUCK_PENALTY = -10.0
    SPEED_REWARD_COEF = 0.025
    PROGRESS_REWARD_COEF = 45
    ALIGNED_STEERING_REWARD_COEF  = 0.55
    WALL_PROXIMITY_PENALTY_COEF = -14
    WALL_PROXIMITY_THRESHOLD = 0.01
    ANTICIPATED_STEERING_REWARD_COEF = 2
    NO_STEER_PENALTY_COEF = -1
    NO_STEER_PENALTY_THRESHOLD = 0.03
    SURVIVAL_BONUS = 0.003
    LAP_COMPLETION_BONUS = 70

    def __init__(self,
                 starting_position=None,
                 starting_orientation=270,
                 image="black_car",
                 car_proportions = pygame.Vector2(272, 429) / (1 * game_core.METER_PIXEL_CONVERSION)):
        """
        Initialise AI Car

        Args:
           starting_position (tuple(float, float)): (x, y) coordinates of start line
            starting_orientation (float): Initial direction to face (degrees).
            image (str): Car sprite filename
            car_proportions (Vector2): Car size
        """
        super().__init__(starting_position, starting_orientation=starting_orientation,
                         image=image, car_proportions=car_proportions)


    def update(self, target_actions):
        """
        Handles car movement. Car is controlled by actions output by the RL model
        
        Args:
            target_actions (tuple(float, float)): (target_throttle, target_steer) with both values [-1, 1]
        """
        self.drive_car(*target_actions)


    def get_sensors(self, track):
        """
        Computes and returns a list of sensor values.
        Sensors values are the distance to the nearest wall in different directions. Distances are determined using
        ray-casting.

        Args:
            track (Track): The track which the car is currently driving on.
        """
        center = pygame.Vector2(self.rect.center)
        rays = []

        forward_ray = pygame.Vector2(0, 1).rotate(self.direction) * self.MAX_RAY_CAST
        rays.append((center, center + forward_ray))
        for angle in self.RAY_CAST_ANGLES:
            rays.append((center, center + forward_ray.rotate(angle)))
            rays.append((center, center + forward_ray.rotate(-angle)))

        collided_rays = []
        sensors = []
        for ray in rays:
            hit_point, normalised_collision_distance = track.ray_cast(ray)
            collided_rays.append((center, hit_point))
            sensors.append(normalised_collision_distance)

        return sensors

    def reset(self):
        super().reset()

    def compute_reward(self, is_stuck, is_lap_finished, prev_progress, sensors):
        """
        Calculate rewards for most recent actions taken by RL model.
        Combines multiple reward components. Returns both total reward, and breakdown of individual components.

        Args:
            is_stuck (bool): If car is stuck being still.
            is_lap_finished (bool): If car finished lap
            prev_progress (float): Previous progress for computing change in progress
            sensors (list[float]): Sensor data of distances to walls in different directions

        Returns:
            tuple(float, list[float]): (total_reward, rewards_breakdown)
                - total_reward: Sum of all rewards
                - rewards_breakdown: Individual reward components for analysis
        """

        rewards_breakdown = []

        # --------------------------------------------------- #
        """ 
        1) Stopping penalty 
        """
        if self.is_crashed:
            if is_stuck:
                stopped_penalty = self.STUCK_PENALTY
            else:
                stopped_penalty = self.CRASH_PENALTY

            """
            OMITTED: Exploration bonus
            """
            # current_progress = self.progress
            # progress_change = current_progress - mean_progress
            # exploration_coef = 140 if progress_change >  0  else 40
            # exploration_reward = (r.sign(progress_change) + 2.5*
            #                       math.log(abs(exploration_coef * progress_change) + 1))
            # reward += exploration_reward
            # rewards_breakdown.append(exploration_reward)

            rewards_breakdown = [stopped_penalty, 0, 0 , 0 , 0 , 0 , 0, 0]
            return sum(rewards_breakdown), rewards_breakdown
        else:
            rewards_breakdown.append(0)
        # --------------------------------------------------- #

        # --------------------------------------------------- #
        """
        2) Efficiency reward
        """
        distance_moved = self.progress - prev_progress

        if distance_moved < -0.5:
            distance_moved += 1.0

        if distance_moved > 1e-4:
            speed = self.velocity.magnitude()
            normalized_speed = speed / self.MAX_SPEED
            speed_reward = self.SPEED_REWARD_COEF * normalized_speed**2
            efficiency_reward = self.PROGRESS_REWARD_COEF * distance_moved + speed_reward

            rewards_breakdown.append(efficiency_reward)
        else:
            rewards_breakdown.append(-0.001)
        # --------------------------------------------------- #

        # --------------------------------------------------- #
        """
        3) Imbalance reward
        """
        left_sensors = [sensors[2], sensors[4], sensors[6], sensors[8]]
        avg_left = sum(left_sensors) / 4
        right_sensors = [sensors[1], sensors[3], sensors[5], sensors[7]]
        avg_right = sum(right_sensors) / 4
        imbalance = avg_right - avg_left

        left_clearance = sum(left_sensors)
        right_clearance = sum(right_sensors)

        aligned_steering_reward = (((right_clearance - left_clearance) * self.current_steer_input )
                                   * self.ALIGNED_STEERING_REWARD_COEF)
        if distance_moved > 1e-4:
            rewards_breakdown.append(aligned_steering_reward)
        else:
            rewards_breakdown.append(0)
        # --------------------------------------------------- #

        # --------------------------------------------------- #
        """
        4) Wall hugging penalty
        """
        min_sensor = min(sensors) if sensors else 0
        if min_sensor < self.WALL_PROXIMITY_THRESHOLD:
            rewards_breakdown.append((0.02 - min_sensor) * self.WALL_PROXIMITY_PENALTY_COEF)
        else:
            rewards_breakdown.append(0)
        # --------------------------------------------------- #

        # --------------------------------------------------- #
        """
        5) Steer anticipation reward
        """
        front_sensor = sensors[0]
        corner_strength = max(0.0, 0.5 - front_sensor) * abs(imbalance)
        steer_anticipation_reward = corner_strength * self.current_steer_input  * self.ANTICIPATED_STEERING_REWARD_COEF
        if distance_moved > 1e-4:
            rewards_breakdown.append(steer_anticipation_reward)
        else:
            rewards_breakdown.append(0)
        # --------------------------------------------------- #

        # --------------------------------------------------- #
        """
        6) No steer penalty
        """
        if min(sensors) <self.NO_STEER_PENALTY_THRESHOLD and distance_moved > 1e-4:
            no_steering_penalty = self.NO_STEER_PENALTY_COEF * (1- abs(self.current_steer_input))
            rewards_breakdown.append(no_steering_penalty)
        else:
            rewards_breakdown.append(0)
        # --------------------------------------------------- #

        # --------------------------------------------------- #
        """
        7) Survival bonus
        """
        rewards_breakdown.append(self.SURVIVAL_BONUS)
        # ---------------------------------------------------

        # --------------------------------------------------- #
        """
        8) Finish lap bonus
        """
        if is_lap_finished:
            rewards_breakdown.append(self.LAP_COMPLETION_BONUS)
        else:
            rewards_breakdown.append(0)
        # --------------------------------------------------- #

        return sum(rewards_breakdown), rewards_breakdown