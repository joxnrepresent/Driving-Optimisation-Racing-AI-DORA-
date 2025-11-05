import math
from multiprocessing.dummy import current_process
from pygame import Vector2
import pygame
from abc import ABC, abstractmethod
from pygame_gui import elements
import resources as r

from gui_custom_elements import UIGaugeMeter

class Car(pygame.sprite.Sprite, ABC):
    """
    This class is the Car sprite and extends the Sprite class. The movement of the car is controlled by user input.
    The throttle and steer values are used to control the car's movement. The car's acceleration, velocity and position
    are calculated using vector math and arbitrary constants. Movement is not modelling in terms of forces in this
    version.In later versions, forces will be used to model car movement which would allow lesser bugs, more accurate
    steering and implementing drifting physics, however for present testing, this level of abstraction is sufficient
    """

    def __init__(self, starting_position = (0,0)):
        super().__init__()
        self.is_crashed = False
        self.position = pygame.Vector2(r.starting_position)
        self.velocity = pygame.Vector2(0,0)
        self.acceleration = pygame.Vector2(0,0)
        self.direction = r.starting_orientation
        self.progress = 0

        self.throttle = 0
        self.steer = 0.0

        """
        The '_original_car' variable is necessary for rotating the sprite properly since the car sprite is rotated
        to an angle with respect to the natural orientation of the sprite on the screen.
        """
        self._original_car = r.set_image("Mclaren")
        self._original_car = pygame.transform.scale(self._original_car, r.car_proportions)
        self._original_car = pygame.transform.rotate(self._original_car, 180)
        self._update_car_sprite_position()

        """
        hitbox -> Stores coordinates of 4 corners of hitbox
        """
        self.hitbox = []
        self._update_hitboxes()


    @abstractmethod
    def car_movement(self, *args, **kwargs):
        pass

    # Changes the state of the car to indicate it has crashed and stop its movement
    def collision_detection(self, track):
        self._update_hitboxes()
        if not self.is_crashed:
            self.is_crashed = track.hitbox_collision_detection(self.hitbox)


    # Calls procedures which handle the movement of the car sprite and update its parameters
    def _update_car_values(self, steer_input, throttle_and_braking_input):
        self._update_steer_value(steer_input)
        self._update_throttle_and_braking_value(throttle_and_braking_input)
        self._update_car_vector_values()
        self._update_car_sprite_position()

    # Increments the steer value.
    # Steer percentage is the value on the steer slider (determines the limit to which the steer can be incremented).
    # The steer_factor is the value by which steer is incremented (gradual changes rather than abrupt updates).
    # Direction of car updated.
    def _update_steer_value(self, steer_percentage):
        steer_limit = r.max_steer * abs(steer_percentage) - r.steer_factor
        self.steer += r.steer_factor * r.sign(steer_percentage)
        self.steer = r.clamp_value(self.steer, -steer_limit, steer_limit)
        self.direction += self.steer
        self.direction = r.wrap_value(self.direction, 0, 360)

    # Sets the throttle value as a percentage of the maximum driving force.
    def _update_throttle_and_braking_value(self, throttle_pedal_amount):
        if throttle_pedal_amount > 0:
            self.throttle = r.driving_force * throttle_pedal_amount
        else:
            self.throttle = r.braking_force * throttle_pedal_amount

    # Updates the acceleration, velocity and position vectors based on the throttle and direction of the car.
    # Velocity gradually decays when throttle = 0 (roughly emulates friction).
    # Grip variable and lerp function are used to smoothly update car position each frame (arbitrary values).
    """This method would be different when modelling forces."""
    def _update_car_vector_values(self):
        dt = 1/ r.FRAME_RATE
        direction_unit_vector = pygame.Vector2(0, 1).rotate(self.direction)
        self.acceleration = self.throttle * direction_unit_vector
        if self.throttle == 0:
            self.velocity *= 0.996
        else:
            self.velocity += self.acceleration * dt
        if self.velocity.length_squared() != 0:
            self.velocity.clamp_magnitude_ip(r.max_speed)
        grip = 0.5
        speed = self.velocity.magnitude()
        self.velocity = self.velocity.lerp(direction_unit_vector * speed, grip)
        self.position += self.velocity * dt

    # Updates the coordinates of the corners of the hitbox w.r.t the position and orientation of the car
    def _update_hitboxes(self):
        center_x, center_y = self.rect.center
        width, length = r.car_proportions / 2
        width -= 5
        corners = [(-width, length), (width, length), (width, -length), (-width, -length)]

        for i in range(4):
            x, y = corners[i]
            angle = math.radians(self.direction)
            corners[i] = (center_x + x * math.cos(angle) - y * math.sin(angle),
                          center_y + x * math.sin(angle) + y * math.cos(angle))
        self.hitbox[:] = corners

    def get_progress(self, track_spine):
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
        self.steering_wheel_amount = 0
        self.throttle_and_braking_pedal_amount = 0

        self.steering_slider = elements.UIHorizontalSlider(
            relative_rect=pygame.Rect((r.SCREEN_DIMENSIONS[0] / 2, 700), (600, 30)),
            start_value=0,
            value_range=(-100, 100),
            manager=r.GUI_MANAGER
        )

        self.throttle_and_braking_meter = elements.UIProgressBar(
            relative_rect=pygame.Rect((r.SCREEN_DIMENSIONS[0] / 4 - 300, 700), (500, 30)),
            manager=r.GUI_MANAGER
        )
        self.speedometer = UIGaugeMeter(
            relative_rect=pygame.Rect((r.SCREEN_DIMENSIONS[0] / 4 - 300, 600), (200, 100)),
            manager=r.GUI_MANAGER
        )

    def car_movement(self):
        self._handle_steering()
        self._handle_throttle()
        self._update_car_values(self.steering_wheel_amount, self.throttle_and_braking_pedal_amount)


    # Updates the steer value when the steering slider is moved.
    def _handle_steering(self):
        self.steering_wheel_amount = max_steer * self.steering_slider.get_current_value()/100

    # Increments the throttle or braking based on user input.
    # Value decays when there is no input to emulate release of throttle/brake.
    """
    For accuracy, the throttle would have to be handled in a similar way as the steering using a slider of some sort
    to allow more control over the magnitude, since by using keys to control throttle, there is no way to steadily hold 
    the throttle partially pressed down. However for testing purposes, this is ideal since its easier to control than
    having 2 separate sliders
    """
    def _handle_throttle(self):
        if pygame.K_w in r.PRESSED_KEYS:
            self.throttle_and_braking_pedal_amount = r.clamp_value(self.throttle_and_braking_pedal_amount
                                                                   + r.throttle_factor, 0, 1)
        elif pygame.K_s in r.PRESSED_KEYS:
            self.throttle_and_braking_pedal_amount = r.clamp_value(self.throttle_and_braking_pedal_amount
                                                                   - r.brake_factor, -1, 0)
        else:
            self.throttle_and_braking_pedal_amount *= 0.9
            if -0.01 < self.throttle_and_braking_pedal_amount < 0.01:
                self.throttle_and_braking_pedal_amount = 0
        throttle_normalised_value = (self.throttle_and_braking_pedal_amount + 1) * 50
        self.throttle_and_braking_meter.set_current_progress(throttle_normalised_value)
        self.speedometer.update_value(self.velocity.magnitude())


class AICar(Car):
    def __init__(self, starting_position = None):
        super().__init__(starting_position)

    def car_movement(self, actions):
        self._update_car_values(*actions)

    # Calls ray cast method of track to get point of collision and normalised distance (w.r.t max ray length)
    # Stores distances as sensor data
    # Adds coordinates of start and end point of each ray to debugger
    def ray_cast(self, track):
        center = pygame.Vector2(self.rect.center)
        rays = []

        forward_ray = pygame.Vector2(0, 1).rotate(self.direction) * r.CAR_MAX_RAY_CAST
        rays.append((center, center + forward_ray))
        for angle in r.RAY_CAST_ANGLES:
            rays.append((center, center + forward_ray.rotate(angle)))
            rays.append((center, center + forward_ray.rotate(-angle)))

        collided_rays = []
        sensors = []
        for ray in rays:
            hit_point, normalised_collision_distance = track.ray_cast(ray)
            collided_rays.append((center, hit_point))
            sensors.append(normalised_collision_distance)
        r.DEBUG_ELEMENTS["rays"] = collided_rays
        return sensors

    def compute_reward(self):
        reward = 0

        if self.is_crashed:
            reward -= 10
        else:
            reward -= 0.0005

            reward += (self.velocity.magnitude() / r.max_speed) * 0.6
            reward += self.progress * 2

        return reward


