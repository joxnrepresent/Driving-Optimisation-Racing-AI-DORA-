import pygame
from pygame import Vector2

from resources import DRAG_COEFFICIENT
from resources import ROLLING_RESISTANCE_COEFFICIENT

import resources
class Car:
    def __init__(self, mass, starting_position, starting_orientation, max_steer, driving_force, steer_factor, dt):
        self.mass = mass
        self.max_steer = max_steer
        self.driving_force = driving_force
        self.steer_factor = steer_factor

        self.position = Vector2(starting_position)
        self.velocity = Vector2(0,0)
        self.acceleration = Vector2(0,0)
        self.dt = dt
        self.direction = starting_orientation

        self.throttle = 0
        self.steer = 0.0
        self.brake = 0

        self.original_car = resources.set_image("Car")
        self.original_car = pygame.transform.scale(self.original_car, (80, 112))
        self.original_car = pygame.transform.rotate(self.original_car, self.direction)
        self.car = self.original_car.copy()
        self.car_rect = self.car.get_rect(topleft=starting_position)

    def apply_steer(self, steer_percentage):
        steer_limit = self.max_steer * abs(steer_percentage) - self.steer_factor
        self.steer += self.steer_factor * resources.sign(steer_percentage)
        self.steer = resources.clamp_value(self.steer, -steer_limit, steer_limit)
        self.direction += self.steer
        self.direction = resources.wrap_value(self.direction, 0, 360)

    def apply_throttle_and_braking(self, throttle_pedal_amount):
        self.throttle = self.driving_force * throttle_pedal_amount

    def update_car_vector_values(self):
        direction_unit_vector = Vector2(0, 1).rotate(self.direction)

        traction_force = self.throttle * direction_unit_vector
        drag_force = DRAG_COEFFICIENT * self.velocity * 400 * self.velocity.magnitude()
        rolling_resistance_force = ROLLING_RESISTANCE_COEFFICIENT * self.velocity *20

        self.acceleration = (traction_force - (drag_force + rolling_resistance_force))

        self.velocity += self.acceleration * self.dt
        if self.velocity.length_squared() != 0:
            self.velocity.clamp_magnitude_ip(300)
        grip = 0.5
        speed = self.velocity.magnitude()
        self.velocity = self.velocity.lerp(direction_unit_vector * speed, grip)
        self.position += self.velocity * self.dt

    def update_car_sprite_position(self):
        self.car = pygame.transform.rotate(self.original_car, -self.direction)  # use -orientation for clockwise
        self.car_rect = self.car.get_rect(center=(int(self.position.x), int(self.position.y)))

    def move_car_sprite(self, steer_percentage, throttle_pedal_amount):
        self.apply_steer(steer_percentage)
        self.apply_throttle_and_braking(throttle_pedal_amount)
        self.update_car_vector_values()
        self.update_car_sprite_position()





