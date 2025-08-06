import pygame
from pygame import Vector2
import resources as r

"""This class is the Car sprite and extends the Sprite class. The throttle and steer values are used to control the 
car's movement. The car's acceleration, velocity and position are calculated using vector math and arbitrary constants. 
Movement is not modelling in terms of forces in this version. In later versions, forces will be used to model car 
movement which would allow lesser bugs, more accurate steering and implementing drifting physics, however for present
testing, this level of abstraction is sufficient"""
class Car(pygame.sprite.Sprite):
    def __init__(self, starting_position):
        super().__init__()
        self.position = Vector2(starting_position)
        self.velocity = Vector2(0,0)
        self.acceleration = Vector2(0,0)
        self.direction = r.starting_orientation

        self.throttle = 0
        self.steer = 0.0

        """The '_original_car' variable is necessary for rotating the sprite properly since the car sprite is rotated
        to an angle with respect to the natural orientation of the sprite on the screen."""
        self._original_car = r.set_image("Mclaren")
        self._original_car = pygame.transform.scale(self._original_car, r.car_proportions * 2)
        self._original_car = pygame.transform.rotate(self._original_car, self.direction)
        self.image = self._original_car.copy()
        self.rect = self.image.get_rect(topleft=starting_position)

    """This calls all the procedures that handle the movement of the car sprite and update its values"""
    def move_car_sprite(self, steer_percentage, throttle_pedal_amount):
        self._apply_steer(steer_percentage)
        self._apply_throttle_and_braking(throttle_pedal_amount)
        self._update_car_vector_values()
        self._update_car_sprite_position()

    """This method increments the steer value. The steer percentage is the value on the steer slider and determines
    the limit to which the steer can be incremented up to. The steer_factor is the value by which steer is incremented
    every game loop. This allows for gradual changes rather than abrupt updates. It also updates the direction the
    car is facing."""
    def _apply_steer(self, steer_percentage):
        steer_limit = r.max_steer * abs(steer_percentage) - r.steer_factor
        self.steer += r.steer_factor * r.sign(steer_percentage)
        self.steer = r.clamp_value(self.steer, -steer_limit, steer_limit)
        self.direction += self.steer
        self.direction = r.wrap_value(self.direction, 0, 360)

    """This sets the throttle value as a percentage of the maximum driving force. Ideally this value would be
    continually incremented like the steer value, however controlling throttle using keys is easier for testing."""
    def _apply_throttle_and_braking(self, throttle_pedal_amount):
        if throttle_pedal_amount > 0:
            self.throttle = r.driving_force * throttle_pedal_amount
        else:
            self.throttle = r.braking_force * throttle_pedal_amount

    """Updates the acceleration, velocity and position vectors based on the throttle and direction the car is facing.
    When throttle is 0, velocity gradually decays to roughly emulate friction. The grip variable and lerp function is
    used to smoothly update car position each frame with excessive sliding caused by this vector based approach. 
    This method would be different when modelling forces."""
    def _update_car_vector_values(self):
        direction_unit_vector = Vector2(0, 1).rotate(self.direction)
        self.acceleration = self.throttle * direction_unit_vector
        if self.throttle == 0:
            self.velocity *= 0.996
        else:
            self.velocity += self.acceleration * r.FRAME_TIME
        if self.velocity.length_squared() != 0:
            self.velocity.clamp_magnitude_ip(300)
        grip = 0.5
        speed = self.velocity.magnitude()
        self.velocity = self.velocity.lerp(direction_unit_vector * speed, grip)
        self.position += self.velocity * r.FRAME_TIME

    """This updates the rotation of the car sprite with respect to its direction and the original orientation, and 
    then updates position (rect) of the car sprite"""
    def _update_car_sprite_position(self):
        self.image = pygame.transform.rotate(self._original_car, -self.direction)
        self.rect = self.image.get_rect(center=(int(self.position.x), int(self.position.y)))