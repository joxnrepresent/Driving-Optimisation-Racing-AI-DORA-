import pygame
import pygame_gui
from pygame import Vector2
import resources as r
from gui_custom_elements import UIGaugeMeter

"""This module is for running a program that creates an instance of the Car class that can be controlled by the user. 
A slider controls the  steering, 'W' for throttle and 'S' for braking. The meter on the bottom left corner represents 
the magnitude of throttle/braking (max throttle = 100, max braking = 0, neutral - 0). The speedometer represents the 
magnitude of the velocity"""

#---------------------------------------------------------------------------------------------------------------------#

class Car(pygame.sprite.Sprite):
    """This class is the Car sprite and extends the Sprite class. The movement of the car is controlled by user input.
    The throttle and steer values are used to control the car's movement. The car's acceleration, velocity and position
    are calculated using vector math and arbitrary constants. Movement is not modelling in terms of forces in this version.
    In later versions, forces will be used to model car movement which would allow lesser bugs, more accurate steering and
    implementing drifting physics, however for present testing, this level of abstraction is sufficient"""
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
        # -----------------------------------#
        self._original_car = r.set_image("Mclaren")
        self._original_car = pygame.transform.scale(self._original_car, r.car_proportions * 2)
        self._original_car = pygame.transform.rotate(self._original_car, self.direction)
        self.image = self._original_car.copy()
        self.rect = self.image.get_rect(topleft=starting_position)

    # Calls procedures which handle the movement of the car sprite and update its parameters
    # -----------------------------------#
    def move_car_sprite(self, steer_percentage, throttle_pedal_amount):
        self._apply_steer(steer_percentage)
        self._apply_throttle_and_braking(throttle_pedal_amount)
        self._update_car_vector_values()
        self._update_car_sprite_position()

    # Increments the steer value.
    # Steer percentage is the value on the steer slider (determines the limit to which the steer can be incremented).
    # The steer_factor is the value by which steer is incremented (gradual changes rather than abrupt updates).
    # Direction of car updated.
    # -----------------------------------#
    def _apply_steer(self, steer_percentage):
        steer_limit = r.max_steer * abs(steer_percentage) - r.steer_factor
        self.steer += r.steer_factor * r.sign(steer_percentage)
        self.steer = r.clamp_value(self.steer, -steer_limit, steer_limit)
        self.direction += self.steer
        self.direction = r.wrap_value(self.direction, 0, 360)

    # Sets the throttle value as a percentage of the maximum driving force.
    # -----------------------------------#
    def _apply_throttle_and_braking(self, throttle_pedal_amount):
        if throttle_pedal_amount > 0:
            self.throttle = r.driving_force * throttle_pedal_amount
        else:
            self.throttle = r.braking_force * throttle_pedal_amount

    # Updates the acceleration, velocity and position vectors based on the throttle and direction of the car.
    # Velocity gradually decays when throttle = 0 (roughly emulates friction).
    # Grip variable and lerp function are used to smoothly update car position each frame (arbitrary values).
    """This method would be different when modelling forces."""
    # -----------------------------------#
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

    # Updates the rotation of the car sprite with respect to its direction and the original orientation
    # Updates position (rect) of the car sprite
    # -----------------------------------#
    def _update_car_sprite_position(self):
        self.image = pygame.transform.rotate(self._original_car, -self.direction)
        self.rect = self.image.get_rect(center=(int(self.position.x), int(self.position.y)))


# Variables for Car simulation
# -----------------------------------#
throttle_and_braking = 0
steer = 0
car = None
steering_slider = None
throttle_and_braking_meter = None
speedometer = None

# Initialises scene if it is the first time simulation is being run.
# Calls the procedure that handles car movement
# -----------------------------------#
def run_car_simulation():
    if r.GAME_MODE == 0:
        initialise_car_simulation()
    car_movement()

# Initialises throttle_and_braking and steer variables, instance of Car class and all GUI elements.
# Game mode is set to 1 to indicate completion of initialisation
# -----------------------------------#
def initialise_car_simulation():
    global throttle_and_braking, steer, car, steering_slider, throttle_and_braking_meter, speedometer
    throttle_and_braking = 0
    steer = 0
    car = Car((500, 500))
    r.GAME_ELEMENTS.add(car)
    steering_slider = pygame_gui.elements.UIHorizontalSlider(
        relative_rect=pygame.Rect((r.SCREEN_DIMENSIONS[0] / 2 - 250, 400), (500, 30)),
        start_value=0,
        value_range=(-100, 100),
        manager=r.GUI_MANAGER
    )
    throttle_and_braking_meter = pygame_gui.elements.UIProgressBar(
        relative_rect=pygame.Rect((r.SCREEN_DIMENSIONS[0] / 4 - 250, 670), (500, 30)),
        manager=r.GUI_MANAGER
    )
    speedometer = UIGaugeMeter(
        relative_rect=pygame.Rect((3 * r.SCREEN_DIMENSIONS[0] / 4 - 100, 600), (200, 100)),
        manager=r.GUI_MANAGER
    )
    r.GAME_MODE = 1

# Calls procedures to update the throttle and steering values.
# Updates the car's properties with respect to these values
# -----------------------------------#
def car_movement():
    handle_steering()
    handle_throttle()
    car.move_car_sprite(steer, throttle_and_braking)

# Updates the steer value when the steering slider is moved.
# -----------------------------------#
def handle_steering():
    global steer
    steer = r.max_steer * steering_slider.get_current_value()/100

# Increments the throttle or braking based on user input.
# Value decays when there is no input to emulate release of throttle/brake.
"""For accuracy, the throttle would have to be handled in a similar way as the steering using a slider of some sort
to allow more control over the magnitude, since by using keys to control throttle, there is no way to steadily hold 
the throttle partially pressed down. However for testing purposes, this is ideal since its easier to control than
having 2 separate sliders"""
# -----------------------------------#
def handle_throttle():
    global throttle_and_braking
    if pygame.K_w in r.PRESSED_KEYS:
        throttle_and_braking = r.clamp_value(throttle_and_braking + r.throttle_factor, 0, 1)
    elif pygame.K_s in r.PRESSED_KEYS:
        throttle_and_braking = r.clamp_value(throttle_and_braking - r.brake_factor, -1, 0)
    else:
        throttle_and_braking *= 0.9
        if -0.01 < throttle_and_braking < 0.01:
            throttle_and_braking = 0
    throttle_normalised_value = (throttle_and_braking + 1) * 50
    throttle_and_braking_meter.set_current_progress(throttle_normalised_value)
    speedometer.update_value(car.velocity.magnitude())

