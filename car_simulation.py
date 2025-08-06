import pygame
import pygame_gui
import resources as r
import constants as c
from user_interface import UIGaugeMeter
from car import Car

"""Variables for Car simulation"""
throttle_and_braking = 0
steer = 0
car = None
steering_slider = None
throttle_and_braking_meter = None
speedometer = None

"""This calls initialisation if it is the first time simulation is being run and calls the procedure that handles 
car movement"""
def run_car_simulation():
    if r.game_mode == 0:
        initialise_car_simulation()
    car_movement()

"""This creates a car sprite on screen that can be controlled by the user. The user using a slider to control the 
steering, 'W' for throttle_and_braking and 'S' for braking. The meter on the bottom left corner 
(throttle_and_braking_meter) represents the magnitude of throttle_and_braking/braking 
(max throttle = 100, max braking = 0, neutral - 0). The speedometer represents the magnitude of the velocity"""
def initialise_car_simulation():
    global throttle_and_braking, steer, car, steering_slider, throttle_and_braking_meter, speedometer
    throttle_and_braking = 0
    steer = 0
    car = Car((500, 500))
    r.game_elements.add(car)
    steering_slider = pygame_gui.elements.UIHorizontalSlider(
        relative_rect=pygame.Rect((c.SCREEN_DIMENSIONS[0] / 2 - 250, 400), (500, 30)),
        start_value=0,
        value_range=(-100, 100),
        manager=r.GUI_MANAGER
    )
    throttle_and_braking_meter = pygame_gui.elements.UIProgressBar(
        relative_rect=pygame.Rect((c.SCREEN_DIMENSIONS[0] / 4 - 250, 670), (500, 30)),
        manager=r.GUI_MANAGER
    )
    speedometer = UIGaugeMeter(
        relative_rect=pygame.Rect((3 * c.SCREEN_DIMENSIONS[0] / 4 - 100, 600), (200, 100)),
        manager=r.GUI_MANAGER
    )
    r.game_mode = 1


"""This calls methods to update the throttle and steering values and then updates the car position based on these
values"""
def car_movement():
    handle_steering()
    handle_throttle()
    car.move_car_sprite(steer, throttle_and_braking)

"""This increments the throttle or braking based on user input. Value also decays when there is no input to 
release throttle/brake."""
def handle_throttle():
    global throttle_and_braking
    if pygame.K_w in c.PRESSED_KEYS:
        throttle_and_braking = r.clamp_value(throttle_and_braking + c.throttle_factor, 0, 1)
    elif pygame.K_s in c.PRESSED_KEYS:
        throttle_and_braking = r.clamp_value(throttle_and_braking - c.brake_factor, -1, 0)
    else:
        throttle_and_braking *= 0.9
        if -0.01 < throttle_and_braking < 0.01:
            throttle_and_braking = 0
    throttle_normalised_value = (throttle_and_braking + 1) * 50
    throttle_and_braking_meter.set_current_progress(throttle_normalised_value)
    speedometer.update_value(car.velocity.magnitude())

"""This updates the steer value when the steering slider is moved"""
def handle_steering():
    global steer
    steer = c.max_steer * steering_slider.get_current_value()/100