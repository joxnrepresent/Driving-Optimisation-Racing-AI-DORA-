import math
from tkinter.ttk import Button

import pygame
import pygame_gui
import resources as r
from gui_custom_elements import UIGaugeMeter
from gui_custom_elements import SpatialHashGrid
from gui_custom_elements import Track
from cars import Car

"""
This module is for running a program that creates an instance of the Car class that can be controlled by the user. 
A slider controls the  steering, 'W' for throttle and 'S' for braking. The meter on the bottom left corner represents 
the magnitude of throttle/braking (max throttle = 100, max braking = 0, neutral - 0). The speedometer represents the 
magnitude of the velocity
"""
#---------------------------------------------------------------------------------------------------------------------#


"""
Running car simulation
"""
# Variables for Car simulation
# -----------------------------------#
throttle_and_braking = 0
steer = 0
car = None
steering_slider = None
throttle_and_braking_meter = None
speedometer = None
track = None
reset_button = None

# Initialises scene if it is the first time simulation is being run.
# Calls the procedure that handles car movement
# -----------------------------------#
def run_car_simulation():
    if not r.IS_INITIALIZED:
        initialise_car_simulation()
        print("car simulation initialization complete")
        r.IS_INITIALIZED = True
    handle_buttons()
    car_movement()

# Initialises throttle_and_braking and steer variables, instance of Car class and all GUI elements.
# Game mode is set to 1 to indicate completion of initialisation
# -----------------------------------#
def initialise_car_simulation():
    global throttle_and_braking, steer, car, steering_slider, throttle_and_braking_meter, speedometer, track, reset_button
    throttle_and_braking = 0
    steer = 0
    car = Car((550, 130))
    r.GAME_SPRITES.add(car)
    r.DEBUG_ELEMENTS["hitboxes"].append(car.hitbox)
    steering_slider = pygame_gui.elements.UIHorizontalSlider(
        relative_rect=pygame.Rect((r.SCREEN_DIMENSIONS[0] / 2, 700), (600, 30)),
        start_value=0,
        value_range=(-100, 100),
        manager=r.GUI_MANAGER
    )
    throttle_and_braking_meter = pygame_gui.elements.UIProgressBar(
        relative_rect=pygame.Rect((r.SCREEN_DIMENSIONS[0] / 4 - 300, 700), (500, 30)),
        manager=r.GUI_MANAGER
    )
    speedometer = UIGaugeMeter(
        relative_rect=pygame.Rect((r.SCREEN_DIMENSIONS[0] / 4 - 300, 600), (200, 100)),
        manager=r.GUI_MANAGER
    )

    reset_button = pygame_gui.elements.UIButton(
        relative_rect=pygame.Rect((300, 600), (100, 30)),
        text="Click to reset",
        manager=r.GUI_MANAGER
    )
    track = SpatialHashGrid(
        relative_rect=pygame.Rect((0, 0), (r.SCREEN_DIMENSIONS[0], r.SCREEN_DIMENSIONS[1])),
        manager=r.GUI_MANAGER,
        track_name= r.CURRENT_TRACK
    )

# Calls procedures to update the throttle and steering values.
# Updates the car's properties with respect to these values
# -----------------------------------#
def car_movement():
    handle_steering()
    handle_throttle()
    car.move_car_sprite(steer, throttle_and_braking)
    car.ray_cast(track)
    car.collision_detection(track)

# Updates the steer value when the steering slider is moved.
# -----------------------------------#
def handle_steering():
    global steer
    steer = r.max_steer * steering_slider.get_current_value()/100

# Increments the throttle or braking based on user input.
# Value decays when there is no input to emulate release of throttle/brake.
"""
For accuracy, the throttle would have to be handled in a similar way as the steering using a slider of some sort
to allow more control over the magnitude, since by using keys to control throttle, there is no way to steadily hold 
the throttle partially pressed down. However for testing purposes, this is ideal since its easier to control than
having 2 separate sliders
"""
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

def handle_buttons():
    if reset_button in r.PRESSED_BUTTONS:
        reinitialise_car_simulation()

def reinitialise_car_simulation():
    r.IS_INITIALIZED = False
    r.GAME_SPRITES.empty()
    r.GUI_MANAGER.clear_and_reset()
    r.DEBUG_ELEMENTS.clear()
    r.PRESSED_BUTTONS.remove(reset_button)