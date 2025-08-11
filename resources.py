import pygame
import math
import pygame_gui
from pygame import Vector2

"""This module contains shared resources like constants, game-state variables (singletons), and utility 
functions/methods used throughout the project."""
#---------------------------------------------------------------------------------------------------------------------#

#-----------------------------------#
"""Singletons"""
#-----------------------------------#

# Program constants
#-----------------------------------#
FRAME_TIME = 1/60
SCREEN_DIMENSIONS = (1200, 750)
SCREEN_FILL = "White"

# Core program components
#-----------------------------------#
MAIN_SCREEN = pygame.display.set_mode(SCREEN_DIMENSIONS)
GUI_MANAGER = None
CLOCK = pygame.time.Clock()
PRESSED_KEYS = set()
GAME_ELEMENTS = pygame.sprite.Group()
GAME_MODE = 0

# Car properties (May vary in later versions)
#-----------------------------------#
car_mass = 800
max_steer = 1.8
driving_force = 50
braking_force = 120
starting_orientation = 180
steer_factor = max_steer/10
throttle_factor = 0.08
brake_factor = 0.01
car_proportions = Vector2(27.18, 42.87)

# -----------------------------------#
"""Utility functions"""
# -----------------------------------#

# Initialisation of GUI manager
# -----------------------------------#
def create_gui_manager():
    global GUI_MANAGER
    GUI_MANAGER = pygame_gui.UIManager((1500, 1000))

# Loads an image using the file name (png only)
# -----------------------------------#
def set_image(image):
    return pygame.image.load(f'Images/{image}.png').convert_alpha()

# Returns the sign of the input
# -----------------------------------#
def sign(x):
    if x > 0:
        return 1
    elif x < 0:
        return -1
    else:
        return 0

# Keeps input value within a range with an upper and lower limit
# -----------------------------------#
def clamp_value(value, lower_limit, upper_limit):
    return max(lower_limit, min(value, upper_limit))

# Wraps a value by looping to the lower limit when the upper limit is crossed
# -----------------------------------#
def wrap_value(value, lower_limit, upper_limit):
    return ((value - lower_limit) % (upper_limit-lower_limit)) + lower_limit



