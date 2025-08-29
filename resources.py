import pygame
import math
import pygame_gui
from pygame import Vector2

"""
This module contains shared resources like constants, game-state variables (singletons), and utility 
functions/methods used throughout the project.
"""
#---------------------------------------------------------------------------------------------------------------------#

"""Singletons"""
# -----------------------------------#

# Program constants
# -----------------------------------#
FRAME_TIME = 1/60
SCREEN_DIMENSIONS = (1200, 750)
SCREEN_FILL = "White"
DEBUG = True
CURRENT_TRACK = "testing_track"

# Core program components
# -----------------------------------#
MAIN_SCREEN = pygame.display.set_mode(SCREEN_DIMENSIONS)
GUI_MANAGER = None
CLOCK = pygame.time.Clock()
PRESSED_KEYS = set()
PRESSED_BUTTONS = set()
GAME_SPRITES = pygame.sprite.Group()
DEBUG_ELEMENTS = []
GAME_MODE = "track maker"
IS_INITIALIZED = False

# Car properties (May vary in later versions)
# -----------------------------------#
car_mass = 800
max_steer = 1.4
driving_force = 30
braking_force = 100
starting_orientation = 270
steer_factor = max_steer/10
throttle_factor = 0.08
brake_factor = 0.01
car_proportions = Vector2(27.18, 42.87)

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

def load_track_from_file(filename):
    try:
        with open(filename + ".txt", "r") as file:
            strokes = []
            for line in file:
                coordinates = line.split(',')
                points = []
                for i in range(0, len(coordinates) - 1, 2):
                    points.append((float(coordinates[i]), float(coordinates[i + 1])))
                strokes.append(points)
            return strokes
    except FileNotFoundError:
        print("File not found")
        return None

def draw_track(surface, strokes, colour = "Black"):
    for stroke in strokes:
        for i in range(len(stroke) - 1):
            pygame.draw.line(surface, colour, stroke[i], stroke[i + 1], 3)

