import math
import pygame
from pygame_gui import UIManager
from collections import defaultdict
from numpy import exp

"""
This module contains shared resources like constants, game-state variables (singletons), and utility 
functions/methods used throughout the project.
"""
# ---------------------------------------------------------------------------------------------------------------------#

"""Singletons"""

# Program constants

FRAME_RATE = 60
TICK_SPEEDUP = 1
SCREEN_DIMENSIONS = (1200, 750)
SCREEN_FILL = "White"
IS_DEBUGGING = True
CURRENT_TRACK = "testing_track"
SHG_CELL_SIZE = 20
CAR_MAX_RAY_CAST = SCREEN_DIMENSIONS[0]
RAY_CAST_ANGLES = [5, 10, 20, 45, 60, 90]
METER_PIXEL_CONVERSION = 10
EPS = 1e-9

# Core program components
MAIN_SCREEN = pygame.display.set_mode(SCREEN_DIMENSIONS)
GUI_MANAGER = None
CLOCK = pygame.time.Clock()
PRESSED_KEYS = set()
PRESSED_BUTTONS = set()
GAME_SPRITES = pygame.sprite.Group()
DEBUG_ELEMENTS = defaultdict(list)
GAME_MODE = "track maker"
IS_INITIALIZED = False

# Car properties (May vary in later versions)
car_mass = 800
max_steer = 1.7
max_speed = 500
driving_force = 60
braking_force = 150
starting_orientation = 270
starting_position = (550, 130)
steer_factor = max_steer / 10
throttle_factor = 0.08
brake_factor = 0.01
car_proportions = pygame.Vector2(2.718, 4.287) * METER_PIXEL_CONVERSION

# Car properties (May vary in later versions)
car_mass = 800
max_steer = 1.7
max_speed = 500
driving_force = 60
braking_force = 150
starting_orientation = 270
starting_position = (550, 130)
steer_factor = max_steer/10
throttle_factor = 0.08
brake_factor = 0.01
car_proportions = pygame.Vector2(2.718, 4.287) * METER_PIXEL_CONVERSION


"""Utility functions"""


# Initialisation of GUI manager
def create_gui_manager():
    global GUI_MANAGER
    GUI_MANAGER = UIManager((1500, 1000))


# Loads an image using the file name (png only)
def set_image(image):
    return pygame.image.load(f'Images/{image}.png').convert_alpha()


# Returns the sign of the input
def sign(x):
    if x > 0:
        return 1
    elif x < 0:
        return -1
    else:
        return 0


# Keeps input value within a range with an upper and lower limit
def clamp_value(value, lower_limit, upper_limit):
    return max(lower_limit, min(value, upper_limit))


# Wraps a value by looping to the lower limit when the upper limit is crossed
def wrap_value(value, lower_limit, upper_limit):
    return ((value - lower_limit) % (upper_limit - lower_limit)) + lower_limit


def load_track_from_file(filename):
    try:
        with open("Tracks/" + filename + ".txt", "r") as file:
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


def draw_track_outline(surface, strokes):
    is_black = True
    for stroke in strokes:
        for i in range(len(stroke) - 1):
            if is_black:
                colour = "black"
            else:
                colour = "red"
            is_black = not is_black
            pygame.draw.line(surface, colour, stroke[i], stroke[i + 1], 3)


def draw_line(surface, colour, p1, p2):
    pygame.draw.line(surface, colour, p1, p2, 2)


def draw_alternating_line_segments(surface, segments, is_black):
    colour = "black" if is_black else "red"
    for (p1, p2) in segments:
        pygame.draw.line(surface, colour, p1, p2, 3)


def draw_grid(surface, color="green"):
    w, h = surface.get_size()
    s = pygame.Surface((w, h), pygame.SRCALPHA)
    for x in range(0, w, SHG_CELL_SIZE):
        pygame.draw.line(s, color, (x, 0), (x, h))
    for y in range(0, h, SHG_CELL_SIZE):
        pygame.draw.line(s, color, (0, y), (w, y))
    surface.blit(s, (0, 0))


def get_line_segments_intersection(seg1, seg2):
    (x1, y1), (x2, y2) = seg1
    (x3, y3), (x4, y4) = seg2

    # Trying to early reject intersection if AABB's do not intersect
    if (max(x1, x2) < min(x3, x4) or max(x3, x4) < min(x1, x2) or
            max(y1, y2) < min(y3, y4) or max(y3, y4) < min(y1, y2)):
        return None

    denominator = (x2 - x1) * (y4 - y3) - (y2 - y1) * (x4 - x3)
    alpha_numerator = (x3 - x1) * (y4 - y3) - (y3 - y1) * (x4 - x3)
    beta_numerator = (x3 - x1) * (y2 - y1) - (y3 - y1) * (x2 - x1)

    if abs(denominator) < EPS:
        if abs(alpha_numerator) < EPS and abs(beta_numerator) < EPS:

            dot_prod = (x2 - x1) * (x4 - x3) + (y2 - y1) * (y4 - y3)

            overlap_start_x = max(min(x1, x2), min(x3, x4))
            overlap_end_x = min(max(x1, x2), max(x3, x4))
            overlap_start_y = max(min(y1, y2), min(y3, y4))
            overlap_end_y = min(max(y1, y2), max(y3, y4))

            if overlap_start_x <= overlap_end_x and overlap_start_y <= overlap_end_y:

                if min(x3, x4) <= x1 <= max(x3, x4) and min(y3, y4) <= y1 <= max(y3, y4):
                    return x1, y1

                p1 = (overlap_start_x, overlap_start_y)
                p2 = (overlap_end_x, overlap_end_y)
                if dot_prod > 0:
                    return p1
                else:
                    return p2
        return None

    alpha = alpha_numerator / denominator
    beta = beta_numerator / denominator

    if alpha >= 0 and 0 <= beta <= 1:
        x = x1 + alpha * (x2 - x1)
        y = y1 + alpha * (y2 - y1)
        return x, y
    else:
        return None


def transformed_sigmoid(x):
    return (2.0 / (1.0 + exp(-x))) - 1