import pygame
import pygame_gui
from pygame import Vector2
from pygame.examples.go_over_there import SCREEN_SIZE

"""This module contains various constants/sets that are used in various contexts, but maintain their value"""

"""Game constants"""
FRAME_TIME = 1/60
CLOCK = pygame.time.Clock()
PRESSED_KEYS = set()
SCREEN_DIMENSIONS = (1200, 750)
MAIN_SCREEN = pygame.display.set_mode(SCREEN_DIMENSIONS)
SCREEN_FILL = "White"

"""Car properties (May vary in later versions)"""
car_mass = 800
max_steer = 1.8
driving_force = 50
braking_force = 120
starting_orientation = 180
steer_factor = max_steer/10
throttle_factor = 0.08
brake_factor = 0.01
car_proportions = Vector2(27.18, 42.87)