import pygame
import math
import pygame_gui

"""This module contains global variables and methods that perform basic commonly used functions"""


game_elements = pygame.sprite.Group()
game_mode = 0
GUI_MANAGER = None
def create_gui_manager():
    global GUI_MANAGER
    GUI_MANAGER = pygame_gui.UIManager((1500, 1000))

"""Sets image of a surface using the file name (png only)"""
def set_image(image):
    return pygame.image.load(f'Images/{image}.png').convert_alpha()

"""Returns 1 if x is positive, 0 if x is 0 and -1 if x is negative"""
def sign(x):
    if x > 0:
        return 1
    elif x < 0:
        return -1
    else:
        return 0

"""Keeps value within a range with an upper and lower limit"""
def clamp_value(value, lower_limit, upper_limit):
    return max(lower_limit, min(value, upper_limit))

"""Wraps a value by looping to the lower limit when the upper limit is crossed"""
def wrap_value(value, lower_limit, upper_limit):
    return ((value - lower_limit) % (upper_limit-lower_limit)) + lower_limit



