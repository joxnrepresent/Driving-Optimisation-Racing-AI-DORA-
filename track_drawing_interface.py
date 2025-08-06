import math
import pygame
from pygame import Vector2, draw, Rect
import pygame_gui
import resources as r
from resources import SCREEN_DIMENSIONS
from user_interface import UIBezierCanvas
def run_track_drawing_interface():
    if r.GAME_MODE == 0:
        initialise_track_drawing_interface()

def initialise_track_drawing_interface():
    instructions_label = pygame_gui.elements.UILabel(
        relative_rect= pygame.Rect((r.SCREEN_DIMENSIONS[0]//2 - 250, 100),(500, 30)),
        text="Click to add control points. Hold SPACE to draw a straight",
        manager=r.GUI_MANAGER,
    )
    canvas = UIBezierCanvas(
        relative_rect=pygame.Rect((0, 200),(SCREEN_DIMENSIONS[0], SCREEN_DIMENSIONS[1])),
        manager= r.GUI_MANAGER
    )
    r.GAME_MODE = 1

