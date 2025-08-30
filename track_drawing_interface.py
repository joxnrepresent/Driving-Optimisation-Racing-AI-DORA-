import math
import pygame
from pygame import Vector2, draw, Rect
import pygame_gui
from sklearn.decomposition import non_negative_factorization

import resources as r
from gui_custom_elements import TrackCanvas


def run_track_maker():
    if not r.IS_INITIALIZED:
        initialise_track_maker()
        print("track maker initialization complete")
        r.IS_INITIALIZED = True
    start_drawing()

start_drawing_button = None
save_button = None
load_button = None
clear_button = None
start_car_sim_button = None
canvas = None

def initialise_track_maker():
    global start_drawing_button, save_button, load_button, clear_button, start_car_sim_button, canvas
    instructions_label = pygame_gui.elements.UILabel(
        relative_rect= pygame.Rect((r.SCREEN_DIMENSIONS[0]//2 - 250, 100),(500, 30)),
        text="Click to add control points. Hold SPACE to draw a straight",
        manager=r.GUI_MANAGER,
    )
    canvas = TrackCanvas(
        relative_rect=pygame.Rect((0, 0),(r.SCREEN_DIMENSIONS[0], r.SCREEN_DIMENSIONS[1])),
        manager= r.GUI_MANAGER
    )

    start_drawing_button = pygame_gui.elements.UIButton(
        relative_rect= pygame.Rect((0, 100), (100, 30)),
        text = "Click to draw",
        manager=r.GUI_MANAGER
    )

    save_button = pygame_gui.elements.UIButton(
        relative_rect= pygame.Rect((100, 100), (100, 30)),
        text = "Click to save",
        manager=r.GUI_MANAGER
    )

    load_button = pygame_gui.elements.UIButton(
        relative_rect= pygame.Rect((200, 100), (100, 30)),
        text = "Click to load",
        manager=r.GUI_MANAGER
    )

    clear_button = pygame_gui.elements.UIButton(
        relative_rect= pygame.Rect((300, 100), (100, 30)),
        text = "Click to clear",
        manager=r.GUI_MANAGER
    )

    start_car_sim_button = pygame_gui.elements.UIButton(
        relative_rect= pygame.Rect((300, 200), (100, 30)),
        text = "Click to start sim",
        manager=r.GUI_MANAGER
    )

def start_drawing():
    if start_drawing_button in r.PRESSED_BUTTONS:
        canvas.is_drawing = not canvas.is_drawing
        if canvas.is_drawing:
            start_drawing_button.set_text("Click to stop")
        else:
            start_drawing_button.set_text("Click to draw")
        r.PRESSED_BUTTONS.remove(start_drawing_button)
    if save_button in r.PRESSED_BUTTONS:
        canvas.save_drawing("testing_track")
        r.PRESSED_BUTTONS.remove(save_button)
    if load_button in r.PRESSED_BUTTONS:
        canvas.load_drawing("testing_track")
        r.PRESSED_BUTTONS.remove(load_button)
    if clear_button in r.PRESSED_BUTTONS:
        canvas.clear()
        r.PRESSED_BUTTONS.remove(clear_button)
    if start_car_sim_button in r.PRESSED_BUTTONS:
        r.GAME_MODE = "car simulation"
        r.IS_INITIALIZED = False
        r.GAME_SPRITES.empty()
        r.DEBUG_ELEMENTS.clear()
        r.GUI_MANAGER.clear_and_reset()
        r.PRESSED_BUTTONS.remove(start_car_sim_button)


