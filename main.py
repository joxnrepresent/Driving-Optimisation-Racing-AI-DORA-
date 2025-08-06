import pygame
import pygame_gui
import resources as r
from car import Car
from user_interface import UIGaugeMeter
from car_simulation import run_car_simulation
from track_drawing_interface import run_track_drawing_interface

"""Global variables"""
pygame.init()
r.create_gui_manager()

"""This is the game loop which runs indefinitely till program is closed"""
def game_loop():
    while True:
        check_events()
        # run_car_simulation()
        run_track_drawing_interface()
        update_frame()

"""This handles user input in keystrokes and interaction with GUI. Keystrokes are recording using a 
'PRESSED_KEYS' set. The GUI manager processes events with respect to the GUI elements on screen"""
def check_events():
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            exit()
        if event.type == pygame.KEYDOWN:
            r.PRESSED_KEYS.add(event.key)
        if event.type == pygame.KEYUP:
            r.PRESSED_KEYS.remove(event.key)
        r.GUI_MANAGER.process_events(event)

"""This updates all the GUI and game elements and draws them onto the screen, then ticks
 the clock to prepare to process the next frame"""
def update_frame():
    r.GUI_MANAGER.update(1/r.FRAME_TIME)
    r.GAME_ELEMENTS.update()
    r.MAIN_SCREEN.fill(r.SCREEN_FILL)
    r.GAME_ELEMENTS.draw(r.MAIN_SCREEN)
    r.GUI_MANAGER.draw_ui(r.MAIN_SCREEN)
    pygame.display.flip()
    r.CLOCK.tick(1/r.FRAME_TIME)

game_loop()

