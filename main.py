import pygame
import pygame_gui
import resources as r
from game_modes import run_car_simulation, run_track_maker
from resources import game_core

"""
Main module for running the program. This handles input, updates, and rendering
"""
#---------------------------------------------------------------------------------------------------------------------#

# pygame and GUI initialisation
pygame.init()
game_core.gui_manager = pygame_gui.UIManager((1500, 1000))

# Game loop which runs indefinitely till program is closed
def game_loop():
    while True:
        for i in range(game_core.tick_speedup):
            if game_core.game_mode == "track maker":
                run_track_maker()
            elif game_core.game_mode == "car simulation":
                run_car_simulation()
            check_events()
            game_core.game_sprites.update()
            game_core.gui_manager.update(1 / game_core.frame_rate / game_core.tick_speedup)
        update_frame()

# Handles user input in keystrokes and interaction with GUI.
# The GUI manager processes events with respect to the GUI elements on screen
# Keystrokes and pressed buttons are appended to respective hash sets
def check_events():
    for event in pygame.event.get():
        game_core.event_handle(event)

# Renders all game and GUI elements onto the screen
# If debugger is on, then also renders hitboxes, rays, and gridlines

# Handles updating frame and internal clock
def update_frame():
    game_core.render()
    pygame.display.flip()
    game_core.clock.tick(game_core.frame_rate)

game_loop()

