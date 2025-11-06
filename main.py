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
            for event in pygame.event.get():
                game_core.event_handle(event)
            game_core.game_sprites.update()
            game_core.gui_manager.update(1 / game_core.frame_rate / game_core.tick_speedup)
        game_core.render()

game_loop()

