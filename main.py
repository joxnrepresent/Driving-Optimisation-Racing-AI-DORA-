from pygame_gui import UIManager
from game_modes import MainMenu
from resources import game_core
import pygame
"""
Main module for running the program. This handles input, updates, and rendering
"""
#---------------------------------------------------------------------------------------------------------------------#

# pygame and GUI initialisation
pygame.init()
game_core.gui_manager = UIManager(game_core.screen_dimensions)
game_core.set_game_mode(MainMenu)
# Game loop which runs indefinitely till program is closed
def game_loop():
    while True:
        for event in pygame.event.get():
            game_core.cache_events(event)

        for _ in range(game_core.tick_speedup):
            game_core.process_frame()
            game_core.sim_time_ms += game_core.dt_ms / game_core.tick_speedup

        game_core.render()
game_loop()