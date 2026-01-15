import pygame
import pygame_gui
from game_modes import MainMenu
from resources import game_core

"""
Main module for running the program. This handles input, updates, and rendering
"""
#----------------------------------------------------f-----------------------------------------------------------------#

# pygame and GUI initialisation
pygame.init()
game_core.gui_manager = pygame_gui.UIManager(game_core.screen_dimensions)
game_core.set_game_mode(MainMenu)

# Game loop which runs indefinitely till program is closed
def game_loop():
    while True:
        for i in range(game_core.tick_speedup):
            for event in pygame.event.get():
                game_core.cache_events(event)
            game_core.process_frame()
        game_core.render()
game_loop()


