from pygame_gui import UIManager
from game_modes import MainMenu
from resources import game_core
import pygame
"""
Main module for running the program.
"""
#---------------------------------------------------------------------------------------------------------------------#

# pygame and GUI initialisation
pygame.init()
game_core.gui_manager = UIManager(game_core.screen_dimensions)
game_core.set_game_mode(MainMenu)

def game_loop():
    """
    Main game loop that runs the program. Repeats indefinitely until the user closes the program.
    """
    while True:
        # Record user events into respective hash sets.
        for event in pygame.event.get():
            game_core.cache_events(event)

        # Update clock: dt_ms is the time between clock updates
        game_core.dt_ms = game_core.clock.tick(game_core.frame_rate)

        # dt_per_step is time per frame processing call -> handles speedups
        dt_per_step = game_core.dt_ms / game_core.tick_speedup

        for _ in range(game_core.tick_speedup):
            """
            When tick speedup is used, many frames of the program are processed and are only rendered once. This helps
            reduce bottleneck caused by rendering.
            """
            game_core.process_frame()
            if not game_core.is_paused:
                game_core.sim_time_ms += dt_per_step

        game_core.render()

game_loop()