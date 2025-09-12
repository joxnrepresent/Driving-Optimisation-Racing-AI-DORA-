import pygame
import pygame_gui
import resources as r
from game_modes import run_car_simulation, run_track_maker

"""
Main module for running the program. This handles input, updates, and rendering
"""
#---------------------------------------------------------------------------------------------------------------------#

# pygame and GUI initialisation
pygame.init()
r.create_gui_manager()

# Game loop which runs indefinitely till program is closed
def game_loop():
    while True:
        if r.GAME_MODE == "track maker":
            run_track_maker()
        elif r.GAME_MODE == "car simulation":
            run_car_simulation()
        check_events()
        update_frame()

# Handles user input in keystrokes and interaction with GUI.
# The GUI manager processes events with respect to the GUI elements on screen
# Keystrokes and pressed buttons are appended to respective hash sets
def check_events():
    for event in pygame.event.get():
        r.GUI_MANAGER.process_events(event)
        if event.type == pygame.QUIT:
            exit()
        if event.type == pygame.KEYDOWN:
            r.PRESSED_KEYS.add(event.key)
        if event.type == pygame.KEYUP:
            r.PRESSED_KEYS.remove(event.key)
        if event.type == pygame.USEREVENT and  event.user_type == pygame_gui.UI_BUTTON_PRESSED:
            r.PRESSED_BUTTONS.add(event.ui_element)

# Renders all game and GUI elements onto the screen
# If debugger is on, then also renders hitboxes, rays, and gridlines
def render():
    r.MAIN_SCREEN.fill(r.SCREEN_FILL)
    if r.IS_DEBUGGING:
        if r.DEBUG_ELEMENTS:
            for element_type, elements in r.DEBUG_ELEMENTS.items():
                for element in elements:
                    if element_type == "hitboxes":
                        pygame.draw.polygon(r.MAIN_SCREEN, "red", element)
                    elif element_type == "rays":
                        try:
                            pygame.draw.line(r.MAIN_SCREEN, "red", element[0], element[1])
                        except:
                            print(element)
                    elif element_type == "grid lines":
                        if element:
                            r.draw_grid(r.MAIN_SCREEN)
    r.GAME_SPRITES.draw(r.MAIN_SCREEN)
    r.GUI_MANAGER.draw_ui(r.MAIN_SCREEN)

# Handles updating frame and internal clock
def update_frame():
    r.GUI_MANAGER.update(1/r.FRAME_TIME)
    r.GAME_SPRITES.update()
    render()
    pygame.display.flip()
    r.CLOCK.tick(1/r.FRAME_TIME)

game_loop()

