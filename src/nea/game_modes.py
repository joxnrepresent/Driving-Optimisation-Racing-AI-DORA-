import math
import copy
import json
from typing import override

import numpy as np
import pygame
import pygame_gui

import nea.resources as r
from nea.resources import game_core, RaceTimeManager
from nea.gui_custom_elements import (UITrackCanvas, UIFileBrowser,
                                 UIOptionSelector, UIEndScreen, UIModelCreator)
from nea.track import Track
from nea.cars import PlayerCar, AICar
from nea.rl_model import MCACModel, REINFORCEModel
from pygame_gui.elements import UIPanel, UILabel, UIButton
from abc import ABC, abstractmethod

"""
Game mode classes for racing simulation.

Implements different simulation modes: main menu navigation, track creation UI, player racing,
AI racing against player, and AI training environment. Each mode manages its own GUI elements,
event handling, and update logic.

Classes:
    GameMode: Abstract base class for all game modes with pause menu functionality
    MainMenu: Main menu interface for mode selection
    TrackMakerUI: Interactive track creation and editing interface
    CarSimulation: Base class for racing simulations with track loading and lap timing
    SoloCarSim: Single player racing mode
    RacingAI: Player vs AI racing mode
    AITrainingEnvironment: Reinforcement learning training mode with performance tracking
"""

class GameMode(ABC):
    """
    Abstract base class for all game modes. Includes global functionality like pause menu, and handling events and
    rendering.

    Attributes:
        game_sprites (Group): Group of all game sprites
        debug_elements (dict): Container for debugger visuals
        gui_manager (UIManager): Manages all pygame_gui UI elements
        pause_menu_visible (bool): Flag of if pause menu is opened
        pause_menu (UIPanel): Container panel for pause controls
        pause_button (UIButton): Button to toggle pause menu
        pause_label (UILabel): Pause menu title
    """

    def __init__(self):
        """Initialise game mode with pause menu and manager."""
        game_core.TICK_SPEEDUP = 1
        self.game_sprites = game_core.game_sprites
        self.debug_elements = game_core.debug_elements
        self.gui_manager = game_core.gui_manager

        try:
            self.gui_manager.get_theme().load_theme('DORA_theme.json')
        except FileNotFoundError as e:
            print("Theme not found")
            raise e

        self.pause_menu_visible = True

        self.pause_menu = UIPanel(
                            relative_rect=pygame.Rect((50,50), (300, 500)),
                            manager=self.gui_manager,
                            object_id='#control_panel'
                            )

        self.pause_button = UIButton(
                                relative_rect=pygame.Rect((370,50), (50, 50)),
                                text='||',
                                manager=self.gui_manager,
                                object_id='#pause_button'
                                )

        self.pause_label = pygame_gui.elements.UILabel(
            relative_rect=pygame.Rect((10, 10), (280, 30)),
            text='Paused',
            manager=self.gui_manager,
            container=self.pause_menu,
            object_id='#panel_title'
        )

        self._minimise_pause_menu()

    @abstractmethod
    def update(self, *args):
        """Update game mode state. Child classes must implement this."""
        pass

    def _minimise_pause_menu(self):
        """Hides pause menu and resumes simulation."""
        game_core.is_paused = False
        if not self.pause_menu_visible:
            return

        self.pause_menu.hide()
        self.pause_menu_visible = False

        self.pause_button.set_relative_position((50, 50))
        self.pause_button.set_text('||')


    def _display_pause_menu(self):
        """Display pause menu and pause simulation."""

        game_core.is_paused = True
        if self.pause_menu_visible:
            return

        self.pause_menu.show()
        self.pause_menu_visible = True

        self.pause_button.set_relative_position((370, 50))
        self.pause_button.set_text('\u25B6')

    @abstractmethod
    def event_handle(self):
        if self.pause_button in game_core.pressed_buttons:
            if game_core.unexpected_error_msg is not None:
                game_core.set_game_mode(MainMenu)
            if self.pause_menu_visible:
                self._minimise_pause_menu()
            else:
                self._display_pause_menu()

"""
Creates an interface that allows the user to create, save and draw custom tracks.
"""
class MainMenu(GameMode):
    """
    Main menu interface. User can choose to start track maker, solo racing, race AI, or train AI. Program returns to
    this mode by default if user cancels actions or if any unexpected error occurs

    Attributes:
        options_panel (UIPanel): Background panel
        track_maker_button (UIButton): Start track maker
        solo_sim_button (UIButton): Start solo racing simulation
        racing_sim_button (UIButton): Start race against AI model
        ai_sim_button (UIButton): Start AI model training environment
        exit_button (UIButton): Quit program
        info_label (UILabel): Instructions
        error_label (UILabel): Unexpected error message
        ack_error_button (UIButton): Acknowledge error and continue
    """

    def __init__(self):
        """Initialise main menu with buttons and background."""
        super().__init__()

        screen_w, screen_h = game_core.screen_dimensions
        game_core.is_debugging = False

        self.options_panel = UIPanel(
            relative_rect=pygame.Rect(0, 0, screen_w, screen_h),
            manager=self.gui_manager,
            object_id='#main_bg'
        )

        bg_image = r.set_image("main_menu_bg")
        bg_image = pygame.transform.scale(
            bg_image, game_core.screen_dimensions
        )

        self.options_panel.image.blit(bg_image, (0, 0))

        self.track_maker_button = UIButton(
            relative_rect=pygame.Rect(screen_w // 2 - 450, 400, 400, 70),
            text='Tracks Maker',
            manager=self.gui_manager,
            container=self.options_panel,
            object_id='#menu_button_main'
        )

        self.solo_sim_button = UIButton(
            relative_rect=pygame.Rect(screen_w // 2 + 50, 400, 400, 70),
            text='Solo Racing Simulation',
            manager=self.gui_manager,
            container=self.options_panel,
            object_id='#menu_button_main'
        )

        self.racing_sim_button = UIButton(
            relative_rect=pygame.Rect(screen_w // 2 - 450, 550, 400, 70),
            text='Race against AI',
            manager=self.gui_manager,
            container=self.options_panel,
            object_id='#menu_button_main'
        )

        self.ai_sim_button = UIButton(
            relative_rect=pygame.Rect(screen_w // 2 + 50, 550, 400, 70),
            text='AI Training Mode',
            manager=self.gui_manager,
            container=self.options_panel,
            object_id='#menu_button_main'
        )

        self.exit_button = UIButton(
            relative_rect=pygame.Rect(screen_w // 2 - 200, 700, 400, 70),
            text='Exit',
            manager=self.gui_manager,
            container=self.options_panel,
            object_id='#exit_button'
        )

        self.error_label = pygame_gui.elements.UILabel(
            relative_rect=pygame.Rect(
                (game_core.screen_dimensions[0] / 2 - 400, game_core.screen_dimensions[1] / 2 - 50), (800, 50)),
            text='',
            manager=self.gui_manager,
            container=self.options_panel,
            object_id='#error_label'
        )

        self.ack_error_button = pygame_gui.elements.UIButton(
            relative_rect=pygame.Rect(
                (game_core.screen_dimensions[0] / 2 - 50, game_core.screen_dimensions[1] / 2 + 10), (100, 50)),
            text="OK",
            manager=self.gui_manager,
            container=self.options_panel,
            object_id='#exit_button'
        )
        self.error_label.hide()
        self.ack_error_button.hide()

        if game_core.unexpected_error_msg is not None:
            self._show_error(game_core.unexpected_error_msg)

    def event_handle(self):
        """Handle mode selection."""
        if self.track_maker_button in game_core.pressed_buttons:
            game_core.set_game_mode(TrackMakerUI)

        elif self.solo_sim_button in game_core.pressed_buttons:
            game_core.set_game_mode(SoloCarSim)

        elif self.racing_sim_button in game_core.pressed_buttons:
            game_core.set_game_mode(RacingAI)

        elif self.ai_sim_button in game_core.pressed_buttons:
            game_core.set_game_mode(AITrainingEnvironment)

        elif self.ack_error_button in game_core.pressed_buttons:
            self._hide_error()

        # This closes the program permanently
        elif self.exit_button in game_core.pressed_buttons:
            pygame.quit()
            exit()

        super().event_handle()

    def update(self, *args):
        self.gui_manager.update(1 / game_core.FRAME_RATE)

    def _show_error(self, error_msg):
        """
        Display error message if unexpected error occurs.

        Args:
            error_msg (str): Error message to show
        """
        self.error_label.set_text(error_msg)
        self.error_label.show()
        self.ack_error_button.show()

    def _hide_error(self):
        """Clear and hide error message."""
        game_core.unexpected_error_msg = None
        self.error_label.set_text("")
        self.error_label.hide()
        self.ack_error_button.hide()

class TrackMakerUI(GameMode):
    """
    Custom track maker interface.
    Allows the user to draw custom track shape using a bezier curve tool by creating and
    moving anchor points.
    Allows user to toggle control handles for changing curvature
    Width mode allows user to create width points and change track width at these points.
    Allows user to save and load tracks as json files.

    Attributes:
        canvas (UITrackCanvas): UI canvas where the user makes their track.
        anchor_toggle_button (UIButton): Switch to mode for editing anchors and track shape
        width_toggle_button (UIButton): Switch to mode for editing track width
        toggle_handles_button (UIButton): Toggle curvature control handles
        clear_button (UIButton): Clear the canvas
        save_button (UIButton): Save track to file
        load_button (UIButton): Load track from file
        info_label (UILabel): Instructions
        menu_button (UIButton): Returns to main menu
        error_label (UILabel): Displays errors if there are issues with saving/loading tracks
        ack_error_button (UIButton): Acknowledges error
        file_selector (UIFileBrowser): File selector screen for saving and loading
        width_slider (UIHorizontalSlider): Slider for changing track width.
    """

    def __init__(self):
        super().__init__()

        self.canvas = UITrackCanvas(
            relative_rect=pygame.Rect((0,80), (game_core.screen_dimensions[0], game_core.screen_dimensions[1] - 160)),
            manager=self.gui_manager,
            is_handles_enabled=False
        )


        self.anchor_toggle_button = pygame_gui.elements.UIButton(
            relative_rect=pygame.Rect((10, 50), (140, 40)),
            text="Anchor Mode",
            manager=self.gui_manager,
            container=self.pause_menu,
            object_id='#panel_button'
        )
        self.anchor_toggle_button.disable()

        self.width_toggle_button = pygame_gui.elements.UIButton(
            relative_rect=pygame.Rect((150, 50), (140, 40)),
            text="Width Mode",
            manager=self.gui_manager,
            container=self.pause_menu,
            object_id='#panel_button'
        )

        self.toggle_handles_button = pygame_gui.elements.UIButton(
            relative_rect=pygame.Rect((10, 100), (140, 40)),
            text="Enable Handles",
            manager=self.gui_manager,
            container=self.pause_menu,
            object_id='#panel_button'
        )

        self.clear_button = pygame_gui.elements.UIButton(
            relative_rect=pygame.Rect((150, 100), (140, 40)),
            text="Clear Tracks",
            manager=self.gui_manager,
            container=self.pause_menu,
            object_id='#panel_button'
        )

        self.save_button = pygame_gui.elements.UIButton(
            relative_rect=pygame.Rect((10, 150), (140, 40)),
            text="Save",
            manager=self.gui_manager,
            container=self.pause_menu,
            object_id='#panel_button'
        )

        self.load_button = pygame_gui.elements.UIButton(
            relative_rect=pygame.Rect((150, 150), (140, 40)),
            text="Load",
            manager=self.gui_manager,
            container=self.pause_menu,
            object_id='#panel_button'
        )

        self.info_label = pygame_gui.elements.UILabel(
            relative_rect=pygame.Rect((10, 300), (280, 80)),
            text='Click: Add\nBackspace: Delete\nDrag: Move',
            manager=self.gui_manager,
            container=self.pause_menu,
            object_id='#info_label'
        )

        self.menu_button = pygame_gui.elements.UIButton(
            relative_rect=pygame.Rect((10, 420), (280, 60)),
            text="Return to Menu",
            manager=self.gui_manager,
            container=self.pause_menu,
            object_id='#exit_button'
        )

        self.error_label = pygame_gui.elements.UILabel(
            relative_rect=pygame.Rect(
                (game_core.screen_dimensions[0] / 2 - 250, game_core.screen_dimensions[1] / 2 - 100), (500, 100)),
            text='',
            manager=self.gui_manager,
            object_id='#error_label'
        )

        self.ack_error_button = pygame_gui.elements.UIButton(
            relative_rect=pygame.Rect(
                (game_core.screen_dimensions[0] / 2 - 50, game_core.screen_dimensions[1] / 2 + 10), (100, 50)),
            text="OK",
            manager=self.gui_manager,
            object_id='#exit_button'
        )


        self.error_label.hide()
        self.ack_error_button.hide()

        self.file_selector = None
        self.width_slider = None
        self.canvas.anchor_mode = False
        self.canvas.width_mode = False

    def update(self):
        self.gui_manager.update(1 / game_core.FRAME_RATE)
        if self.canvas.mode == "width":
            self._changing_widths()

    def _changing_widths(self):
        """
        Handles user changing track widths.

        Updates width slider position based on where width point is. Updates track width values
        based on slider input.
        """
        if self.canvas.selected_point and self.canvas.selected_point[0] == 'w':
            width_val_index = self.canvas.selected_point[1]
            width_point_index = math.floor(max(min(width_val_index * len(self.canvas.track_spine),
                                                   len(self.canvas.track_spine) - 1), 0))
            point_position = self.canvas.track_spine[int(width_point_index)]
            if self.width_slider is None:
                # Create slider
                start_val = self.canvas.widths_dict[width_val_index]

                self.width_slider = pygame_gui.elements.UIHorizontalSlider(
                    relative_rect=pygame.Rect((point_position[0] -150, point_position[1] + 100),
                                               (300, 30)),
                    start_value=start_val,
                    value_range=(60, 100),
                    manager=self.gui_manager
                )

                self.width_label = pygame_gui.elements.UILabel(
                    relative_rect=pygame.Rect((point_position[0] -30, point_position[1] + 130), (60, 30)),
                    text=str(int(start_val)),
                    manager=self.gui_manager
                )

            else:
                # Set width value from slider value
                val = int(round(self.width_slider.get_current_value()))
                self.canvas.widths_dict[width_val_index] = val
                self.width_label.set_text(str(val))
        else:
            # Remove slider if changing away from width mode.
            if self.width_slider and self.width_label:
                self.width_slider.kill()
                self.width_label.kill()
            self.width_slider = None
            self.width_label = None


    def event_handle(self):
        if self.anchor_toggle_button in game_core.pressed_buttons:
            self._toggle_anchor_mode()

        if self.width_toggle_button in game_core.pressed_buttons:
            self._toggle_width_mode()

        if self.save_button in game_core.pressed_buttons:
            if self.canvas.validity_issues is not None:
                self._display_validity_issues()
            else:
                self._save_track()

        if self.load_button in game_core.pressed_buttons:
            self._load_track()

        if self.clear_button in game_core.pressed_buttons:
            self._reset_canvas()

        if self.menu_button in game_core.pressed_buttons:
            game_core.set_game_mode(MainMenu)

        if self.toggle_handles_button in game_core.pressed_buttons:
            self._toggle_handles()

        if self.ack_error_button in game_core.pressed_buttons:
            self.ack_error_button.hide()
            self.error_label.hide()
        super().event_handle()

    def _toggle_anchor_mode(self):
        self.canvas.mode = "anchor"
        self.canvas.selected_point = None
        self.anchor_toggle_button.disable()
        self.width_toggle_button.enable()

    def _toggle_width_mode(self):
        self.canvas.mode = "width"
        self.canvas.selected_point = None
        self.width_toggle_button.disable()
        self.anchor_toggle_button.enable()

    def _display_validity_issues(self):
        self.error_label.show()
        self.ack_error_button.show()
        if self.canvas.validity_issues == "invalid walls":
            self.error_label.set_text("Tracks has overlapping/ kinked walls")
        elif self.canvas.validity_issues == "incomplete":
            self.error_label.set_text("Tracks is incomplete")
        elif self.canvas.validity_issues == "too short":
            self.error_label.set_text("Tracks is too short")

    def _toggle_handles(self):
        self.canvas.is_handles_enabled = not self.canvas.is_handles_enabled
        self.canvas.selected_point = None
        if self.canvas.is_handles_enabled:
            self.canvas._generate_all_controls()
            self.toggle_handles_button.set_text("Disable Handles")
        else:
            self.canvas.control_points.clear()
            self.toggle_handles_button.set_text("Enable Handles")

    def _reset_canvas(self):
        handle_status = self.canvas.is_handles_enabled
        self.canvas.kill()
        self.canvas = UITrackCanvas(
            relative_rect=pygame.Rect((0, 80),
                                      (game_core.screen_dimensions[0], game_core.screen_dimensions[1] - 160)),
            manager=self.gui_manager,
            is_handles_enabled=handle_status
        )

    def _save_track(self):
        """Open file selector screen for saving track."""
        self.file_selector = UIFileBrowser(pygame.Rect((0, 0), game_core.screen_dimensions),
                                           self.gui_manager,
                                            "Tracks",
                                            "save",
                                           lambda filename: self._write_track_data(filename))

    def _load_track(self):
        """Open file selector screen for loading track."""
        self.file_selector = UIFileBrowser(pygame.Rect((0, 0), game_core.screen_dimensions),
                                           self.gui_manager,
                                            "Tracks",
                                            "load",
                                           lambda filename: self._load_track_data(filename), )

    def _write_track_data(self, filename=game_core.current_track):
        """
        Create data frame with correct formatting and save it to a JSON file. Ensures handles are generated
        before saving.

        Args:
            filename (str): Track name
        """
        if filename is not None:
            if not self.canvas.is_handles_enabled:
                self.canvas._generate_all_controls()
            data = {
                "anchors": [(point.x, point.y) for point in self.canvas.anchor_points],
                "controls": [(point.x, point.y) for point in self.canvas.control_points],
                "widths": {str(k): v for k, v in self.canvas.widths_dict.items()}
            }
            with open(f"Tracks/{filename}.json", "w") as file:
                json.dump(data, file, indent=2)

    def _load_track_data(self, filename=game_core.current_track):
        """
        Load track from JSON file.

        Sets the canvas anchor points, control points and widths attributes to the data stored in the file.
        Displays error if file is corrupted.

        Args:
            filename (str): Track file name
        """
        try:
            if filename is not None:
                (self.canvas.anchor_points, self.canvas.control_points,
                 self.canvas.widths_dict) = r.load_bezier_track(filename)
            else:
                self.error_label.set_text("File not found")
                self.error_label.show()
                self.ack_error_button.show()
        except Exception as e:
            self.error_label.set_text("Error when loading track (File is likely corrupted)")
            self.error_label.show()
            self.ack_error_button.show()
            print(e)



class CarSimulation(GameMode):
    """
    Abstract base class for racing simulations

    Manages track setting, lap setting, car initialisation, collision detection, progress updates, and lap timing.
    Child classes implement car control and update logic.

    Attributes:
        reset_button (UIButton): Restart simulation
        debugger_button (UIButton): Toggle debugger visuals
        menu_button (UIButton): Return to main menu
        track_loader (UIFileBrowser): File screen for track selection
        num_of_laps_setter (UIOptionSelector): Option screen for setting laps
        track (Track): Track sprite
        cars (list[Car]): Cars being simulated
        num_of_cars (int): Number of cars
        is_crashed (list[bool]): List of flags corresponding to if each car being simulated has crashed
        end_screen (UIEndScreen): End of simulation statistics screen
        time_manager (RaceTimeManager): Lap time manager
        starting_orientation (float): Car starting direction
    """


    def __init__(self, num_of_cars):
        super().__init__()
        self.reset_button = pygame_gui.elements.UIButton(
            relative_rect=pygame.Rect((5, 130), (285, 40)),
            text="Reset simulation",
            manager=self.gui_manager,
            container=self.pause_menu,
            object_id='#panel_button'
        )
        self.debugger_button = pygame_gui.elements.UIButton(
            relative_rect=pygame.Rect((5, 350), (285, 60)),
            text="Toggle debugger",
            manager=self.gui_manager,
            container=self.pause_menu,
            object_id='#exit_button'
        )

        self.menu_button = pygame_gui.elements.UIButton(
            relative_rect=pygame.Rect((5, 420), (285, 60)),
            text="Return to Menu",
            manager=self.gui_manager,
            container=self.pause_menu,
            object_id='#exit_button'
        )

        # Tracks setup
        self.track = None
        self.track_loader = None
        self.end_screen = None

        self.time_manager = RaceTimeManager()

        # Car management
        self.cars = []
        self.num_of_cars = num_of_cars
        self.is_crashed = [False] * num_of_cars

        self.num_of_laps_setter = None
        self.initialise_track()
        self.starting_orientation = 0

    def update(self):
        """
        Updates state of all active cars:
        Skips updates if paused or race finished.
        """
        self.gui_manager.update(1 / game_core.FRAME_RATE)

        if game_core.is_paused or self.end_screen is not None:
            return

        if self.track:
            self.track.update()
        for i, car in enumerate(self.cars):

            # Skip crashed cars
            if self.is_crashed[i] or self.time_manager.is_all_laps_finished[i]:
                continue

            car.collision_detection(self.track)
            if car.is_crashed:
                self.debug_elements["hitboxes"].append(self.cars[0].hitbox.copy())
                self.is_crashed[i] = True

            progress, is_lap_completed = car.update_and_get_progress(self.track.track_spine)
            laps_completed = math.floor(progress)
            self.time_manager.update(i, laps_completed, is_lap_completed)
            # if (pygame.K_q in game_core.pressed_keys):
            #     pygame.draw.circle(self.track.image, (255, 0, 0), car.position, 2)

        # If all laps finished, show end screen
        if all(self.time_manager.is_all_laps_finished):
            results = self.get_results()
            self.display_end_screen(results)
            return

    def event_handle(self):
        super().event_handle()
        if self.reset_button in game_core.pressed_buttons:
            self.reinitialise_simulation()
        if self.menu_button in game_core.pressed_buttons:
            game_core.set_game_mode(MainMenu)
        if self.debugger_button in game_core.pressed_buttons:
            game_core.is_debugging = not game_core.is_debugging

    def initialise_track(self):
        """Open file selector for track loading."""
        self.track_loader = UIFileBrowser(pygame.Rect((0, 0), game_core.screen_dimensions),
                                          self.gui_manager,
                                           "Tracks",
                                           "load",
                                          callback=lambda filename: self.set_track(filename),
                                          return_mode= MainMenu)

    def set_track(self, filename):
        """
        Load and initialise track from file. Loads bezier curve and width data and creates Track object.
        Returns to main menu if error.

        Args:
            filename (str): Track file to load (without extension)
        """
        try:
            anchors, controls, widths = r.load_bezier_track(filename, enforce_centering=True)
            self.track = Track(anchors, controls, widths)
            self.game_sprites.add(self.track)
        except Exception as e:
            r.game_core.unexpected_error_msg = f"Error occurred when loading {filename}: (File is likely corrupted)"
            r.game_core.set_game_mode(MainMenu)
            print("Track load failed:", e)


    def initialise_num_of_laps(self):
        """Open dialog for selecting number of laps."""
        self.num_of_laps_setter = UIOptionSelector(pygame.Rect((0, 0), game_core.screen_dimensions),
                         self.gui_manager,
                         title= "Select number of laps",
                         options =["1", "2", "3", "4", "5", "6", "7", "8", "9", "10"],
                         callback=lambda num_of_laps: self.set_num_of_laps(num_of_laps),
                         return_mode=MainMenu)

    def set_num_of_laps(self, num_of_laps):
        """
        Set race length and initialise cars.

        Args:
            num_of_laps (str): Number of laps
        """
        self.time_manager.num_of_laps = int(num_of_laps)
        self.time_manager.initialise_manager(self.num_of_cars)
        self.init_cars()

    @abstractmethod
    def init_cars(self):
        """Initialise cars. Child classes must implement."""
        pass

    @abstractmethod
    def display_end_screen(self, results):
        """
        Handle end screen display of race statistics. Child classes must implement this

        Args:
            results (dict): Race statistics from time manager
        """
        pass

    def reset_gui(self):
        """Reset GUI elements. Child classes must implement this."""
        pass

    def reset_environment(self):
        """Reset time manager for new race."""
        self.time_manager.initialise_manager(self.num_of_cars)

    def reset_cars(self):
        """Reset all cars to starting positions and progress and clear crash states."""
        for car in self.cars:
            car.reset()
        self.is_crashed = [False] * self.num_of_cars

    def get_results(self):
        """
        Get race statistics from time manager.

        Returns:
            dict: Lap times and total times for all cars
        """
        return self.time_manager.get_stats()

    def end_options(self, option):
        if option == "restart":
            self.reinitialise_simulation()
        else:
            game_core.set_game_mode(MainMenu)

    def reinitialise_simulation(self):
        self.reset_gui()
        self.reset_environment()
        self.init_cars()
        self.end_screen = None


class SoloCarSim(CarSimulation):
    """
    Single player racing mode with a single user controlled car.
    Allows player to race alone with HUD displaying speed, throttle, and steering. Supports both keyboard and slider
    steering controls.

    Attributes:
        controls_toggle_button (UIButton): Switch between keyboard and slider steering
    """
    def __init__(self):
        super().__init__(num_of_cars=1)
        """Initialise solo racing mode with player controls."""
        self.controls_toggle_button = pygame_gui.elements.UIButton(
            relative_rect=pygame.Rect((5, 80), (285, 40)),
            text="Enable steering slider",
            manager=self.gui_manager,
            container=self.pause_menu,
            object_id='#panel_button'
        )

    def update(self, *args):
        super().update()
        if game_core.is_paused or self.end_screen is not None:
            return
        if self.cars:
            self.cars[0].update()
        if self.is_crashed[0]:
            self.reset_cars()
            self.cars[0].progress = math.floor(self.cars[0].progress)

    def event_handle(self):
        if self.controls_toggle_button in game_core.pressed_buttons:
            self.cars[0].toggle_slider()
            if self.cars[0].is_slider_enabled:
                self.controls_toggle_button.set_text("Disable steer slider")
            else:
                self.controls_toggle_button.set_text("Enable steer slider")

        super().event_handle()

    @override
    def init_cars(self):
        """Create 1 player car at track start position."""
        if self.cars:
            self.game_sprites.remove(self.cars[0])
            self.cars[0].delete_control_panel()
        self.cars = [PlayerCar(self.track.track_spine[0], starting_orientation=self.track.get_starting_orientation())]
        self.game_sprites.add(self.cars[0])
        self.debug_elements["hitboxes"].append(self.cars[0].hitbox)

    @override
    def display_end_screen(self, results):
        """
        Display completion screen with lap times.

        Args:
            results (dict): Contains "car 0" (i.e player car) lap times list along with total race time
        """
        lap_times = results["car 0"]
        results_content = ""
        for i in range(len(lap_times) - 1):
            results_content += f"Lap {i + 1}: {lap_times[i]} s\n"

        results_content += f"\nTotal Time: {lap_times[-1]}"
        self.end_screen = UIEndScreen(pygame.Rect((0, 0), game_core.screen_dimensions),
                                    self.gui_manager,
                                    callback= lambda option: self.end_options(option),
                                    title="Performance Breakdown",
                                    subtitle="Lap times",
                                    content=results_content,
                                      return_mode= MainMenu
        )

    @override
    def reset_gui(self):
        """Remove and reinitialise player HUD elements."""
        self.cars[0].speedometer.kill()
        self.cars[0].throttle_and_braking_meter.kill()
        self.cars[0].steering_slider.kill()

    @override
    def reset_environment(self):
        super().reset_environment()

    @override
    def reset_cars(self):
        super().reset_cars()


    def set_track(self, filename):
        """
        Load track initialise lap selection.

        Args:
            filename (str): Track file to load
        """
        super().set_track(filename)
        if game_core.unexpected_error_msg is None:
            self.initialise_num_of_laps()

class RacingAI(CarSimulation):
    """
    Player vs AI racing mode.

    Player races against AI opponent controlled by loaded RL model. Displays winner and lap times.
    Cars reset individually on crash.

    Attributes:
       state_size (int): Size of state vector for AI (sensors + speed + steering)
       rl_model (REINFORCEModel): Loaded AI model
       controls_toggle_button (UIButton): Switch between keyboard and slider steering
       model_loader (UIFileBrowser): File selector screen for model selection
    """

    def __init__(self, state_size=15):
        super().__init__(num_of_cars=2)

        self.state_size = state_size

        self.controls_toggle_button = pygame_gui.elements.UIButton(
            relative_rect=pygame.Rect((5, 80), (285, 40)),
            text="Enable steering slider",
            manager=self.gui_manager,
            container=self.pause_menu,
            object_id='#panel_button'
        )

        self.rl_model = None
        self.model_loader = None

    def update(self, *args):
        super().update()
        if game_core.is_paused or self.end_screen is not None:
            return
        for i, car in enumerate(self.cars):
            if self.time_manager.is_all_laps_finished[i]:
                if game_core.sim_time_ms - self.time_manager.lap_times[i][-1] > 60 * 1000:
                    results = self.get_results()
                    self.display_end_screen(results)
                continue
            if self.is_crashed[i]:
                car.reset()
                self.is_crashed[i] = False
            else:
                if isinstance(car, AICar):
                    sensors = car.get_sensors(self.track)
                    state = sensors + [car.velocity.magnitude() / car.MAX_SPEED, car.steer / car.MAX_STEER_RAD]
                    actions = self.rl_model.get_stochastic_actions(state)[0]
                    car.update(actions[0])
                else:
                    car.update()

    def event_handle(self):
        if self.controls_toggle_button in game_core.pressed_buttons:
            self.cars[0].toggle_slider()
            if self.cars[0].is_slider_enabled:
                self.controls_toggle_button.set_text("Disable steer slider")
            else:
                self.controls_toggle_button.set_text("Enable steer slider")
        super().event_handle()

    def set_track(self, filename):
        """
        Load track and initialise RL model selection.

        Args:
            filename (str): Track file to load
        """
        super().set_track(filename)
        if r.game_core.unexpected_error_msg is None:
            self._initialise_rl_model()

    def _initialise_rl_model(self):
        """Open file selector for loading raceable AI model."""
        self.model_loader = UIFileBrowser(pygame.Rect((0, 0), game_core.screen_dimensions),
                                          self.gui_manager,
                                           "Models",
                                           "load raceable",
                                          callback=lambda filename: self._set_model(filename),
                                          return_mode=MainMenu)

    def _set_model(self, filename=None):
        """
        Load AI model from file and extract architecture.

        Args:
            filename (str): Model file to load
        """
        try:
            save_data = np.load(f"Models/{filename}.npy", allow_pickle=True).item()
            hidden_layer_sizes = []
            weights = save_data["actor_params"]["weights"]
            for w in weights[:-1]:
                hidden_layer_sizes.append(w.shape[1])
            self.rl_model = REINFORCEModel(
                input_size=self.state_size,
                activations=save_data.get('actor_activations', ['elu'] * len(hidden_layer_sizes)+ ['linear']),
                output_size=2,
                hidden_layer_sizes=hidden_layer_sizes
            )
            self.rl_model.actor.set_params(save_data['actor_params'])
            self.initialise_num_of_laps()
        except Exception as e:
            game_core.unexpected_error_msg  = f"ERROR LOADING MODEL:{str(e)}"
            game_core.set_game_mode(MainMenu)

    def init_cars(self):
        """Create player car and AI car at track start position."""
        if self.cars:
            self.game_sprites.remove(self.cars[0], self.cars[1])
            self.cars[0].delete_control_panel()
        self.cars = [PlayerCar(self.track.track_spine[0],starting_orientation=self.track.get_starting_orientation()),
                     AICar(self.track.track_spine[0],starting_orientation=self.track.get_starting_orientation())]
        self.game_sprites.add(self.cars[0], self.cars[1])
        self.debug_elements["hitboxes"].extend([self.cars[0].hitbox, self.cars[1].hitbox])

    def display_end_screen(self, results):
        """
        Display winner and comparative lap times.

        Args:
            results (dict): Contains lap times for both player (car 0) and AI (car 1)
        """
        player_lap_times = results["car 0"]
        player_results_content = ""
        for i in range(len(player_lap_times) - 1):
            player_results_content += f"Lap {i+1}: {player_lap_times[i]} \n"
        player_results_content += f"\nTotal Time: {player_lap_times[-1]}"

        ai_lap_times = results["car 1"]
        ai_results_content = ""
        for i in range(len(ai_lap_times) - 1):
            ai_results_content += f"Lap {i+1}: {ai_lap_times[i]} \n"
        ai_results_content += f"\nTotal Time: {ai_lap_times[-1]}"

        if player_lap_times[-1] is not None and ai_lap_times[-1] is not None:
            winner = "Player" if player_lap_times[-1] < ai_lap_times[-1] else "AI"
        else:
            winner = "Player" if ai_lap_times[-1] is None else "AI"

        end_screen_content = (f"Player Lap times: \n" + player_results_content + "\n\n" +
                              f"AI Lap times: \n" + ai_results_content)
        self.end_screen = UIEndScreen(pygame.Rect((0, 0), game_core.screen_dimensions),
                                      self.gui_manager,
                                      callback= lambda option: self.end_options(option),
                                      title=f"{winner} Wins!",
                                      subtitle="Lap times",
                                      content=end_screen_content,
                                      return_mode= MainMenu
                                      )

class AITrainingEnvironment(CarSimulation):
    """
    Reinforcement learning training mode with performance tracking.

    Trains RL models (REINFORCE or MCAC) by simulating multiple AI cars simultaneously.
    Tracks performance metrics, implements model rollback on performance degradation, and displays real-time training
    statistics.

    Attributes:
        tick_speedup_button (UIButton): Adjust simulation speed
        save_ai_button (UIButton): Save current model
        episode_num (int): Current episode number
        actions (ndarray): Actions for all cars
        stuck_timer (list[int]): Number of consecutive frames each car has been idle for
        prev_progress (list[float]): Previous progress for each car
        max_epoch_progress (list[float]): Maximum progress achieved this episode
        historic_progress (list[float]): Rolling window of max progress values
        time_step_rewards_breakdown (list): Reward components per timestep
        epoch_rewards_breakdown (ndarray): Total rewards of current episode
        performance_history (list[float]): Recent performance history
        best_hist_avg (float): Best historical average performance
        best_model_params (dict): Parameters of best model
        rollback_counter (int): Number of times model has been rolled back
        avg_progress (float): Average progress achieved
        max_progress (float): Maximum progress achieved
        avg_lap_time (float): Average lap time
        best_lap_time (float): Best lap time
        model_type (str): Model type
        rl_model (RLModel): RL model being trained
        is_model_raceable (bool): Whether model has achieved sufficient efficiency
        model_type_selector (UIOptionSelector): Model type selection screen
        model_loader (UIFileBrowser): Model loading screen
        hyper_param_setter (UIModelCreator): Hyperparameter configuration screen
        model_saver (UIFileBrowser): Model saving screen
    """

    HISTORY_WINDOW = 10
    ROLLBACK_THRESHOLD = 0.7
    REWARD_SIZE = 8
    STUCK_LIMIT = 1000
    STATE_SIZE = 15

    PROGRESS_THRESHOLD = None
    LAP_TIME_THRESHOLD = None

    def __init__(self, simulation_size=12):
        super().__init__(num_of_cars=simulation_size)
        self.episode_num = 0
        self.episodes_completed_label = pygame_gui.elements.UILabel(
            relative_rect=pygame.Rect((10, 50), (280, 25)),
            text=f'Episode: {self.episode_num}',
            manager=self.gui_manager,
            container=self.pause_menu,
            object_id='#info_label'
        )

        self.speed_label = pygame_gui.elements.UILabel(
            relative_rect=pygame.Rect((10, 80), (280, 25)),
            text=f'Speed: {game_core.TICK_SPEEDUP}X',
            manager=self.gui_manager,
            container=self.pause_menu,
            object_id='#info_label'
        )

        self.tick_speedup_button = pygame_gui.elements.UIButton(
            relative_rect=pygame.Rect((5, 170), (285, 40)),
            text="Adjust Speed",
            manager=self.gui_manager,
            container=self.pause_menu,
            object_id='#panel_button'
        )

        self.save_ai_button = pygame_gui.elements.UIButton(
            relative_rect=pygame.Rect((5, 300), (285, 40)),
            text="Save Model",
            manager=self.gui_manager,
            container=self.pause_menu,
            object_id='#panel_button'
        )

        self.stats_panel = pygame_gui.elements.UIPanel(
                            relative_rect=pygame.Rect((5, game_core.screen_dimensions[1] - 200),
                                                      (game_core.screen_dimensions[0]-5, 200)),
                            manager=self.gui_manager,
                            object_id='#control_panel'
                            )

        self.episodic_log_box = pygame_gui.elements.UITextBox(
            html_text="",
            relative_rect=pygame.Rect((5, 5),(game_core.screen_dimensions[0] / 3-10, 190)),
            manager=self.gui_manager,
            container= self.stats_panel
        )

        self.performance_log_box = pygame_gui.elements.UITextBox(
            html_text="",
            relative_rect=pygame.Rect((game_core.screen_dimensions[0] / 3+10, 5), (game_core.screen_dimensions[0] / 3-10, 190)),
            manager=self.gui_manager,
            container=self.stats_panel
        )
        self.training_log_box = pygame_gui.elements.UITextBox(
            html_text="",
            relative_rect=pygame.Rect((game_core.screen_dimensions[0] * 2/ 3 + 10, 5),
                                      (game_core.screen_dimensions[0] / 3 - 25, 190)),
            manager=self.gui_manager,
            container=self.stats_panel
        )

        # Simulation state
        self.actions = np.zeros([self.num_of_cars, 2], np.float32)

        # Progress tracking
        self.stuck_timer = [0] * self.num_of_cars
        self.prev_progress = [0.0] * self.num_of_cars
        self.max_epoch_progress = [0.0] * self.num_of_cars
        self.historic_progress = [0]
        self.performance_history = []
        self.best_hist_avg = 0
        self.best_model_params = None

        # Reward tracking
        self.time_step_rewards_breakdown = []
        self.epoch_rewards_breakdown = np.zeros([self.num_of_cars, self.REWARD_SIZE], float)

        self.rollback_counter = 0
        self.avg_progress = 0
        self.max_progress = 0
        self.avg_lap_time = None
        self.best_lap_time = None
        self.is_model_raceable = False

        self.performance_log_box.set_text((f"Average progress: {self.avg_progress:.2f}% \n"
                                  f"Best progress: {self.max_progress:.2f}% \n"
                                  f"Average lap time: None \n"
                                  f"Fastest lap time: None \n"
                                  f"Minimum progress threshold:{self.PROGRESS_THRESHOLD} \n" 
                                  f"Maximum lap time threshold:{self.LAP_TIME_THRESHOLD} \n" 
                                  f"Model ready?: {self.is_model_raceable} \n"
                                  f"Number of rollback: {self.rollback_counter} \n"))

        self.episodic_log_data = ["-----------------"]
        self.training_log_box.set_text(("Average return: None \n"
                                   "Average reward: None"
                                   f"Exploration (Std): None \n"))

        self.model_type = None
        self.rl_model = None

        self.model_type_selector = None
        self.model_loader = None
        self.num_of_laps_setter = None
        self.model_saver = None
        self.hyper_param_setter = None

    def update(self, *args):
        super().update()
        if not game_core.is_paused and self.rl_model is not None:
            self._update_ai_step()

        if all(self.is_crashed):
            super().update()
            self.reinitialise_simulation()
            return

    def event_handle(self):
        super().event_handle()
        if self.reset_button in game_core.pressed_buttons:
            self.reinitialise_simulation()
        if self.tick_speedup_button in game_core.pressed_buttons:
            self._toggle_speedup()
        if self.save_ai_button in game_core.pressed_buttons:
            self._initialise_model_saver()

    def set_track(self, filename):
        """
        Load track and initialise model.

       Args:
           filename (str): Track file to load
       """
        super().set_track(filename)
        self._initialise_model_type()

    def _initialise_model_type(self):
        """Open selector screen for selecting model type (REINFORCE or MCAC)."""
        self.model_type_selector = UIOptionSelector(pygame.Rect((0, 0), game_core.screen_dimensions),
                                                    self.gui_manager,
                                                    ["REINFORCE", "Monte Carlo Actor Critic (MCAC)"],
                                                    callback=lambda model_type: self._set_model_type(model_type),
                                                    return_mode=MainMenu)

    def _set_model_type(self, model_type):
        """
        Set model type and proceed to model loading.

        Args:
            model_type (str): Selected model type
        """
        self.model_type = model_type
        self._initialise_rl_model()

    def _initialise_rl_model(self):
        """Open selector screen for loading existing model or creating new one."""
        self.model_loader = UIFileBrowser(pygame.Rect((0, 0), game_core.screen_dimensions),
                                          self.gui_manager,
                                           "Models",
                                           f"load {self.model_type}",
                                          callback=lambda filename: self._set_model(filename))

    def _set_model(self, filename):
        """
        Load model from file or initialise hyperparameter configuration.

        Args:
            filename (str): Model file to load (None to create new model)
        """
        if filename is None:
            self._initialise_hyper_params()
        else:
            try:
                save_data = np.load(f"Models/{filename}.npy", allow_pickle=True).item()
                self._load_model(save_data)
                self.initialise_num_of_laps()
            except Exception as e:
                game_core.unexpected_error_msg = f"ERROR LOADING MODEL:{str(e)}"
                game_core.set_game_mode(MainMenu)

    def _initialise_hyper_params(self):
        """Open hyperparameter configuration screen for new model."""
        self.hyper_param_setter = UIModelCreator(
            pygame.Rect((0, 0), game_core.screen_dimensions),
            self.gui_manager,
            self.model_type,
            callback=lambda config: self._create_model(config),
            return_mode=MainMenu
        )

    def _load_model(self, save_data):
        """
        Load model architecture and parameters from save data.

        Args:
            save_data (dict): Saved model parameters and configuration
        """
        actor_layer_sizes = []
        weights = save_data["actor_params"]["weights"]

        for w in weights[:-1]:
            actor_layer_sizes.append(w.shape[1])

        # Load environment data
        self.is_model_raceable = save_data['is_model_raceable']
        self.episode_num = save_data['episode_num']
        self.num_of_cars = save_data.get('num_of_cars', 12)
        self.best_model_params = copy.deepcopy(save_data['best_model_params'])

        # Create model
        if save_data['model_type'] == 'REINFORCE':
            self.rl_model = REINFORCEModel(
                input_size=self.STATE_SIZE,
                output_size=2,
                hidden_layer_sizes=actor_layer_sizes,
                activations=save_data.get('actor_activations', ['elu'] * len(actor_layer_sizes) + ['linear']),
                gamma=save_data.get('gamma', 0.99),
                entropy_bonus=save_data.get('entropy', 0.02),
                l2_lambda=save_data.get('l2_lambda', 0.01),
                init_log_std=save_data['actor_params']['log_std'][0],
                learning_rate=save_data.get('actor_learning_rate', 0.003),
                beta1=save_data.get('actor_beta1', 0.9),
                beta2=save_data.get('actor_beta2', 0.999)
            )
        else:
            # Critic architecture
            critic_layer_sizes = []
            weights = save_data["critic_params"]["weights"]

            for w in weights[:-1]:
                critic_layer_sizes.append(w.shape[1])

            self.rl_model = MCACModel(
                input_size=self.STATE_SIZE,
                output_size=2,
                actor_layer_sizes=actor_layer_sizes,
                actor_activations=save_data.get('actor_activations', ['elu'] * len(actor_layer_sizes) + ['linear']),
                critic_layer_sizes=critic_layer_sizes,
                critic_activations=save_data.get('critic_activations', ['tanh'] * len(critic_layer_sizes) + ['linear']),
                gamma=save_data.get('gamma', 0.99),
                l2_lambda=save_data.get('l2_lambda', 0.01),
                entropy_bonus=save_data.get('entropy', 0.02),
                actor_init_log_std=save_data['actor_params']['log_std'][0],
                actor_learning_rate=save_data.get('actor_learning_rate', 0.003),
                actor_beta1=save_data.get('actor_beta1', 0.9),
                actor_beta2=save_data.get('actor_beta2', 0.999),
                critic_init_log_std=save_data['critic_params']['log_std'][0],
                critic_learning_rate=save_data.get('critic_learning_rate', 0.003),
                critic_beta1=save_data.get('critic_beta1', 0.9),
                critic_beta2=save_data.get('critic_beta2', 0.999)
            )

        self.rl_model.actor.set_params(save_data['actor_params'])

        if hasattr(self.rl_model, 'critic') and 'critic_params' in save_data:
            self.rl_model.critic.set_params(save_data['critic_params'])


    def _create_model(self, config):
        """
        Create new RL model from configuration.

        Args:
            config (dict): Model hyperparameters and architecture
        """
        try:
            self.num_of_cars = config['num_of_cars']
            if self.model_type == "REINFORCE":
                self.rl_model = REINFORCEModel(input_size=self.STATE_SIZE,
                                               output_size= 2,
                                               gamma=config['gamma'],
                                               entropy_bonus=config['entropy'],
                                               l2_lambda=config['l2_lambda'],
                                               hidden_layer_sizes=config['actor_hidden_layers'],
                                               activations= config['actor_activations'],
                                               learning_rate=config['actor_learning_rate'],
                                               beta1=config['actor_adam_beta1'],
                                               beta2=config['actor_adam_beta2'],
                                               init_log_std=config['actor_init_log_std'])
            else:
                self.rl_model = MCACModel(input_size=self.STATE_SIZE,
                                          output_size=2,
                                          gamma=config['gamma'],
                                          l2_lambda=config['l2_lambda'],
                                          entropy_bonus=config['entropy'],
                                          actor_layer_sizes=config['actor_hidden_layers'],
                                          actor_activations=config['actor_activations'],
                                          actor_init_log_std=config['actor_init_log_std'],
                                          actor_learning_rate=config['actor_learning_rate'],
                                          actor_beta1=config['actor_adam_beta1'],
                                          actor_beta2=config['actor_adam_beta2'],
                                          critic_layer_sizes= config['critic_hidden_layers'],
                                          critic_activations= config['critic_activations'],
                                          critic_init_log_std=config['critic_init_log_std'],
                                          critic_learning_rate= config['critic_learning_rate'],
                                          critic_beta1=config['critic_adam_beta1'],
                                          critic_beta2=config['critic_adam_beta2'])

            self.initialise_num_of_laps()
        except Exception as e:
            game_core.unexpected_error_msg  = f"ERROR LOADING MODEL:{str(e)}"
            game_core.set_game_mode(MainMenu)

    def initialise_num_of_laps(self):
        super().initialise_num_of_laps()

    def init_cars(self):
        self.LAP_TIME_THRESHOLD = len(self.track.track_spine) / 10
        self.PROGRESS_THRESHOLD = 0.75 * self.time_manager.num_of_laps

        """Create AI cars at track start position."""
        if self.cars:
            for car in self.cars:
                self.game_sprites.remove(car)

        self.cars = []
        for i in range(self.num_of_cars):
            car = AICar(self.track.track_spine[0],starting_orientation=self.track.get_starting_orientation())
            self.cars.append(car)
            self.game_sprites.add(car)
            self.debug_elements["hitboxes"].append(car.hitbox)

    def _update_ai_step(self):
        """
        Perform single training step for active cars.
        """
        states = np.zeros((self.num_of_cars, self.STATE_SIZE), np.float32)
        for i, car in enumerate(self.cars):
            if not self.is_crashed[i]:
                # State vector: sensors + current speed + current steering.
                sensors = car.get_sensors(self.track)
                states[i] = sensors + [car.velocity.magnitude() / car.MAX_SPEED, car.steer / car.MAX_STEER_RAD]

        # Get batch actions
        self.actions, pre_squash = self.rl_model.get_stochastic_actions(states)
        rewards = np.zeros(self.num_of_cars, np.float32)

        for i, car in enumerate(self.cars):
            # Iterate through active cars.
            if not self.is_crashed[i]:
                car.update(self.actions[i])
                # Get sensors at new position for reward computation
                sensors = car.get_sensors(self.track)
                rewards[i] = self._process_car_and_get_reward(i, car, sensors)

        self.rl_model.update_trajectory(states, pre_squash, rewards)

    def _process_car_and_get_reward(self, i, car, sensors):
        """
        Process reward after action is taken.

        Args:
            i (int): car index
            car (AICar): car instance
            sensors (list[float]): sensor values
        """
        car.collision_detection(self.track)
        progress, is_lap_finished = car.update_and_get_progress(self.track.track_spine)

        if is_lap_finished:
            laps_completed = math.floor(progress)
            if self.time_manager.lap_times[i][laps_completed - 1] is None:
                self.time_manager.update(i, laps_completed, is_lap_finished)

        is_stuck = self._check_stuck(i, progress)

        reward, rewards_array = car.compute_reward(is_stuck, is_lap_finished, self.prev_progress[i], sensors)
        if car.progress >= self.time_manager.num_of_laps:
            self.is_crashed[i] = True  #
            car.is_crashed = True
            reward += 80
            rewards_array[-1] += 80

        clipped_reward = np.clip(reward, -50, 100 * self.time_manager.num_of_laps)

        # Log rewards
        reward_breakdown = np.array(rewards_array)
        self.epoch_rewards_breakdown[i] += reward_breakdown
        self.time_step_rewards_breakdown.append(reward_breakdown)

        self.max_epoch_progress[i] = max(progress, self.max_epoch_progress[i])
        self.prev_progress[i] = progress

        return clipped_reward

    def _check_stuck(self, i, current_progress):
        """
        Check if car is stuck and update stuck timer.
        """
        if current_progress - self.prev_progress[i] < 0.001:
            self.stuck_timer[i] += 1
        else:
            self.stuck_timer[i] = 0

        if self.stuck_timer[i] > self.STUCK_LIMIT:
            self.cars[i].is_crashed = True
            return True
        return False

    def reset_environment(self):
        super().reset_environment()
        self.stuck_timer = [0] * self.num_of_cars
        self.prev_progress = [0.0] * self.num_of_cars
        self.is_crashed = [False] * self.num_of_cars

    def reinitialise_simulation(self):
        """
        Reinitialise the environemnt, cars and update RL model parameters.
        """
        if len(self.rl_model.states) > 0:
            self.episode_num += 1
            self.episodes_completed_label.set_text(f'Episodes completed: {self.episode_num}')
            episode_log = self._get_episode_log()

            avg_progress = np.mean(self.max_epoch_progress)
            self._update_metrics(avg_progress)
            self._cache_best_params(self.max_epoch_progress)
            episode_log += self._check_rollback()
            self._update_logs(episode_log)

        for car in self.cars:
            car.progress = math.floor(car.progress)

        self.epoch_rewards_breakdown = np.zeros([self.num_of_cars, self.REWARD_SIZE], float)
        self.time_step_rewards_breakdown = []


        if len(self.rl_model.states) > 0 and len(self.rl_model.rewards) > 0:
            training_log_data= self.rl_model.update_params()
            self.training_log_box.set_text(training_log_data)

        self.max_epoch_progress = [0.0] * self.num_of_cars
        super().reinitialise_simulation()


    def _update_logs(self, log_msg):
        self.episodic_log_data.append(log_msg)

        if len(self.episodic_log_data) > 40:
            self.episodic_log_data = self.episodic_log_data[-40:]

        self.episodic_log_box.set_text("<br>".join(self.episodic_log_data))
        if self.episodic_log_box.scroll_bar:
            self.episodic_log_box.scroll_bar.set_scroll_from_start_percentage(1.0)

        rounded_avg_lap_time = round(self.avg_lap_time, 5) if self.avg_lap_time else None
        rounded_best_lap_time = round(self.best_lap_time, 5) if self.best_lap_time else None
        performance_log_data = (f"Average progress: {self.avg_progress:.2f}% \n"
                                f"Best progress: {self.max_progress:.2f}% \n"
                                f"Average lap time: {rounded_avg_lap_time} \n"
                                f"Fastest lap time: {rounded_best_lap_time} \n"
                                f"Minimum progress threshold: {self.PROGRESS_THRESHOLD} \n"
                                f"Maximum lap time threshold: {self.LAP_TIME_THRESHOLD} \n"
                                f"Model ready?: {self.is_model_raceable} \n"
                                f"Number of rollback: {self.rollback_counter} \n")

        self.performance_log_box.set_text(performance_log_data)

    def _get_episode_log(self):
        """
        Get episode log data.
        Returns:
            str: Episode log data.
        """
        print(f"Episode: {self.episode_num}")

        avg_epoch_rewards = np.round(self.epoch_rewards_breakdown.mean(axis=0), 3)
        print(f"Epoch rewards: {avg_epoch_rewards.tolist()}")

        time_step_rewards = np.array(self.time_step_rewards_breakdown)
        avg_time_step_rewards = np.round(time_step_rewards.mean(axis=0), 3)
        print(f"Time step rewards: {avg_time_step_rewards.tolist()}")
        print()
        avg_progress = sum(car.progress for car in self.cars) / len(self.cars)
        episode_duration = (game_core.sim_time_ms  - self.time_manager.start_time)/1000
        return (f"Episode: {self.episode_num} \n"
                f"Episode duration: {episode_duration:.2f} \n"
                f"Avg progress: {avg_progress:.2f} \n"
                f"Avg reward: {sum(avg_epoch_rewards.tolist()):.2f} \n")

    def _check_rollback(self):
        """
        Check for degrading performance, and rollback model to the best parameters if necessary.

        Returns:
            str: Rollback message.
        """
        if len(self.performance_history) < self.HISTORY_WINDOW:
            return ""

        recent_performance = np.mean(self.performance_history[-self.HISTORY_WINDOW:])

        if recent_performance < self.ROLLBACK_THRESHOLD * self.best_hist_avg:
            self.rl_model.actor.set_params(self.best_model_params['actor'])
            if hasattr(self.rl_model, 'critic'):
                self.rl_model.critic.set_params(self.best_model_params['critic'])

            print()
            print("----xx--------xx--------xx--------xx--------xx--------xx--------xx--------xx----")
            print("Model Rolled Back")
            print("----xx--------xx--------xx--------xx--------xx--------xx--------xx--------xx----")
            print()

            self.rollback_counter += 1
            self.rl_model.clear_trajectory()
            self.performance_history.clear()
            return "----xx" * 5 + "\n" + "Model Rolled Back" + "\n" + "----xx" * 5 + "\n \n"
        return  ""

    def _update_metrics(self, avg_total_reward):
        """
        Update performance metrics and update best model if improved.

        Args:
            avg_total_reward (float): Average total reward.
        """
        self.avg_progress = sum(car.progress for car in self.cars) / len(self.cars) * 100
        self.max_progress = max(self.max_progress, self.avg_progress)
        if self.time_manager.get_average_lap_time() is not None:
            self.avg_lap_time = self.time_manager.get_average_lap_time() * game_core.TICK_SPEEDUP
        else:
            self.avg_lap_time = None
        if self.avg_lap_time is not None:
            if self.best_lap_time is not None:
                self.best_lap_time = min(self.best_lap_time, self.avg_lap_time)
            else:
                self.best_lap_time = self.avg_lap_time

        if self.avg_progress is not None and self.avg_lap_time is not None:
            if (self.avg_progress/100 > self.PROGRESS_THRESHOLD and
                    self.avg_lap_time <= self.LAP_TIME_THRESHOLD):
                self.is_model_raceable = True
            else:
                self.is_model_raceable = False
        else:
            self.is_model_raceable = False
        self.performance_history.append(avg_total_reward)

    def _cache_best_params(self, epoch_progresses):
        """
        Save best model parameters to roll back to if historic average has improved.

        Args:
            epoch_progresses (list[float]): Epoch progresses.
        """

        if len(self.performance_history) > 2 * self.HISTORY_WINDOW:
            self.performance_history.pop(0)

        # Calculate relevant historical average
        if len(self.performance_history) >= self.HISTORY_WINDOW:
            relevant_hist = self.performance_history[-self.HISTORY_WINDOW:]
        else:
            relevant_hist = self.performance_history

        historic_avg = np.mean(relevant_hist)

        # Update best model if performance improved
        if historic_avg > self.best_hist_avg:
            self.best_hist_avg = historic_avg
            cache_params = {'actor': self.rl_model.actor.get_params()}
            if hasattr(self.rl_model, 'critic'):
                cache_params['critic'] = self.rl_model.critic.get_params()
            self.best_model_params = copy.deepcopy(cache_params)

        self.historic_progress.append(max(epoch_progresses))
        if len(self.historic_progress) > self.HISTORY_WINDOW:
            self.historic_progress.pop(0)

    def _initialise_model_saver(self):
        self.model_saver = UIFileBrowser(pygame.Rect((0, 0), game_core.screen_dimensions),
                                         self.gui_manager,
                                          "Models",
                                          "save",
                                         callback= lambda filename: self._save_model(filename))

    def reset_cars(self):
        self.stuck_timer = [0] * self.num_of_cars
        self.prev_progress = [0.0] * self.num_of_cars

        for car in self.cars:
            car.reset()
            car.progress = 0

    def _save_model(self, filename=None):
        if filename is None:
            return

        # Extract activation names from layers
        actor_activations = [layer.activation_name for layer in self.rl_model.actor.layers]

        save_data = {
            'is_model_raceable': self.is_model_raceable,
            'model_type': 'MCAC' if hasattr(self.rl_model, 'critic') else 'REINFORCE',
            'episode_num': self.episode_num,
            'best_model_params': self.best_model_params,
            'num_of_cars': self.num_of_cars,

            # Actor configuration
            'actor_params': self.rl_model.actor.get_params(),
            'actor_activations': actor_activations,
            'gamma': self.rl_model.gamma,
            'entropy': self.rl_model.entropy,
            'l2_lambda': self.rl_model.l2_lambda,
            'actor_learning_rate': self.rl_model.actor.alpha,
            'actor_beta1': self.rl_model.actor.beta1,
            'actor_beta2': self.rl_model.actor.beta2
        }

        if save_data['model_type'] == 'MCAC':
            critic_activations = [layer.activation_name for layer in self.rl_model.critic.layers]
            save_data['critic_params'] = self.rl_model.critic.get_params()
            save_data['critic_activations'] = critic_activations
            save_data['critic_learning_rate'] = self.rl_model.critic.alpha
            save_data['critic_beta1'] = self.rl_model.critic.beta1
            save_data['critic_beta2'] = self.rl_model.critic.beta2

        np.save(f"Models/{filename}.npy", save_data, allow_pickle=True)
        self.model_saver = None

    def _toggle_speedup(self):
        """
        Set simulation speedup factor.
        """
        if game_core.TICK_SPEEDUP == 1:
            game_core.TICK_SPEEDUP = 2
        elif game_core.TICK_SPEEDUP == 2:
            game_core.TICK_SPEEDUP = 5
        elif game_core.TICK_SPEEDUP == 5:
            game_core.TICK_SPEEDUP = 10
        else:
            # Cycle value
            game_core.TICK_SPEEDUP = 1

        self.speed_label.set_text(f"Speed: {game_core.TICK_SPEEDUP}X")

    def display_end_screen(self, results):
        pass

