import math
import copy
import pickle
import json
import sys
import numpy as np
import pygame
import pygame_gui
import cars
import resources as r
from resources import game_core, RaceTimeManager
from gui_custom_elements import (UIGaugeMeter, UITrackCanvas, UIFileSelector,
                                 UIOptionSelector, UIEndScreen, UIModelCreator)
from track import Track
from cars import PlayerCar, AICar
from rl_model import MCACModel, REINFORCEModel
from pygame_gui.elements import UIPanel, UILabel, UIButton
from abc import ABC, abstractmethod

"""This module contains classes """
#---------------------------------------------------------------------------------------------------------------------#

class GameMode(ABC):
    """Base class for all game modes."""

    def __init__(self):
        game_core.tick_speedup = 1
        self.game_sprites = game_core.game_sprites
        self.debug_elements = game_core.debug_elements
        self.gui_manager = game_core.gui_manager

        try:
            self.gui_manager.get_theme().load_theme('DORA_theme.json')
        except:
            pass

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

        self.minimise_pause_menu()

    def update(self, *args):
        pass

    def minimise_pause_menu(self):

        game_core.is_paused = False
        if not self.pause_menu_visible:
            return


        self.pause_menu.hide()
        self.pause_menu_visible = False

        self.pause_button.set_relative_position((50, 50))
        self.pause_button.set_text('||')


    def display_pause_menu(self):

        game_core.is_paused = True
        if self.pause_menu_visible:
            return


        self.pause_menu.show()
        self.pause_menu_visible = True

        self.pause_button.set_relative_position((370, 50))
        self.pause_button.set_text('\u25B6')


    def event_handle(self):
        if self.pause_button in game_core.pressed_buttons:
            if game_core.unexpected_error_msg is not None:
                game_core.set_game_mode(MainMenu)
            if self.pause_menu_visible:
                self.minimise_pause_menu()
            else:
                self.display_pause_menu()

"""
Creates an interface that allows the user to create, save and draw custom tracks.
"""
class MainMenu(GameMode):
    def __init__(self):
        super().__init__()

        screen_w, screen_h = game_core.screen_dimensions


        self.options_panel = UIPanel(
            relative_rect=pygame.Rect(0, 0, screen_w, screen_h),
            manager=self.gui_manager,
            object_id='#main_bg'
        )

        self.title = UILabel(
            relative_rect=pygame.Rect(screen_w // 2 - 300, 80, 600, 100),
            text='DRIVING OPTIMISATION RACING AI (DORA)',
            manager=self.gui_manager,
            container=self.options_panel,
            object_id='#title_label',
        )

        self.subtitle = UILabel(
            relative_rect=pygame.Rect(screen_w // 2 - 250, 190, 500, 40),
            text='Reinforcement Learning Racing Simulation',
            manager=self.gui_manager,
            container=self.options_panel,
            object_id='#subtitle_label'
        )

        self.track_maker_button = UIButton(
            relative_rect=pygame.Rect(screen_w // 2 - 450, 300, 400, 70),
            text='Tracks Maker',
            manager=self.gui_manager,
            container=self.options_panel,
            object_id='#menu_button_main'
        )

        self.solo_sim_button = UIButton(
            relative_rect=pygame.Rect(screen_w // 2 + 50, 300, 400, 70),
            text='Solo Racing Simulation',
            manager=self.gui_manager,
            container=self.options_panel,
            object_id='#menu_button_main'
        )

        self.racing_sim_button = UIButton(
            relative_rect=pygame.Rect(screen_w // 2 - 450, 420, 400, 70),
            text='Race against AI',
            manager=self.gui_manager,
            container=self.options_panel,
            object_id='#menu_button_main'
        )

        self.ai_sim_button = UIButton(
            relative_rect=pygame.Rect(screen_w // 2 + 50, 420, 400, 70),
            text='AI Training Mode',
            manager=self.gui_manager,
            container=self.options_panel,
            object_id='#menu_button_main'
        )

        self.exit_button = UIButton(
            relative_rect=pygame.Rect(screen_w // 2 - 200, 560, 400, 70),
            text='Exit',
            manager=self.gui_manager,
            container=self.options_panel,
            object_id='#exit_button'
        )

        self.info_label = UILabel(
            relative_rect=pygame.Rect(screen_w // 2 - 300, screen_h - 100, 600, 30),
            text='Select a mode to begin',
            manager=self.gui_manager,
            container=self.options_panel,
            object_id='#info_label'
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
            self.show_error(game_core.unexpected_error_msg)
            game_core.unexpected_error_msg = None


    def event_handle(self):

        if self.track_maker_button in game_core.pressed_buttons:
            game_core.set_game_mode(TrackMakerUI)

        elif self.solo_sim_button in game_core.pressed_buttons:
            game_core.set_game_mode(SoloCarSim)

        elif self.racing_sim_button in game_core.pressed_buttons:
            game_core.set_game_mode(RacingSim)

        elif self.ai_sim_button in game_core.pressed_buttons:
            game_core.set_game_mode(AITrainingEnvironment)

        elif self.ack_error_button in game_core.pressed_buttons:
            self.hide_error()

        elif self.exit_button in game_core.pressed_buttons:
            pygame.quit()
            exit()

        super().event_handle()

    def update(self, *args):
        self.gui_manager.update(1 / game_core.frame_rate)

    def show_error(self, error_msg):
        self.error_label.set_text(error_msg)
        self.error_label.show()
        self.ack_error_button.show()

    def hide_error(self):
        self.error_label.set_text("")
        self.error_label.hide()
        self.ack_error_button.hide()

class TrackMakerUI(GameMode):
    """
    Class stores the variables, objects and methods needed to run the track maker.
    """

    def __init__(self):
        super().__init__()

        self.canvas = UITrackCanvas(
            relative_rect=pygame.Rect((0,80), (game_core.screen_dimensions[0], game_core.screen_dimensions[1] - 160)),
            manager=self.gui_manager
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
        self.gui_manager.update(1 / game_core.frame_rate)
        if self.canvas.mode == "width":
            self.changing_widths()

    def changing_widths(self):
        if self.canvas.selected_point and self.canvas.selected_point[0] == 'w':
            width_val_index = self.canvas.selected_point[1]
            width_point_index = math.floor(max(min(width_val_index * len(self.canvas.track_spine),
                                                   len(self.canvas.track_spine) - 1), 0))
            point_position = self.canvas.track_spine[int(width_point_index)]
            if self.width_slider is None:
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
                val = int(round(self.width_slider.get_current_value()))
                self.canvas.widths_dict[width_val_index] = val
                self.width_label.set_text(str(val))
        else:
            if self.width_slider and self.width_label:
                self.width_slider.kill()
                self.width_label.kill()
            self.width_slider = None
            self.width_label = None


    # Handles button presses:
    def event_handle(self):
        if self.anchor_toggle_button in game_core.pressed_buttons:
            self.canvas.mode = "anchor"
            self.canvas.selected_point = None
            self.anchor_toggle_button.disable()
            self.width_toggle_button.enable()

        if self.width_toggle_button in game_core.pressed_buttons:
            self.canvas.mode = "width"
            self.canvas.selected_point = None
            self.width_toggle_button.disable()
            self.anchor_toggle_button.enable()

        if self.save_button in game_core.pressed_buttons:
            if self.canvas.validity_issues is not None:
                self.error_label.show()
                self.ack_error_button.show()
                if self.canvas.validity_issues == "invalid walls":
                    self.error_label.set_text("Tracks has overlapping/ kinked walls")
                elif self.canvas.validity_issues == "incomplete":
                    self.error_label.set_text("Tracks is incomplete")
                elif self.canvas.validity_issues == "too short":
                    self.error_label.set_text("Tracks is too short")
            else:
                self.save_track()

        if self.load_button in game_core.pressed_buttons:
            self.load_track()

        if self.clear_button in game_core.pressed_buttons:
            self.canvas.kill()
            self.canvas = UITrackCanvas(
                relative_rect=pygame.Rect((0, 80),
                                          (game_core.screen_dimensions[0], game_core.screen_dimensions[1] - 160)),
                manager=self.gui_manager
            )

        if self.menu_button in game_core.pressed_buttons:
            game_core.set_game_mode(MainMenu)

        if self.toggle_handles_button in game_core.pressed_buttons:
            self.canvas.is_handles_enabled = not self.canvas.is_handles_enabled
            self.canvas.selected_point = None
            if self.canvas.is_handles_enabled:
                self.canvas.generate_all_controls()
                self.toggle_handles_button.set_text("Disable Handles")
            else:
                self.canvas.control_points.clear()
                self.toggle_handles_button.set_text("Enable Handles")

        if self.ack_error_button in game_core.pressed_buttons:
            self.ack_error_button.hide()
            self.error_label.hide()

        super().event_handle()

    def save_track(self):
        self.file_selector = UIFileSelector(pygame.Rect((0, 0), game_core.screen_dimensions),
                                            self.gui_manager,
                                            "Tracks",
                                            "save",
                                            lambda filename: self.write_track_data(filename))

    def load_track(self):
        self.file_selector = UIFileSelector(pygame.Rect((0, 0), game_core.screen_dimensions),
                                            self.gui_manager,
                                            "Tracks",
                                            "load",
                                            lambda filename: self.load_track_data(filename),)

    def write_track_data(self, filename=game_core.current_track):
        if filename is not None:
            if not self.canvas.is_handles_enabled:
                self.canvas.generate_all_controls()
            data = {
                "anchors": [(point.x, point.y) for point in self.canvas.anchor_points],
                "controls": [(point.x, point.y) for point in self.canvas.control_points],
                "widths": {str(k): v for k, v in self.canvas.widths_dict.items()}
            }
            with open(f"Tracks/{filename}.json", "w") as file:
                json.dump(data, file, indent=2)

    def load_track_data(self, filename=game_core.current_track):
        try:
            if filename is not None:
                self.canvas.anchor_points, self.canvas.control_points, self.canvas.widths_dict = r.load_bezier_track(filename)
                return True
        except:
            self.error_label.set_text("Error when loading track (File is likely corrupted)")
            self.error_label.show()
            self.ack_error_button.show()
            return False



class CarSimulation(GameMode):
    """Base class for car simulation modes."""

    def __init__(self, num_of_cars):
        super().__init__()
        self.reset_button = pygame_gui.elements.UIButton(
            relative_rect=pygame.Rect((5, 130), (285, 40)),
            text="Reset simulation",
            manager=self.gui_manager,
            container=self.pause_menu,
            object_id='#panel_button'
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

    def initialise_track(self):
        self.track_loader = UIFileSelector(pygame.Rect((0, 0), game_core.screen_dimensions),
                                           self.gui_manager,
                                           "Tracks",
                                           "load",
                                           callback=lambda filename: self.set_track(filename),
                                           return_mode= MainMenu)


    def set_track(self, filename):
        try:
            anchors, controls, widths = r.load_bezier_track(filename, enforce_centering=True)
            self.track = Track(anchors, controls, widths)
            self.game_sprites.add(self.track)
            return True
        except Exception as e:
            r.game_core.unexpected_error_msg = f"Error occurred when loading {filename}: (File is likely corrupted)"
            r.game_core.set_game_mode(MainMenu)
            print("Track load failed:", e)
            return False


    def initialise_num_of_laps(self):
        self.num_of_laps_setter = UIOptionSelector(pygame.Rect((0, 0), game_core.screen_dimensions),
                         self.gui_manager,
                         title= "Select number of laps",
                         options =["1", "2", "3", "4", "5", "6", "7", "8", "9", "10"],
                         callback=lambda num_of_laps: self.set_num_of_laps(num_of_laps),
                         return_mode=MainMenu)

    def set_num_of_laps(self, num_of_laps):
        self.time_manager.num_of_laps = int(num_of_laps)
        self.time_manager.initialise_manager(self.num_of_cars)
        self.init_cars()

    @abstractmethod
    def init_cars(self):
        pass

    def update(self):
        # Update active cars

        self.gui_manager.update(1 / game_core.frame_rate)

        if game_core.is_paused or self.end_screen is not None:
            return

        if self.track:
            self.track.update()
        for i, car in enumerate(self.cars):


            if self.is_crashed[i] or self.time_manager.is_all_laps_finished[i]:
                continue

            car.collision_detection(self.track)
            if car.is_crashed:
                self.is_crashed[i] = True

            progress, is_lap_completed = car.update_and_get_progress(self.track.track_spine)
            laps_completed = math.floor(progress)
            self.time_manager.update(i, laps_completed, is_lap_completed)

        if all(self.time_manager.is_all_laps_finished):
            results = self.get_results()
            self.display_end_screen(results)
            return



    @abstractmethod
    def display_end_screen(self, results):
        pass

    def get_results(self):
        return self.time_manager.get_stats()

    def end_options(self, option):
        if option == "restart":
            self._reinitialise_simulation()
        else:
            game_core.set_game_mode(MainMenu)

    def event_handle(self):
        """Handle reset button press."""
        super().event_handle()
        if self.reset_button in game_core.pressed_buttons:
            self._reinitialise_simulation()
        if self.menu_button in game_core.pressed_buttons:
            game_core.set_game_mode(MainMenu)



    def reset_environment(self):
        """Reset the simulation environment."""
        self.time_manager.initialise_manager(self.num_of_cars)

    def reset_gui(self):
        """Reset GUI elements."""
        pass

    def reset_cars(self):
        for car in self.cars:
            car.reset()
        self.is_crashed = [False] * self.num_of_cars

    def _reinitialise_simulation(self):
        """Reinitialize the entire simulation."""
        self.reset_gui()
        self.reset_environment()
        self.init_cars()
        self.end_screen = None


class SoloCarSim(CarSimulation):
    """Player-controlled racing simulation."""

    def __init__(self):
        super().__init__(num_of_cars=1)
        self.info_label = pygame_gui.elements.UILabel(
            relative_rect=pygame.Rect((10, 220), (280, 30)),
            text='W: Throttle | S: Brake',
            manager=self.gui_manager,
            container=self.pause_menu,
            object_id = '#info_label'
        )
        self.controls_toggle_button = pygame_gui.elements.UIButton(
            relative_rect=pygame.Rect((5, 80), (285, 40)),
            text="Enable steering slider",
            manager=self.gui_manager,
            container=self.pause_menu,
            object_id='#panel_button'
        )

    def reset_gui(self):
        self.cars[0].speedometer.kill()
        self.cars[0].throttle_and_braking_meter.kill()
        self.cars[0].steering_slider.kill()

    def init_cars(self):
        if self.cars:
            self.game_sprites.remove(self.cars[0])
            self.cars[0].delete_control_panel()
        self.cars = [PlayerCar(self.track.track_spine[0])]
        self.game_sprites.add(self.cars[0])
        self.debug_elements["hitboxes"].append(self.cars[0].hitbox)

    def set_track(self, filename):
        successful = super().set_track(filename)
        if successful:
            self.initialise_num_of_laps()

    def event_handle(self):
        """Handle events and clear button presses."""
        if self.controls_toggle_button in game_core.pressed_buttons:
            self.cars[0].toggle_slider()
            if self.cars[0].is_slider_enabled:
                self.controls_toggle_button.set_text("Disable steer slider")
            else:
                self.controls_toggle_button.set_text("Enable steer slider")

        super().event_handle()

    def display_end_screen(self, results):
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

    def update(self, *args):
        super().update()
        if game_core.is_paused or self.end_screen is not None:
            return
        if self.cars:
            self.cars[0].update()
        if self.is_crashed[0]:
            self.reset_cars()
            self.cars[0].progress = math.floor(self.cars[0].progress)



class RacingSim(CarSimulation):
    """Player-controlled racing simulation."""

    def __init__(self, state_size=15):
        super().__init__(num_of_cars=2)

        self.state_size = state_size

        self.info_label = pygame_gui.elements.UILabel(
            relative_rect=pygame.Rect((10, 100), (280, 30)),
            text='W: Throttle | S: Brake',
            manager=self.gui_manager,
            container=self.pause_menu,
            object_id='#info_label'
        )

        self.rl_model = None
        self.model_loader = None
        game_core.debug_elements["rays"] = [0]

    def set_track(self, filename):

        success = super().set_track(filename)
        if success:
            self.initialise_rl_model()

    def initialise_rl_model(self):
        self.model_loader = UIFileSelector(pygame.Rect((0, 0), game_core.screen_dimensions),
                                           self.gui_manager,
                                           "Models",
                                           "load raceable",
                                           callback=lambda filename: self.set_model(filename),
                                           return_mode=MainMenu)

    def set_model(self, filename=None):
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
        if self.cars:
            self.game_sprites.remove(self.cars[0], self.cars[1])
            self.cars[0].delete_control_panel()
        self.cars = [PlayerCar(self.track.track_spine[0]), AICar(self.track.track_spine[0])]
        self.cars[1].ray_cast(self.track, 0)
        self.game_sprites.add(self.cars[0], self.cars[1])
        self.debug_elements["hitboxes"].extend([self.cars[0].hitbox, self.cars[1].hitbox])


    def update(self, *args):
        super().update()
        if game_core.is_paused or self.end_screen is not None:
            return
        for i, car in enumerate(self.cars):
            if self.time_manager.is_all_laps_finished[i]:
                continue
            if self.is_crashed[i]:
                car.reset()
                self.is_crashed[i] = False
            else:
                if isinstance(car, AICar):
                    sensors = car.ray_cast(self.track, 0)
                    state = sensors + [car.velocity.magnitude() / car.max_speed, car.steer / car.max_steer]
                    actions = self.rl_model.get_stochastic_actions(state)
                    car.update(actions[0])
                else:
                    car.update()



    def event_handle(self):
        super().event_handle()

    def display_end_screen(self, results):
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

        winner = "Player" if player_lap_times[-1] < ai_lap_times[-1] else "AI"

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
    """AI-powered car simulation with reinforcement learning."""

    def __init__(self, simulation_size=12, state_size=15):
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
            text=f'Speed: {game_core.tick_speedup}X',
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

        self.learning_log_box = pygame_gui.elements.UITextBox(
            html_text="",
            relative_rect=pygame.Rect((game_core.screen_dimensions[0] / 3+10, 5), (game_core.screen_dimensions[0] / 3-10, 190)),
            manager=self.gui_manager,
            container=self.stats_panel
        )
        self.model_training_box = pygame_gui.elements.UITextBox(
            html_text="",
            relative_rect=pygame.Rect((game_core.screen_dimensions[0] * 2/ 3 + 10, 5),
                                      (game_core.screen_dimensions[0] / 3 - 25, 190)),
            manager=self.gui_manager,
            container=self.stats_panel
        )

        # AI Configuration
        self.state_size = state_size
        self.reward_size = 9

        # Simulation state
        self.actions = np.zeros([self.num_of_cars, 2], np.float32)

        # Progress tracking
        self.stuck_timer = [0] * self.num_of_cars
        self.prev_progress = [0.0] * self.num_of_cars
        self.max_epoch_progress = [0.0] * self.num_of_cars
        self.stuck_limit = 1000
        self.historic_progress = [0]

        # Reward tracking
        self.time_step_rewards_breakdown = []
        self.epoch_rewards_breakdown = np.zeros([self.num_of_cars, self.reward_size], float)

        # Model management
        self.performance_history = []
        self.history_window = 20
        self.rollback_threshold = 0.7
        self.best_hist_avg = 0
        self.best_model_params = None

        self.rollback_counter = 0
        self.avg_progress = 0
        self.max_progress = 0
        self.avg_lap_time = None
        self.best_lap_time = None


        self.model_type_selector = None
        self.model_loader = None
        self.num_of_laps_setter = None
        self.model_saver = None
        self.hyper_param_setter = None
        self.model_type = None
        self.rl_model = None
        self.is_model_raceable= False

        self.learning_log_data = (f"Average progress: {self.avg_progress:.2f}% \n"
                                  f"Best progress: {self.max_progress:.2f}% \n"
                                  f"Average lap time: None \n"
                                  f"Fastest lap time: None \n"
                                  f"Model ready?: {self.is_model_raceable} \n"
                                  f"Number of rollback: {self.rollback_counter} \n")

        self.learning_log_box.set_text(self.learning_log_data)

        self.episodic_log_data = ["-----------------"]
        self.model_training_log = ("Average return: None \n"
                                   "Average reward: None")
        self.model_training_box.set_text(self.model_training_log)
        # Debug setup
        game_core.debug_elements["rays"] = [0] * simulation_size


    def set_track(self, filename):
        success = super().set_track(filename)
        if success:
            self.initialise_model_type()

    def initialise_model_type(self):
        self.model_type_selector = UIOptionSelector(pygame.Rect((0, 0), game_core.screen_dimensions),
                                                    self.gui_manager,
                                                    ["REINFORCE", "Monte Carlo Actor Critic (MCAC)"],
                                                    callback=lambda model_type: self.set_model_type(model_type),
                                                    return_mode=MainMenu)

    def set_model_type(self, model_type):
        self.model_type = model_type
        self.initialise_rl_model()

    def initialise_rl_model(self):
        self.model_loader = UIFileSelector(pygame.Rect((0, 0), game_core.screen_dimensions),
                                           self.gui_manager,
                                           "Models",
                                           f"load {self.model_type}",
                                           callback=lambda filename: self.set_model(filename))


    def set_model(self, filename):
        if filename is None:
            self.initialise_hyper_params()
        else:
            try:
                save_data = np.load(f"Models/{filename}.npy", allow_pickle=True).item()
                self.load_model(save_data)
                self.initialise_num_of_laps()
            except Exception as e:
                game_core.unexpected_error_msg = f"ERROR LOADING MODEL:{str(e)}"
                game_core.set_game_mode(MainMenu)

    def initialise_hyper_params(self):
        self.hyper_param_setter = UIModelCreator(
            pygame.Rect((0, 0), game_core.screen_dimensions),
            self.gui_manager,
            self.model_type,
            callback=lambda config: self.create_model(config),
            return_mode=MainMenu
        )

    def load_model(self, save_data):
        actor_layer_sizes = []
        weights = save_data["actor_params"]["weights"]

        for w in weights[:-1]:
            actor_layer_sizes.append(w.shape[1])

        # Load basic info
        self.is_model_raceable = save_data['is_model_raceable']
        self.episode_num = save_data['episode_num']
        self.num_of_cars = save_data.get('num_of_cars', 12)

        # Create the appropriate model
        if save_data['model_type'] == 'REINFORCE':
            self.rl_model = REINFORCEModel(
                input_size=self.state_size,
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
            # Extract critic architecture
            critic_layer_sizes = []
            weights = save_data["critic_params"]["weights"]

            for w in weights[:-1]:
                critic_layer_sizes.append(w.shape[1])

            self.rl_model = MCACModel(
                input_size=self.state_size,
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

        self.best_model_params = copy.deepcopy(save_data['best_model_params'])

    def create_model(self, config):
        try:
            self.num_of_cars = config['num_of_cars']
            if self.model_type == "REINFORCE":
                self.rl_model = REINFORCEModel(input_size=self.state_size,
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
                self.rl_model = MCACModel(input_size=self.state_size,
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
        except:
            game_core.unexpected_error_msg  = f"ERROR LOADING MODEL:{str(e)}"
            game_core.set_game_mode(MainMenu)



    def init_cars(self):
        if self.cars:
            for car in self.cars:
                self.game_sprites.remove(car)
                self.debug_elements["hitboxes"].remove(car.hitbox)

        self.cars = []
        for i in range(self.num_of_cars):
            car = AICar(self.track.track_spine[0])
            car.ray_cast(self.track, i)
            self.cars.append(car)
            self.game_sprites.add(car)
            self.debug_elements["hitboxes"].append(car.hitbox)


    def update(self, *args):
        """Update AI simulation."""
        super().update()
        if not game_core.is_paused and self.rl_model is not None:
            self._update_ai_step()

        if all(self.is_crashed):
            super().update()
            self._handle_episode_end()
            return



    def _update_ai_step(self):
        """Perform one AI training step."""
        # Get current states
        states = np.zeros((self.num_of_cars, self.state_size), np.float32)

        for i, car in enumerate(self.cars):
            if not self.is_crashed[i]:
                sensors = car.ray_cast(self.track, i)
                states[i] = sensors + [car.velocity.magnitude() / car.max_speed, car.steer / car.max_steer]

        self.actions, pre_squash = self.rl_model.get_stochastic_actions(states)
        rewards = np.zeros(self.num_of_cars, np.float32)

        for i, car in enumerate(self.cars):
            if not self.is_crashed[i]:
                car.update(self.actions[i])
                # Get sensors again for reward computation (they're computed after the action)
                sensors = car.ray_cast(self.track, i)
                rewards[i] = self._process_car_reward(i, car, sensors)

        self.rl_model.update_trajectory(states, pre_squash, rewards)

    def _process_car_reward(self, i, car, sensors):
        """Process a single car's reward after action is taken."""
        car.collision_detection(self.track)
        progress, is_lap_finished = car.update_and_get_progress(self.track.track_spine)

        if is_lap_finished:
            laps_completed = math.floor(progress)
            # Ensure we don't double count if super().update() also runs
            if self.time_manager.lap_times[i][laps_completed - 1] is None:
                self.time_manager.update(i, laps_completed, is_lap_finished)


        is_stuck = self._check_if_stuck(i, progress)

        # Compute reward
        reward, rewards_array = car.compute_reward(
            is_stuck,
            is_lap_finished,
            self.prev_progress[i],
            np.mean(self.historic_progress),
            sensors
        )

        # Check if completed max laps
        if car.progress >= self.time_manager.num_of_laps:
            self.is_crashed[i] = True  # Mark as finished
            car.is_crashed = True  # Actually stop the car
            reward += 80
            rewards_array[-1] += 80

        clipped_reward = np.clip(reward, -50, 100 * self.time_manager.num_of_laps)

        # Tracks rewards
        reward_breakdown = np.array(rewards_array)
        self.epoch_rewards_breakdown[i] += reward_breakdown
        self.time_step_rewards_breakdown.append(reward_breakdown)

        # Update progress tracking
        self.max_epoch_progress[i] = max(progress, self.max_epoch_progress[i])
        self.prev_progress[i] = progress

        return clipped_reward

    def _check_if_stuck(self, i, current_progress):
        """Check if car is stuck and update stuck timer."""
        if current_progress - self.prev_progress[i] < 0.001:
            self.stuck_timer[i] += 1
        else:
            self.stuck_timer[i] = 0

        if self.stuck_timer[i] > self.stuck_limit:
            self.cars[i].is_crashed = True
            return True
        return False

    def reset_environment(self):
        """Reset the simulation environment."""
        super().reset_environment()
        self.stuck_timer = [0] * self.num_of_cars
        self.prev_progress = [0.0] * self.num_of_cars
        self.is_crashed = [False] * self.num_of_cars

    def _reinitialise_simulation(self):
        """Reinitialize simulation and update model parameters."""
        # Only update params if there's actual trajectory data
        if len(self.rl_model.states) > 0 and len(self.rl_model.rewards) > 0:
            self.model_training_log = self.rl_model.update_params()
            self.model_training_box.set_text(self.model_training_log)

        self.max_epoch_progress = [0.0] * self.num_of_cars
        super()._reinitialise_simulation()

    def _handle_episode_end(self):
        """Handle end of episode - logging, rollback check, reset."""
        # Only process episode if we have trajectory data
        if len(self.rl_model.states) > 0:
            self.episode_num += 1
            self.episodes_completed_label.set_text(f'Episodes completed: {self.episode_num}')
            log_msg = self._log_episode_results()

            avg_progress = np.mean(self.max_epoch_progress)
            self._cache_performance(avg_progress, self.max_epoch_progress)
            log_msg += self._check_rollback()
            self.update_logger(log_msg)
        for car in self.cars:
            car.progress = math.floor(car.progress)
        # Always reset reward tracking
        self.epoch_rewards_breakdown = np.zeros([self.num_of_cars, self.reward_size], float)
        self.time_step_rewards_breakdown = []

        self._reinitialise_simulation()

    def update_logger(self, log_msg):
        self.episodic_log_data.append(log_msg)

        if len(self.episodic_log_data) > 40:
            self.episodic_log_data = self.episodic_log_data[-40:]

        self.episodic_log_box.set_text("<br>".join(self.episodic_log_data))
        if self.episodic_log_box.scroll_bar:
            self.episodic_log_box.scroll_bar.set_scroll_from_start_percentage(1.0)

        self.log_learning_progress()
        self.learning_log_box.set_text(self.learning_log_data)

    def log_learning_progress(self):
        rounded_avg_lap_time = round(self.avg_lap_time, 5) if self.avg_lap_time else None
        rounded_best_lap_time = round(self.best_lap_time, 5) if self.best_lap_time else None
        self.learning_log_data = (f"Average progress: {self.avg_progress:.2f}% \n"
                                  f"Best progress: {self.max_progress:.2f}% \n"
                                  f"Average lap time: {rounded_avg_lap_time} \n"
                                  f"Fastest lap time: {rounded_best_lap_time} \n"
                                  f"Model ready?: {self.is_model_raceable} \n"
                                  f"Number of rollback: {self.rollback_counter} \n")

    def _log_episode_results(self):
        """Log episode statistics."""
        print(f"Episode: {self.episode_num}")

        avg_epoch_rewards = np.round(self.epoch_rewards_breakdown.mean(axis=0), 3)
        print(f"Epoch rewards: {avg_epoch_rewards.tolist()}")

        time_step_rewards = np.array(self.time_step_rewards_breakdown)
        avg_time_step_rewards = np.round(time_step_rewards.mean(axis=0), 3)
        print(f"Time step rewards: {avg_time_step_rewards.tolist()}")
        print()
        avg_progress = sum(car.progress for car in self.cars) / len(self.cars)
        episode_duration = (pygame.time.get_ticks() - self.time_manager.start_time)/1000
        return (f"Episode: {self.episode_num} \n"
                f"Episode duration: {episode_duration:.2f} \n"
                f"Avg progress: {avg_progress:.2f} \n"
                f"Avg reward: {sum(avg_epoch_rewards.tolist()):.2f} \n")

    def _check_rollback(self):
        """Check if model should be rolled back due to poor performance."""
        if len(self.performance_history) < self.history_window:
            return ""

        recent_performance = np.mean(self.performance_history[-self.history_window:])

        if recent_performance < self.rollback_threshold * self.best_hist_avg:
            # Rollback to best model
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

    def _cache_performance(self, avg_total, epoch_progresses):
        """Cache performance metrics and update best model if improved."""
        self.avg_progress = sum(car.progress for car in self.cars) / len(self.cars) * 100
        self.max_progress = max(self.max_progress, self.avg_progress)
        self.avg_lap_time = self.time_manager.get_average_lap_time()
        if self.avg_lap_time is not None:
            if self.best_lap_time is not None:
                self.best_lap_time = min(self.best_lap_time, self.avg_lap_time)
            else:
                self.best_lap_time = self.avg_lap_time


        if self.avg_progress/100 > 0.75 * self.time_manager.num_of_laps:
            self.is_model_raceable = True
        else:
            self.is_model_raceable = False
        self.performance_history.append(avg_total)

        # Maintain performance history window
        if len(self.performance_history) > 2 * self.history_window:
            self.performance_history.pop(0)

        # Calculate relevant historical average
        if len(self.performance_history) >= self.history_window:
            relevant_hist = self.performance_history[-self.history_window:]
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
        if len(self.historic_progress) > self.history_window:
            self.historic_progress.pop(0)



    def event_handle(self):
        """Handle user input events."""
        # Reset simulation
        super().event_handle()
        if self.reset_button in game_core.pressed_buttons:
            self._reinitialise_simulation()

        # Speed control
        if self.tick_speedup_button in game_core.pressed_buttons:
            self._toggle_speed()

        # Save model
        if self.save_ai_button in game_core.pressed_buttons:
            self.initialise_model_saver()



    def initialise_model_saver(self):
        self.model_saver = UIFileSelector(pygame.Rect((0, 0), game_core.screen_dimensions),
                                          self.gui_manager,
                                          "Models",
                                          "save",
                                          callback= lambda filename: self.save_model(filename))

    def reset_cars(self):
        """Reset all AI cars."""
        self.stuck_timer = [0] * self.num_of_cars
        self.prev_progress = [0.0] * self.num_of_cars

        for car in self.cars:
            car.reset()
            car.progress = 0

    def save_model(self, filename=None):
        """Save the current model to disk."""
        if filename is None:
            return

        # Extract activation names from layers
        actor_activations = [layer.activation_name for layer in self.rl_model.actor.layers]

        save_data = {
            'is_model_raceable': self.is_model_raceable,
            'model_type': 'MCAC' if hasattr(self.rl_model, 'critic') else 'REINFORCE',
            'episode_num': self.episode_num,
            'best_model_params': self.best_model_params,

            # Actor configuration
            'actor_params': self.rl_model.actor.get_params(),
            'actor_activations': actor_activations,  # SAVE THIS
            'gamma': self.rl_model.gamma,
            'entropy': self.rl_model.entropy,
            'l2_lambda': self.rl_model.l2_lambda,
            'actor_learning_rate': self.rl_model.actor.alpha,
            'actor_beta1': self.rl_model.actor.beta1,
            'actor_beta2': self.rl_model.actor.beta2,

            # Number of cars
            'num_of_cars': self.num_of_cars
        }

        if save_data['model_type'] == 'MCAC':
            critic_activations = [layer.activation_name for layer in self.rl_model.critic.layers]
            save_data['critic_params'] = self.rl_model.critic.get_params()
            save_data['critic_activations'] = critic_activations  # SAVE THIS
            save_data['critic_learning_rate'] = self.rl_model.critic.alpha
            save_data['critic_beta1'] = self.rl_model.critic.beta1
            save_data['critic_beta2'] = self.rl_model.critic.beta2

        np.save(f"Models/{filename}.npy", save_data, allow_pickle=True)
        self.model_saver = None

    def _toggle_speed(self):
        if game_core.tick_speedup == 1:
            game_core.tick_speedup = 2
        elif game_core.tick_speedup == 2:
            game_core.tick_speedup = 5
        elif game_core.tick_speedup == 5:
            game_core.tick_speedup = 10
        else:
            game_core.tick_speedup = 1

        self.speed_label.set_text(f"Speed: {game_core.tick_speedup}X")

    def display_end_screen(self, results):
        pass

