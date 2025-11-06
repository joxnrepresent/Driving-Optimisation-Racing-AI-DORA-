import math
from logging import setLogRecordFactory
import pygame
import resources as r
from pygame_gui import elements
from gui_custom_elements import UIGaugeMeter, UITrackCanvas
from track import Track
from cars import PlayerCar, AICar
from rl_model import VPGModel
from numpy import clip, exp
from abc import ABC, abstractmethod
from resources import game_core

"""This module contains classes """
#---------------------------------------------------------------------------------------------------------------------#

car_simulation = None
def run_car_simulation():
    global car_simulation
    if not game_core.is_initialized:
        # car_simulation = RacingSim()
        car_simulation = AICarSim()
        game_core.is_initialized= True
    car_simulation.update_car_simulation()

class CarSimulation(ABC):
    """
    This module is for running a program that creates an instance of the Car class that can be controlled by the user.
    A slider controls the  steering, 'W' for throttle and 'S' for braking. The meter in the bottom left corner represents
    the magnitude of throttle/braking (max throttle = 100, max braking = 0, neutral - 0). The speedometer represents the
    magnitude of the velocity
    """
    """Class stores the variables, objects and methods needed to run the car simulation."""
    def __init__(self):
        self.track = Track(game_core.current_track)
        game_core.game_sprites.add(self.track)
        game_core.starting_position = self.track.track_spine[0]
        self.reset_button = elements.UIButton(
            relative_rect=pygame.Rect((300, 600), (100, 30)),
            text="Click to reset",
            manager=game_core.gui_manager
        )
        self.car = []
        self._reset_cars()

    def update_car_simulation(self):
        if self.car.is_crashed:
            self._reinitialise_car_simulation()
        self._drive_car()
        self._handle_buttons()

    @abstractmethod
    def _reset_cars(self):
        pass

    # Calls procedures to update the throttle and steering values.
    # Updates the car's properties with respect to these values
    @abstractmethod
    def _drive_car(self):
        pass

    def _reset_gui(self):
        game_core.gui_manager.clear_and_reset()
        self.reset_button = elements.UIButton(
            relative_rect=pygame.Rect((300, 600), (100, 30)),
            text="Click to reset",
            manager=game_core.gui_manager
        )

    def _handle_buttons(self):
        if self.reset_button in game_core.pressed_buttons:
            self._reinitialise_car_simulation()
        game_core.pressed_buttons.clear()

    def _reinitialise_car_simulation(self):
        game_core.game_sprites.remove(self.car)
        game_core.debug_elements["hitboxes"].clear()
        game_core.debug_elements["rays"].clear()
        game_core.debug_elements["AABB"].clear()
        self._reset_gui()
        self._reset_cars()


class RacingSim(CarSimulation):
    def __init__(self):
        super().__init__()

    def _reset_cars(self):
        self.car = PlayerCar(game_core.starting_position)
        game_core.GAME_SPRITES.add(self.car)
        game_core.debug_elements["hitboxes"].append(self.car.hitbox)

    def _drive_car(self):
        if not self.car.is_crashed:
            progress = self.car.get_progress(self.track.track_spine)
            print(progress)
            self.car.car_movement()
            self.car.collision_detection(self.track)


class AICarSim(CarSimulation):
    def __init__(self):
        super().__init__()
        """
        VPG model testing variables
        """
        self.vpg_model = VPGModel(16)
        self.actions = (0.0,0.0)
        self.sensors = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
        self.total_reward = 0
        self.tick_speedup = elements.UIButton(
            relative_rect=pygame.Rect((400, 600), (100, 30)),
            text="Click to slow",
            manager=game_core.gui_manager
        )
        self.save_ai = elements.UIButton(
            relative_rect=pygame.Rect((400, 650), (100, 30)),
            text="Click to save",
            manager=game_core.gui_manager
        )

    def _reset_cars(self):
        self.car = AICar(game_core.starting_position)
        game_core.game_sprites.add(self.car)
        game_core.debug_elements["hitboxes"].append(self.car.hitbox)
        self.throttle_and_braking_meter = elements.UIProgressBar(
            relative_rect=pygame.Rect((game_core.screen_dimensions[0] / 4 - 300, 700), (500, 30)),
            manager=game_core.gui_manager
        )
        self.speedometer = UIGaugeMeter(
            relative_rect=pygame.Rect((game_core.screen_dimensions[0] / 4 - 300, 600), (200, 100)),
            manager=game_core.gui_manager
        )

    def _drive_car(self):
        if not self.car.is_crashed:
            self._model_drive_car()
            self.car.car_movement(self.actions)
            self.car.collision_detection(self.track)
            reward = self.car.compute_reward()
            self.total_reward += reward
            self.vpg_model.append_return(reward)
            """
            update testing meters
            """
            self.throttle_and_braking_meter.set_current_progress(self.actions[1])
            self.speedometer.update_value(self.car.velocity.magnitude())

    def _model_drive_car(self):
        state_vector = [*self.actions]
        progress = self.car.get_progress(self.track.track_spine)
        if progress >= 0.9:
            game_core.screen_fill = "green"
        state_vector.append(progress)
        self.sensors = self.car.ray_cast(self.track)
        state_vector.extend(self.sensors)
        self.actions = clip(self.vpg_model.get_actions(state_vector)[0], -1, 1)

    def _reinitialise_car_simulation(self):
        expected_return = self.vpg_model.update_params()
        print(self.total_reward, expected_return, exp(self.vpg_model.neural_net.log_std))
        self.total_reward = 0
        super()._reinitialise_car_simulation()

    def _handle_buttons(self):
        if self.reset_button in game_core.pressed_buttons:
            self._reinitialise_car_simulation()
        if self.tick_speedup in game_core.pressed_buttons:
            if game_core.tick_speedup == 1:
                game_core.tick_speedup = 250
            elif game_core.tick_speedup == 250:
                game_core.tick_speedup = 750
            else:
                game_core.tick_speedup = 1
        game_core.pressed_buttons.clear()
        if self.save_ai in game_core.pressed_buttons:
            with open("Model_weights/" + "testing_weights" + "1" + ".txt", "w") as file:
                file.write(self.vpg_model.neural_net.get_params())
            print("saved")

    def _reset_gui(self):
        super()._reset_gui()
        self.tick_speedup = elements.UIButton(
            relative_rect=pygame.Rect((400, 600), (100, 30)),
            text="Click to slow",
            manager=game_core.gui_manager
        )
        self.save_ai = elements.UIButton(
            relative_rect=pygame.Rect((400, 650), (100, 30)),
            text="Click to save",
            manager=game_core.gui_manager
        )

"""
Creates an interface that allows the user to create, save and draw custom tracks.
"""
#---------------------------------------------------------------------------------------------------------------------#
track_maker = None
# global method to initialise and run the track maker
def run_track_maker():
    global track_maker
    if not game_core.is_initialized:
        track_maker = TrackMakerUI()
        game_core.is_initialized = True
    track_maker.update_drawing_interface()


class TrackMakerUI:
    """
    Class stores the variables, objects and methods needed to run the track maker.
    """
    def __init__(self):
        self.instructions_label = elements.UILabel(
            relative_rect= pygame.Rect((game_core.screen_dimensions[0] // 2 - 250, 100), (500, 30)),
            text="Click to add control points. Hold SPACE to draw a straight",
            manager=game_core.gui_manager,
        )
        self.canvas = UITrackCanvas(
            relative_rect= pygame.Rect((0, 0), (game_core.screen_dimensions[0], game_core.screen_dimensions[1])),
            manager=game_core.gui_manager
        )

        self.start_drawing_button = elements.UIButton(
            relative_rect= pygame.Rect((0, 100), (100, 30)),
            text="Click to draw",
            manager=game_core.gui_manager
        )

        self.save_button = elements.UIButton(
            relative_rect= pygame.Rect((100, 100), (100, 30)),
            text="Click to save",
            manager=game_core.gui_manager
        )

        self.load_button = elements.UIButton(
            relative_rect= pygame.Rect((200, 100), (100, 30)),
            text="Click to load",
            manager=game_core.gui_manager
        )

        self.clear_button = elements.UIButton(
            relative_rect= pygame.Rect((300, 100), (100, 30)),
            text="Click to clear",
            manager=game_core.gui_manager
        )

        self.start_car_sim_button = elements.UIButton(
            relative_rect= pygame.Rect((300, 200), (100, 30)),
            text="Click to start sim",
            manager=game_core.gui_manager
        )

    # Handles button presses:
    def _handle_buttons(self):
        if self.start_drawing_button in game_core.pressed_buttons:
            self.canvas.is_drawing = not self.canvas.is_drawing
            if self.canvas.is_drawing:
                self.start_drawing_button.set_text("Click to stop")
            else:
                self.start_drawing_button.set_text("Click to draw")

        if self.save_button in game_core.pressed_buttons:
            self.canvas.save_drawing("testing_track")

        if self.load_button in game_core.pressed_buttons:
            self.canvas.load_drawing("testing_track")

        if self.clear_button in game_core.pressed_buttons:
            self.canvas.clear()

        if self.start_car_sim_button in game_core.pressed_buttons:
            game_core.game_mode = "car simulation"
            game_core.is_initialized = False
            game_core.game_sprites.empty()
            game_core.debug_elements.clear()
            game_core.gui_manager.clear_and_reset()
            global track_maker
            track_maker = None

        game_core.pressed_buttons.clear()

    def update_drawing_interface(self):
        self._handle_buttons()