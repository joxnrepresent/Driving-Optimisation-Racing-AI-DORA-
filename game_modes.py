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

"""This module contains classes """
#---------------------------------------------------------------------------------------------------------------------#

car_simulation = None
def run_car_simulation():
    global car_simulation
    if not r.IS_INITIALIZED:
        # car_simulation = RacingSim()
        car_simulation = AICarSim()
        r.IS_INITIALIZED = True
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
        self.track = Track(r.CURRENT_TRACK)
        r.GAME_SPRITES.add(self.track)
        r.starting_position = self.track.track_spine[0]
        self.reset_button = elements.UIButton(
            relative_rect=pygame.Rect((300, 600), (100, 30)),
            text="Click to reset",
            manager=r.GUI_MANAGER
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
        r.GUI_MANAGER.clear_and_reset()
        self.reset_button = elements.UIButton(
            relative_rect=pygame.Rect((300, 600), (100, 30)),
            text="Click to reset",
            manager=r.GUI_MANAGER
        )

    def _handle_buttons(self):
        if self.reset_button in r.PRESSED_BUTTONS:
            self._reinitialise_car_simulation()
        r.PRESSED_BUTTONS.clear()

    def _reinitialise_car_simulation(self):
        r.GAME_SPRITES.remove(self.car)
        r.DEBUG_ELEMENTS["hitboxes"].clear()
        r.DEBUG_ELEMENTS["rays"].clear()
        r.DEBUG_ELEMENTS["AABB"].clear()
        self._reset_gui()
        self._reset_cars()


class RacingSim(CarSimulation):
    def __init__(self):
        super().__init__()

    def _reset_cars(self):
        self.car = PlayerCar(r.starting_position)
        r.GAME_SPRITES.add(self.car)
        r.DEBUG_ELEMENTS["hitboxes"].append(self.car.hitbox)

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
        self.tick_speed_up = elements.UIButton(
            relative_rect=pygame.Rect((400, 600), (100, 30)),
            text="Click to slow",
            manager=r.GUI_MANAGER
        )
        self.save_ai = elements.UIButton(
            relative_rect=pygame.Rect((400, 650), (100, 30)),
            text="Click to save",
            manager=r.GUI_MANAGER
        )

    def _reset_cars(self):
        self.car = AICar(r.starting_position)
        r.GAME_SPRITES.add(self.car)
        r.DEBUG_ELEMENTS["hitboxes"].append(self.car.hitbox)
        self.throttle_and_braking_meter = elements.UIProgressBar(
            relative_rect=pygame.Rect((r.SCREEN_DIMENSIONS[0] / 4 - 300, 700), (500, 30)),
            manager=r.GUI_MANAGER
        )
        self.speedometer = UIGaugeMeter(
            relative_rect=pygame.Rect((r.SCREEN_DIMENSIONS[0] / 4 - 300, 600), (200, 100)),
            manager=r.GUI_MANAGER
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
            r.SCREEN_FILL = "green"
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
        if self.reset_button in r.PRESSED_BUTTONS:
            self._reinitialise_car_simulation()
        if self.tick_speed_up in r.PRESSED_BUTTONS:
            if r.TICK_SPEEDUP == 1:
                r.TICK_SPEEDUP = 250
            elif r.TICK_SPEEDUP == 250:
                r.TICK_SPEEDUP = 750
            else:
                r.TICK_SPEEDUP = 1
        r.PRESSED_BUTTONS.clear()
        if self.save_ai in r.PRESSED_BUTTONS:
            with open("Model_weights/" + "testing_weights" + "1" + ".txt", "w") as file:
                file.write(self.vpg_model.neural_net.get_params())
            print("saved")

    def _reset_gui(self):
        super()._reset_gui()
        self.tick_speed_up = elements.UIButton(
            relative_rect=pygame.Rect((400, 600), (100, 30)),
            text="Click to slow",
            manager=r.GUI_MANAGER
        )
        self.save_ai = elements.UIButton(
            relative_rect=pygame.Rect((400, 650), (100, 30)),
            text="Click to save",
            manager=r.GUI_MANAGER
        )

"""
Creates an interface that allows the user to create, save and draw custom tracks.
"""
#---------------------------------------------------------------------------------------------------------------------#
track_maker = None
# global method to initialise and run the track maker
def run_track_maker():
    global track_maker
    if not r.IS_INITIALIZED:
        track_maker = TrackMakerUI()
        r.IS_INITIALIZED = True
    track_maker.update_drawing_interface()


class TrackMakerUI:
    """
    Class stores the variables, objects and methods needed to run the track maker.
    """
    def __init__(self):
        self.instructions_label = elements.UILabel(
            relative_rect= pygame.Rect((r.SCREEN_DIMENSIONS[0] // 2 - 250, 100), (500, 30)),
            text="Click to add control points. Hold SPACE to draw a straight",
            manager=r.GUI_MANAGER,
        )
        self.canvas = UITrackCanvas(
            relative_rect= pygame.Rect((0, 0), (r.SCREEN_DIMENSIONS[0], r.SCREEN_DIMENSIONS[1])),
            manager=r.GUI_MANAGER
        )

        self.start_drawing_button = elements.UIButton(
            relative_rect= pygame.Rect((0, 100), (100, 30)),
            text="Click to draw",
            manager=r.GUI_MANAGER
        )

        self.save_button = elements.UIButton(
            relative_rect= pygame.Rect((100, 100), (100, 30)),
            text="Click to save",
            manager=r.GUI_MANAGER
        )

        self.load_button = elements.UIButton(
            relative_rect= pygame.Rect((200, 100), (100, 30)),
            text="Click to load",
            manager=r.GUI_MANAGER
        )

        self.clear_button = elements.UIButton(
            relative_rect= pygame.Rect((300, 100), (100, 30)),
            text="Click to clear",
            manager=r.GUI_MANAGER
        )

        self.start_car_sim_button = elements.UIButton(
            relative_rect= pygame.Rect((300, 200), (100, 30)),
            text="Click to start sim",
            manager=r.GUI_MANAGER
        )

    # Handles button presses:
    def _handle_buttons(self):
        if self.start_drawing_button in r.PRESSED_BUTTONS:
            self.canvas.is_drawing = not self.canvas.is_drawing
            if self.canvas.is_drawing:
                self.start_drawing_button.set_text("Click to stop")
            else:
                self.start_drawing_button.set_text("Click to draw")

        if self.save_button in r.PRESSED_BUTTONS:
            self.canvas.save_drawing("testing_track")

        if self.load_button in r.PRESSED_BUTTONS:
            self.canvas.load_drawing("testing_track")

        if self.clear_button in r.PRESSED_BUTTONS:
            self.canvas.clear()

        if self.start_car_sim_button in r.PRESSED_BUTTONS:
            r.GAME_MODE = "car simulation"
            r.IS_INITIALIZED = False
            r.GAME_SPRITES.empty()
            r.DEBUG_ELEMENTS.clear()
            r.GUI_MANAGER.clear_and_reset()
            global track_maker
            track_maker = None

        r.PRESSED_BUTTONS.clear()

    def update_drawing_interface(self):
        self._handle_buttons()