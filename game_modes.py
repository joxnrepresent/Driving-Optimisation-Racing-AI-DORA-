import math
import pygame
import resources as r
from pygame_gui import elements
from gui_custom_elements import UIGaugeMeter, UITrackCanvas
from track import Track
from cars import PlayerCar, AICar
from rl_model import VPGModel
from numpy import clip

"""This module contains classes """
#---------------------------------------------------------------------------------------------------------------------#
car_simulation = None
class CarSimulation:
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
        self.reset_cars()
        """
        VPG model testing variables
        """
        self.vpg_model = VPGModel()
        self.actions = (0.0,0.0)
        self.sensors = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
        self.total_reward = 0

    def reset_cars(self):
        self.car = AICar(r.starting_position)
        r.GAME_SPRITES.add(self.car)

        # r.DEBUG_ELEMENTS["hitboxes"] = (self.car.hitbox)
        self.reset_button = elements.UIButton(
            relative_rect=pygame.Rect((300, 600), (100, 30)),
            text="Click to reset",
            manager=r.GUI_MANAGER
        )
        self.throttle_and_braking_meter = elements.UIProgressBar(
            relative_rect=pygame.Rect((r.SCREEN_DIMENSIONS[0] / 4 - 300, 700), (500, 30)),
            manager=r.GUI_MANAGER
        )
        self.speedometer = UIGaugeMeter(
            relative_rect=pygame.Rect((r.SCREEN_DIMENSIONS[0] / 4 - 300, 600), (200, 100)),
            manager=r.GUI_MANAGER
        )

    def _model_drive_car(self):
        self.sensors = self.car._ray_cast(self.track)
        state_vector = [*self.actions]
        state_vector.extend(self.sensors)
        self.actions = clip(self.vpg_model.get_actions(state_vector)[0], -1, 1)

    def update_car_simulation(self):
        self._drive_car()
        self._handle_buttons()

    # Calls procedures to update the throttle and steering values.
    # Updates the car's properties with respect to these values
    def _drive_car(self):
        if not self.car.is_crashed:
            # self.car.car_movement()

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

    def _handle_buttons(self):
        if self.reset_button in r.PRESSED_BUTTONS:
            expected_return = self.vpg_model.update_params()
            print(self.total_reward, expected_return)
            self.total_reward = 0
            self._reinitialise_car_simulation()
        r.PRESSED_BUTTONS.clear()

    def _reinitialise_car_simulation(self):
        r.GAME_SPRITES.remove(self.car)
        r.GUI_MANAGER.clear_and_reset()
        r.DEBUG_ELEMENTS.clear()
        self.reset_cars()
        r.DEBUG_ELEMENTS["grid lines"] = [True]



def run_car_simulation():
    global car_simulation
    if not r.IS_INITIALIZED:
        car_simulation = CarSimulation()
        r.IS_INITIALIZED = True
    car_simulation.update_car_simulation()




class PlayerCarSim(CarSimulation):
    def __init__(self):
        super().__init__()


class AICarSim(CarSimulation):
    def __init__(self):
        super().__init__()


"""
This module creates an interface that allows the user to create, save and draw custom tracks.
"""
#---------------------------------------------------------------------------------------------------------------------#
track_maker = None
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

# global method to initialise and run the track maker
def run_track_maker():
    global track_maker
    if not r.IS_INITIALIZED:
        track_maker = TrackMakerUI()
        r.IS_INITIALIZED = True
    track_maker.update_drawing_interface()
