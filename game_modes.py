import math
import pygame
import resources as r
from pygame_gui import elements
from gui_custom_elements import UIGaugeMeter, UITrackCanvas
from track import Track
from cars import Car
from rl_model import VPGModel

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
        self.throttle_and_braking = 0
        self.steer = 0
        self.car = Car(r.starting_position)
        self.track = Track(r.CURRENT_TRACK)
        self.vpg_model = VPGModel()
        r.GAME_SPRITES.add(self.car)
        r.GAME_SPRITES.add(self.track)
        r.DEBUG_ELEMENTS["hitboxes"].append(self.car.hitbox)
        self.steering_slider = elements.UIHorizontalSlider(
            relative_rect=pygame.Rect((r.SCREEN_DIMENSIONS[0] / 2, 700), (600, 30)),
            start_value=0,
            value_range=(-100, 100),
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

        self.reset_button = elements.UIButton(
            relative_rect=pygame.Rect((300, 600), (100, 30)),
            text="Click to reset",
            manager=r.GUI_MANAGER
        )

    def update_car_simulation(self):
        self._car_movement()
        self._handle_buttons()

    # Calls procedures to update the throttle and steering values.
    # Updates the car's properties with respect to these values
    def _car_movement(self):
        self._handle_steering(self.steering_slider.get_current_value()/100)
        self._handle_throttle()
        self.car.car_movement(self.steer, self.throttle_and_braking)
        self.car.ray_cast(self.track)
        self.car.collision_detection(self.track)

    def _handle_buttons(self):
        if self.reset_button in r.PRESSED_BUTTONS:
            self._reinitialise_car_simulation()

        r.PRESSED_BUTTONS.clear()

    # Updates the steer value when the steering slider is moved.
    def _handle_steering(self, steer_ratio):
        self.steer = r.max_steer * steer_ratio


    # Increments the throttle or braking based on user input.
    # Value decays when there is no input to emulate release of throttle/brake.
    """
    For accuracy, the throttle would have to be handled in a similar way as the steering using a slider of some sort
    to allow more control over the magnitude, since by using keys to control throttle, there is no way to steadily hold 
    the throttle partially pressed down. However for testing purposes, this is ideal since its easier to control than
    having 2 separate sliders
    """
    def _handle_throttle(self):
        if pygame.K_w in r.PRESSED_KEYS:
            self.throttle_and_braking = r.clamp_value(self.throttle_and_braking + r.throttle_factor, 0, 1)
        elif pygame.K_s in r.PRESSED_KEYS:
            self.throttle_and_braking = r.clamp_value(self.throttle_and_braking - r.brake_factor, -1, 0)
        else:
            self.throttle_and_braking *= 0.9
            if -0.01 < self.throttle_and_braking < 0.01:
                self.throttle_and_braking = 0
        throttle_normalised_value = (self.throttle_and_braking + 1) * 50
        self.throttle_and_braking_meter.set_current_progress(throttle_normalised_value)
        self.speedometer.update_value(self.car.velocity.magnitude())

    @staticmethod
    def _reinitialise_car_simulation():
        r.IS_INITIALIZED = False
        r.GAME_SPRITES.empty()
        r.GUI_MANAGER.clear_and_reset()
        r.DEBUG_ELEMENTS.clear()


def run_car_simulation():
    global car_simulation
    if not r.IS_INITIALIZED:
        car_simulation = CarSimulation()
        r.IS_INITIALIZED = True
    car_simulation.update_car_simulation()



    # def _get_actions_from_model(self):
    #     state_vector = [self.car.throttle, self.car.steer]
    #     state_vector.extend(self.car.sensors)
    #     actions = self.vpg_model.get_actions(state_vector)[0]
    #     self.steering_slider.set_current_value(actions[0])
    #     self.throttle_and_braking = actions[1]




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