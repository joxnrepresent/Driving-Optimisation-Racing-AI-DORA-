import math
import pickle

import numpy
import numpy as np
import pygame
import pygame_gui
import resources as r
from resources import game_core
from gui_custom_elements import UIGaugeMeter, UITrackCanvas
from track import Track
from cars import PlayerCar, AICar
from rl_model import A2CModel, REINFORCEModel
from numpy import clip, exp
from abc import ABC, abstractmethod
from collections import defaultdict

episode_num = 0
"""This module contains classes """
#---------------------------------------------------------------------------------------------------------------------#

class GameMode(ABC):
    def __init__(self):
        self.game_sprites = game_core.game_sprites
        self.debug_elements = game_core.debug_elements
        self.gui_manager = game_core.gui_manager

    @abstractmethod
    def update(self, *args):
        self.game_sprites.update(*args)
        self.gui_manager.update(1/game_core.frame_rate)
        pass

    @abstractmethod
    def event_handle(self):
        pass

class CarSimulation(GameMode):
    """Class stores the variables, objects and methods needed to run the car simulation."""
    def __init__(self):
        super().__init__()
        self.track = Track(game_core.current_track)
        self.game_sprites.add(self.track)
        game_core.starting_position = self.track.track_spine[0]
        self.reset_button = pygame_gui.elements.UIButton(
            relative_rect=pygame.Rect((300, 600), (100, 30)),
            text="Click to reset",
            manager=self.gui_manager
        )
        self.car = None


    def update(self, *args):
        if self.car.is_crashed:
            self._reinitialise_car_simulation()
        super().update(*args)

    @abstractmethod
    def _reset_cars(self):
        pass

    def _reset_gui(self):
        self.gui_manager.clear_and_reset()
        self.reset_button = pygame_gui.elements.UIButton(
            relative_rect=pygame.Rect((300, 600), (100, 30)),
            text="Click to reset",
            manager=self.gui_manager
        )

    def event_handle(self):
        if self.reset_button in game_core.pressed_buttons:
            self.car.is_crashed = True

    def _reinitialise_car_simulation(self):
        self._reset_gui()
        self._reset_cars()


class RacingSim(CarSimulation):
    def __init__(self):
        super().__init__()
        self._reset_cars()
        self.debug_elements["AABB"].append(self.car)

    def _reset_cars(self):
        self.game_sprites.remove(self.car)
        self.car = PlayerCar(game_core.starting_position)
        self.game_sprites.add(self.car)
        self.debug_elements["hitboxes"].append(self.car.hitbox)

    def _drive_car(self):
        if not self.car.is_crashed:
            self.car.update()
            self.car.collision_detection(self.track)

    def event_handle(self):
        super().event_handle()
        game_core.pressed_buttons.clear()


class AICarSim(CarSimulation):
    def __init__(self, simulation_size = 5, state_size = 11):
        super().__init__()
        self.reward_size = 9

        self.sim_size = simulation_size
        self.state_size = state_size
        self.cars = []
        self.rl_model = A2CModel(self.state_size)
        # self.rl_model = REINFORCEModel(self.state_size)
        if game_core.load_model:
            params = np.load("Model_weights/testing_weights1.npy", allow_pickle=True)
            self.rl_model.actor.set_network_params(params)
        self.actions = np.zeros([self.sim_size, 2], np.float32)
        self.is_crashed = [False] * self.sim_size
        self.stuck_timer = [0] * self.sim_size
        self.prev_progress = [0.0] * self.sim_size
        self.max_progress = [0.0] * self.sim_size
        self.stuck_limit = 500
        self.tick_speedup = pygame_gui.elements.UIButton(
            relative_rect=pygame.Rect((400, 600), (100, 30)),
            text="Click to slow",
            manager=self.gui_manager
        )
        self.save_ai = pygame_gui.elements.UIButton(
            relative_rect=pygame.Rect((400, 650), (100, 30)),
            text="Click to save",
            manager=self.gui_manager
        )
        game_core.debug_elements["rays"] = [0] * simulation_size
        self._reset_cars()
        self.time_step_rewards_breakdown = []
        self.epoch_rewards_breakdown = np.zeros([self.sim_size, self.reward_size], float)

    def update(self):
        if all(self.is_crashed):
            if len(self.rl_model.rewards) > 0:
                global episode_num
                episode_num += 1
                print(f"Episode:{episode_num}")
                avg_epoch_rewards = np.round(self.epoch_rewards_breakdown.mean(axis=0), 3)
                self.time_step_rewards_breakdown = np.array(self.time_step_rewards_breakdown)
                avg_time_step_rewards = np.round(self.time_step_rewards_breakdown.mean(axis=0), 3)
                print(f"Epoch rewards: {avg_epoch_rewards.tolist()}")
                print(f"Time step rewards: {avg_time_step_rewards.tolist()}")
                print()
                self.epoch_rewards_breakdown = np.zeros([self.sim_size, self.reward_size], float)
                self.time_step_rewards_breakdown = []
                self._reinitialise_car_simulation()
                return

        states = np.zeros((self.sim_size, self.state_size), np.float32)
        rewards = np.zeros(self.sim_size, np.float32)
        for i, car in enumerate(self.cars):
            if not self.is_crashed[i]:
                car.collision_detection(self.track)
                sensors = car.ray_cast(self.track, i)

                progress = car.update_and_get_progress(self.track.track_spine)
                rewards[i], rewards_array = car.compute_reward(self.prev_progress[i], self.max_progress[i], sensors, self.actions[i])

                reward_breakdown = np.array(rewards_array)
                self.epoch_rewards_breakdown[i] += reward_breakdown
                self.time_step_rewards_breakdown.append(reward_breakdown)
                self.max_progress[i] = max(progress, self.max_progress[i])


                if progress - self.prev_progress[i] < 0.001:
                    self.stuck_timer[i] += 1
                else:
                    self.stuck_timer[i] = 0

                self.prev_progress[i] = progress


                if self.stuck_timer[i] > self.stuck_limit:
                    car.is_crashed = True

                if car.is_crashed:
                    self.is_crashed[i] = True

                states[i] = sensors

        self.actions, pre_squash = self.rl_model.get_stochastic_actions(states)

        self.rl_model.update_trajectory(states, pre_squash, rewards)
        for i, car in enumerate(self.cars):
            if not self.is_crashed[i]:
                car.update(self.actions[i])
        self.gui_manager.update(1/game_core.frame_rate)


    def event_handle(self):
        if self.reset_button in game_core.pressed_buttons:
            self._reinitialise_car_simulation()
        if self.tick_speedup in game_core.pressed_buttons:
            if game_core.tick_speedup == 1:
                game_core.tick_speedup = 50
            elif game_core.tick_speedup == 50:
                game_core.tick_speedup = 100
            else:
                game_core.tick_speedup = 1
            self.tick_speedup.set_text(f"speed: {game_core.tick_speedup}")


        if self.save_ai in game_core.pressed_buttons:
            params = self.rl_model.actor.get_network_params()
            np.save("Model_weights/testing_weights1.npy", np.array(params, dtype=object), allow_pickle=True)
            print(params)
        game_core.pressed_buttons.clear()



    def _reset_cars(self):
        self.cars.clear()
        self.game_sprites.empty()
        self.game_sprites.add(self.track)

        self.is_crashed = [False] * self.sim_size
        self.stuck_timer = [0] * self.sim_size
        self.prev_progress = [0.0] * self.sim_size

        for i in range(self.sim_size):
            car = AICar(game_core.starting_position)
            self.cars.append(car)
            self.debug_elements["hitboxes"].append(car.hitbox)
            self.debug_elements["AABB"].append(car)
            self.game_sprites.add(car)

    def _reset_gui(self):
        super()._reset_gui()
        self.tick_speedup = pygame_gui.elements.UIButton(
            relative_rect=pygame.Rect((400, 600), (100, 30)),
            text=f"Speed:{game_core.tick_speedup}",
            manager=self.gui_manager
        )
        self.save_ai = pygame_gui.elements.UIButton(
            relative_rect=pygame.Rect((400, 650), (100, 30)),
            text="Click to save",
            manager=self.gui_manager
        )

    def _reinitialise_car_simulation(self):
        self.rl_model.update_params()
        super()._reinitialise_car_simulation()


"""
Creates an interface that allows the user to create, save and draw custom tracks.
"""
#---------------------------------------------------------------------------------------------------------------------#
# track_maker = None
# # global method to initialise and run the track maker
# def run_track_maker():
#     global track_maker
#     if not game_core.is_initialized:
#         track_maker = TrackMakerUI()
#         game_core.is_initialized = True
#     track_maker.update_drawing_interface()


class TrackMakerUI(GameMode):
    """
    Class stores the variables, objects and methods needed to run the track maker.
    """
    def __init__(self):
        super().__init__()

        self.canvas = UITrackCanvas(
            relative_rect= pygame.Rect((0, 0), (game_core.screen_dimensions[0], game_core.screen_dimensions[1])),
            manager=self.gui_manager
        )

        self.start_drawing_button = pygame_gui.elements.UIButton(
            relative_rect= pygame.Rect((0, 100), (100, 30)),
            text="Click to draw",
            manager=self.gui_manager
        )

        self.save_button = pygame_gui.elements.UIButton(
            relative_rect= pygame.Rect((100, 100), (100, 30)),
            text="Click to save",
            manager=self.gui_manager
        )

        self.load_button = pygame_gui.elements.UIButton(
            relative_rect= pygame.Rect((200, 100), (100, 30)),
            text="Click to load",
            manager=self.gui_manager
        )

        self.clear_button = pygame_gui.elements.UIButton(
            relative_rect= pygame.Rect((300, 100), (100, 30)),
            text="Click to clear",
            manager=self.gui_manager
        )

        self.start_car_sim_button = pygame_gui.elements.UIButton(
            relative_rect= pygame.Rect((300, 200), (100, 30)),
            text="Click to start sim",
            manager=self.gui_manager
        )


    # Handles button presses:
    def event_handle(self):
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
            game_core.set_game_mode(AICarSim)

        game_core.pressed_buttons.clear()

    def update(self):
        self.gui_manager.update(1 / game_core.frame_rate / game_core.tick_speedup)