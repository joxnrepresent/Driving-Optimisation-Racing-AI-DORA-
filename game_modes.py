import math
import pickle
import json

import numpy
import numpy as np
import pygame
import pygame_gui
from pygame_gui.core.colour_parser import is_float_str

import resources as r
from resources import game_core
from gui_custom_elements import UIGaugeMeter, UITrackCanvas
from track import Track
from cars import PlayerCar, AICar
from rl_model import A2CModel, REINFORCEModel
from numpy import clip, exp
from abc import ABC, abstractmethod
from collections import defaultdict


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

    def update(self, *args):
        super().update(*args)
        self._drive_car()

    def _drive_car(self):
        if not self.car.is_crashed:
            self.car.update()
            self.car.collision_detection(self.track)

    def event_handle(self):
        super().event_handle()
        game_core.pressed_buttons.clear()


class AICarSim(CarSimulation):
    def __init__(self, simulation_size = 12, state_size =15):
        super().__init__()
        self.reward_size = 9
        self.episode_num = 0


        self.sim_size = simulation_size
        self.state_size = state_size
        self.cars = []
        # self.rl_model = A2CModel(self.state_size)
        self.rl_model = REINFORCEModel(self.state_size)
        if game_core.load_model:
            self.load_model()
        self.actions = np.zeros([self.sim_size, 2], np.float32)
        self.is_crashed = [False] * self.sim_size
        self.stuck_timer = [0] * self.sim_size
        self.prev_progress = [0.0] * self.sim_size
        self.max_epoch_progress = [0.0] * self.sim_size
        self.stuck_limit = 1000
        self.historic_progress = [0]
        self.tick_speedup = pygame_gui.elements.UIButton(
            relative_rect=pygame.Rect((400, 600), (100, 30)),
            text="Click to slow",
            manager=self.gui_manager
        )
        self.save_ai_button = pygame_gui.elements.UIButton(
            relative_rect=pygame.Rect((400, 650), (100, 30)),
            text="Click to save",
            manager=self.gui_manager
        )

        # Rollback mechanism attributes
        self.performance_history = []  # Track recent episode performance
        self.history_window = 20  # Compare against last N episodes
        self.rollback_threshold = 0.8  # Rollback if avg reward drops by this factor
        self.best_hist_avg = 0
        self.max_laps = 5
        self.best_model_params = {'actor': self.rl_model.actor.get_params()}
        if hasattr(self.rl_model, 'critic'):
            self.best_model_params['critic'] = self.rl_model.critic.get_params()

        game_core.debug_elements["rays"] = [0] * simulation_size
        self._reset_cars()
        self.time_step_rewards_breakdown = []
        self.epoch_rewards_breakdown = np.zeros([self.sim_size, self.reward_size], float)

    def update(self):
        if all(self.is_crashed):
            if len(self.rl_model.rewards) > 0:
                self.episode_num += 1
                print(f"Episode:{self.episode_num}")
                avg_epoch_rewards = np.round(self.epoch_rewards_breakdown.mean(axis=0), 3)
                self.time_step_rewards_breakdown = np.array(self.time_step_rewards_breakdown)
                avg_time_step_rewards = np.round(self.time_step_rewards_breakdown.mean(axis=0), 3)
                print(f"Epoch rewards: {avg_epoch_rewards.tolist()}")
                print(f"Time step rewards: {avg_time_step_rewards.tolist()}")
                print()

                # avg_total_reward = np.sum(avg_epoch_rewards)
                avg_progress = np.mean(self.max_epoch_progress)
                self.cache_performance(avg_progress, self.max_epoch_progress)
                self.check_rollback()

                self.epoch_rewards_breakdown = np.zeros([self.sim_size, self.reward_size], float)
                self.time_step_rewards_breakdown = []
                self._reinitialise_car_simulation()
                return

        states = np.zeros((self.sim_size, self.state_size), np.float32)
        rewards = np.zeros(self.sim_size, np.float32)
        for i, car in enumerate(self.cars):
            if not self.is_crashed[i]:
                is_stuck = False
                car.collision_detection(self.track)
                sensors = car.ray_cast(self.track, i)

                progress, is_lap_finished = car.update_and_get_progress(self.track.track_spine)

                if progress - self.prev_progress[i] < 0.001:
                    self.stuck_timer[i] += 1
                else:
                    self.stuck_timer[i] = 0

                if self.stuck_timer[i] > self.stuck_limit:
                    car.is_crashed = True
                    is_stuck = True

                if car.is_crashed:
                    self.is_crashed[i] = True

                rewards[i], rewards_array = car.compute_reward(is_stuck, is_lap_finished, self.prev_progress[i],
                                                               np.mean(self.historic_progress), sensors)

                if car.progress % 1 >= self.max_laps:
                    self.is_crashed[i] = True
                    rewards[i] += 20
                    rewards_array.append(20)

                reward_breakdown = np.array(rewards_array)
                rewards[i] = np.clip(rewards[i], -10, 100)
                self.epoch_rewards_breakdown[i] += reward_breakdown
                self.time_step_rewards_breakdown.append(reward_breakdown)

                self.max_epoch_progress[i] = max(progress, self.max_epoch_progress[i])
                self.prev_progress[i] = progress

                states[i] = self.get_state_vector(sensors, car)

        self.actions, pre_squash = self.rl_model.get_stochastic_actions(states)

        self.rl_model.update_trajectory(states, pre_squash, rewards)
        for i, car in enumerate(self.cars):
            if not self.is_crashed[i]:
                car.update(self.actions[i])
        self.gui_manager.update(1/game_core.frame_rate)

    def get_state_vector(self, sensors, car):
        return sensors + [car.velocity.magnitude()/car.max_speed, car.steer/car.max_steer]


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


        if self.save_ai_button in game_core.pressed_buttons:
            self.save_model()
        game_core.pressed_buttons.clear()


    def check_rollback(self):
        if len(self.performance_history) < self.history_window:
            return
        recent_performance = np.mean(self.performance_history[-self.history_window:])
        if recent_performance < self.rollback_threshold * self.best_hist_avg:
            self.rl_model.actor.set_params(self.best_model_params['actor'])
            if hasattr(self.rl_model, 'critic'):
                self.rl_model.critic.set_params(self.best_model_params['critic'])
            print()
            print("----xx--------xx--------xx--------xx--------xx--------xx--------xx--------xx----")
            print("Model Rolled Back")
            print("----xx--------xx--------xx--------xx--------xx--------xx--------xx--------xx----")
            print()
            self.performance_history.clear()

    def cache_performance(self, avg_total, epoch_progresses):
        self.performance_history.append(avg_total)
        if len(self.performance_history) > 2* self.history_window:
            self.performance_history.pop(0)
        if len(self.performance_history) >= self.history_window:
            relevant_hist = self.performance_history[-self.history_window:]
        else:
            relevant_hist = self.performance_history
        historic_avg = np.mean(relevant_hist)
        if historic_avg > self.best_hist_avg and min(relevant_hist) > 0.9 * historic_avg:
            self.best_hist_avg = historic_avg
            cache_params = {'actor': self.rl_model.actor.get_params()}
            if hasattr(self.rl_model, 'critic'):
                cache_params['critic'] = self.rl_model.critic.get_params()
            self.best_model_params = cache_params

        self.historic_progress.append(max(epoch_progresses))
        if len(self.historic_progress) > self.history_window:
            self.historic_progress.pop(0)


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
        self.save_ai_button = pygame_gui.elements.UIButton(
            relative_rect=pygame.Rect((400, 650), (100, 30)),
            text="Click to save",
            manager=self.gui_manager
        )

    def save_model(self, filename = game_core.current_model):

        save_data = {
            'model_type': 'A2C' if hasattr(self.rl_model, 'critic') else 'REINFORCE',
            'actor_params': self.rl_model.actor.get_params(),
            'episode_num': self.episode_num,
            'best_model_params': self.best_model_params
        }

        if save_data['model_type'] == 'A2C':
            save_data['critic_params'] = self.rl_model.critic.get_params()

        np.save(f"Saved Model/{filename}.npy", save_data, allow_pickle=True)

    def load_model(self, filename= game_core.current_model):
        save_data = np.load(f"Saved Model/{filename}.npy", allow_pickle=True).item()

        self.rl_model.actor.set_params(save_data['actor_params'])
        if hasattr(self.rl_model, 'critic') and 'critic_params' in save_data:
            self.rl_model.critic.set_params(save_data['critic_params'])
        self.best_model_params = save_data['best_model_params']
        self.episode_num = save_data['episode_num']


    def _reinitialise_car_simulation(self):
        self.rl_model.update_params()
        self.max_epoch_progress = [0.0] * self.sim_size
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
            relative_rect=pygame.Rect((0, 50), (game_core.screen_dimensions[0], game_core.screen_dimensions[1] - 50) ),
            manager=self.gui_manager
        )

        self.anchor_toggle_button = pygame_gui.elements.UIButton(
            relative_rect=pygame.Rect((10, 10), (50, 40)),
            text="A",
            manager=self.gui_manager
        )

        self.width_toggle_button = pygame_gui.elements.UIButton(
            relative_rect=pygame.Rect((10, 60), (50, 40)),
            text="W",
            manager=self.gui_manager
        )
        self.width_slider = None
        self.width_label = None
        self.save_button = pygame_gui.elements.UIButton(
            relative_rect=pygame.Rect((100, 100), (100, 30)),
            text="Click to save",
            manager=self.gui_manager
        )

        self.load_button = pygame_gui.elements.UIButton(
            relative_rect=pygame.Rect((200, 100), (100, 30)),
            text="Click to load",
            manager=self.gui_manager
        )

        self.clear_button = pygame_gui.elements.UIButton(
            relative_rect=pygame.Rect((300, 100), (100, 30)),
            text="Click to clear",
            manager=self.gui_manager
        )

        self.start_car_sim_button = pygame_gui.elements.UIButton(
            relative_rect=pygame.Rect((300, 200), (100, 30)),
            text="Click to start sim",
            manager=self.gui_manager
        )

        self.canvas.anchor_mode = False
        self.canvas.width_mode = False

    def update(self):
        self.gui_manager.update(1 / game_core.frame_rate / game_core.tick_speedup)
        if self.canvas.mode == "width":
            self.changing_widths()

    def changing_widths(self):
        if self.canvas.selected_point and self.canvas.selected_point[0] == 'w':
            width_point_index = self.canvas.selected_point[1]
            if self.width_slider is None:
                start_val = self.canvas.widths_dict[width_point_index]
                self.width_slider = pygame_gui.elements.UIHorizontalSlider(
                    relative_rect=pygame.Rect((100, 10), (300, 30)),
                    start_value=start_val,
                    value_range=(60, 100),
                    manager=self.gui_manager
                )
                self.width_label = pygame_gui.elements.UILabel(
                    relative_rect=pygame.Rect((410, 10), (60, 30)),
                    text=str(int(start_val)),
                    manager=self.gui_manager
                )
            else:
                val = int(round(self.width_slider.get_current_value()))
                self.canvas.widths_dict[width_point_index] = val
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
            self.anchor_toggle_button.disable()
            self.width_toggle_button.enable()

        if self.width_toggle_button in game_core.pressed_buttons:
            self.canvas.mode = "width"
            self.width_toggle_button.disable()
            self.anchor_toggle_button.enable()

        if self.save_button in game_core.pressed_buttons:
            self.save_track()

        if self.load_button in game_core.pressed_buttons:
            self.load_track()

        if self.clear_button in game_core.pressed_buttons:
            self.canvas.kill()
            self.canvas = UITrackCanvas(
                relative_rect=pygame.Rect((0, 50),
                                          (game_core.screen_dimensions[0], game_core.screen_dimensions[1] - 50)),
                manager=self.gui_manager
            )
        if self.start_car_sim_button in game_core.pressed_buttons:
            game_core.set_game_mode(AICarSim)

        game_core.pressed_buttons.clear()

    def save_track(self, filename=game_core.current_track):
        data = {
            "anchors": [(point.x, point.y) for point in self.canvas.anchor_points],
            "controls": [(point.x, point.y) for point in self.canvas.control_points],
            "widths": {str(k): v for k, v in self.canvas.widths_dict.items()}
        }
        with open(f"Track/{filename}.json", "w") as file:
            json.dump(data, file, indent=2)

    def load_track(self, filename=game_core.current_track):
        self.canvas.anchor_points, self.canvas.control_points, self.canvas.widths_dict = r.load_bezier_track(filename)

