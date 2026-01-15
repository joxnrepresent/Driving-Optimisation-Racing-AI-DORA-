import math
import os.path
import pickle
import json

import numpy
import numpy as np
import pygame
import pygame_gui
from pygame_gui.core.colour_parser import is_float_str
from sklearn.externals.array_api_extra.testing import override
from pygame_gui.elements import UIPanel, UILabel, UIButton

import resources as r
from resources import game_core, RaceTimeManager
from gui_custom_elements import UIGaugeMeter, UITrackCanvas, UIFileSelector, UIOptionSelector
from track import Track
from cars import PlayerCar, AICar
from rl_model import MCACModel, REINFORCEModel
from numpy import clip, exp
from abc import ABC, abstractmethod
from collections import defaultdict

"""This module contains classes """
#---------------------------------------------------------------------------------------------------------------------#

class GameMode(ABC):
    """Base class for all game modes."""

    def __init__(self):
        self.game_sprites = game_core.game_sprites
        self.debug_elements = game_core.debug_elements
        self.gui_manager = game_core.gui_manager

        try:
            self.gui_manager.get_theme().load_theme('DORA_theme.json')
        except:
            pass

        self.control_panel_visible = True

        self.control_panel = pygame_gui.elements.UIPanel(
                            relative_rect=pygame.Rect((50,50), (300, 500)),
                            manager=self.gui_manager,
                            object_id='#control_panel'
                            )

        self.minimise_button = pygame_gui.elements.UIButton(
                                relative_rect=pygame.Rect((370,50), (50, 50)),
                                text='–',
                                manager=self.gui_manager,
                                object_id='#minimise_button'
                                )


    def update(self, *args):
        self.game_sprites.update(*args)
        self.gui_manager.update(1 / game_core.frame_rate)

    def minimise_control_panel(self):
        if not self.control_panel_visible:
            return

        self.control_panel.hide()
        self.control_panel_visible = False

        self.minimise_button.set_relative_position((50, 50))
        self.minimise_button.set_text('+')
        # self.control_panel.show()


    def restore_control_panel(self):
        if self.control_panel_visible:
            return

        self.control_panel.show()
        self.control_panel_visible = True

        self.minimise_button.set_relative_position((370,50))
        self.minimise_button.set_text('–')

    def event_handle(self):
        if self.minimise_button in game_core.pressed_buttons:
            if self.control_panel_visible:
                self.minimise_control_panel()
            else:
                self.restore_control_panel()

        game_core.pressed_buttons.clear()

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
            text='AI RACING SIMULATOR',
            manager=self.gui_manager,
            container=self.options_panel,
            object_id='#title_label'
        )

        self.subtitle = UILabel(
            relative_rect=pygame.Rect(screen_w // 2 - 250, 190, 500, 40),
            text='Reinforcement Learning Track Simulation',
            manager=self.gui_manager,
            container=self.options_panel,
            object_id='#subtitle_label'
        )

        self.track_maker_button = UIButton(
            relative_rect=pygame.Rect(screen_w // 2 - 450, 300, 400, 70),
            text='Track Editor',
            manager=self.gui_manager,
            container=self.options_panel,
            object_id='#menu_button'
        )

        self.solo_sim_button = UIButton(
            relative_rect=pygame.Rect(screen_w // 2 + 50, 300, 400, 70),
            text='Manual Racing Mode',
            manager=self.gui_manager,
            container=self.options_panel,
            object_id='#menu_button'
        )

        self.racing_sim_button = UIButton(
            relative_rect=pygame.Rect(screen_w // 2 - 450, 420, 400, 70),
            text='Race AI Mode',
            manager=self.gui_manager,
            container=self.options_panel,
            object_id='#menu_button'
        )

        self.ai_sim_button = UIButton(
            relative_rect=pygame.Rect(screen_w // 2 + 50, 420, 400, 70),
            text='AI Training Mode',
            manager=self.gui_manager,
            container=self.options_panel,
            object_id='#menu_button'
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

    def event_handle(self):

        if self.track_maker_button in game_core.pressed_buttons:
            game_core.set_game_mode(TrackMakerUI)

        elif self.solo_sim_button in game_core.pressed_buttons:
            game_core.set_game_mode(SoloCarSim)

        elif self.racing_sim_button in game_core.pressed_buttons:
            game_core.set_game_mode(RacingSim)

        elif self.ai_sim_button in game_core.pressed_buttons:
            game_core.set_game_mode(AICarSim)

        elif self.exit_button in game_core.pressed_buttons:
            pygame.quit()
            exit()

        super().event_handle()


class CarSimulation(GameMode):
    """Base class for car simulation modes."""

    def __init__(self, num_of_cars):
        super().__init__()

        # Track setup
        self.track_loader = UIFileSelector(pygame.Rect((0, 0), game_core.screen_dimensions),
                                           self.gui_manager,
                                           "Track",
                                           "load",
                                           callback=lambda filename: self.load_track(filename),
                                           return_mode= MainMenu)

        self.time_manager = RaceTimeManager()

        # Car management
        self.cars = []
        self.num_of_cars = num_of_cars
        self.is_crashed = [False] * num_of_cars

        # GUI setup
        self._init_gui()

    def load_track(self, filename):

        self.track = Track(game_core.current_track)
        self.game_sprites.add(self.track)
        game_core.starting_position = self.track.track_spine[0]

        self._reset_cars()


    def _init_gui(self):
        pass

    def update(self):
        """Update simulation state."""
        if all(self.is_crashed):
            self._reinitialise_simulation()
            return

        # Update active cars
        for i, car in enumerate(self.cars):
            if self.is_crashed[i]:
                continue

            car.collision_detection(self.track)

            if car.is_crashed:
                self.is_crashed[i] = True
        super().update()

    def event_handle(self):
        """Handle reset button press."""
        if game_core.is_paused:
            return

        if self.reset_button in game_core.pressed_buttons:
            for i in range(len(self.cars)):
                self.is_crashed[i] = True

        super().event_handle()

    def _reset_environment(self):
        """Reset the simulation environment."""
        for car in self.cars:
            if isinstance(car, PlayerCar):
                car.delete_control_panel()
        self.cars.clear()
        self.game_sprites.empty()
        self.game_sprites.add(self.track)
        self.is_crashed = [False] * self.num_of_cars

    def _reset_gui(self):
        """Reset GUI elements."""
        pass

    @abstractmethod
    def _reset_cars(self):
        """Reset cars. Must be implemented by subclasses."""
        pass

    def _reinitialise_simulation(self):
        """Reinitialize the entire simulation."""
        self._reset_gui()
        self._reset_environment()
        self._reset_cars()


class SoloCarSim(CarSimulation):
    """Player-controlled racing simulation."""

    def __init__(self):
        super().__init__(num_of_cars=1)

        self.reset_button = pygame_gui.elements.UIButton(
            relative_rect=pygame.Rect((200, 130), (90, 40)),
            text="Reset Car",
            manager=self.gui_manager,
            container=self.control_panel,
            object_id= '#panel_button'
        )

        self.info_label = pygame_gui.elements.UILabel(
            relative_rect=pygame.Rect((10, 220), (280, 30)),
            text='W: Throttle | S: Brake',
            manager=self.gui_manager,
            container=self.control_panel,
            object_id = '#info_label'
        )

        self.menu_button = pygame_gui.elements.UIButton(
            relative_rect=pygame.Rect((10, 420), (280, 60)),
            text="Return to Menu",
            manager=self.gui_manager,
            container=self.control_panel,
            object_id='#exit_button'
        )


    def _reset_cars(self):
        """Reset player car."""
        # Remove old car if exists
        if self.cars:
            self.game_sprites.remove(self.cars[0])
            self.cars[0].throttle_and_braking_meter.kill()
            self.cars[0].steering_slider.kill()
            self.cars[0].speedometer.kill()

        # Create new player car
        self.cars = [PlayerCar(game_core.starting_position)]
        self.game_sprites.add(self.cars[0])
        self.debug_elements["hitboxes"].append(self.cars[0].hitbox)

    def event_handle(self):
        """Handle events and clear button presses."""

        if self.menu_button in game_core.pressed_buttons:
            game_core.set_game_mode(MainMenu)

        super().event_handle()

    def update(self, *args):
        super().update()


class RacingSim(CarSimulation):
    """Player-controlled racing simulation."""

    def __init__(self, state_size=15):
        super().__init__(num_of_cars=2)

        self.state_size = state_size
        self.num_of_laps = 0
        self.start_time = None
        self.end_times = [None, None]
        self.lap_times = None


        self.reset_button = pygame_gui.elements.UIButton(
            relative_rect=pygame.Rect((10, 50), (130, 40)),
            text="Reset Race",
            manager=self.gui_manager,
            container=self.control_panel,
            object_id='#panel_button'
        )

        self.load_model_button = pygame_gui.elements.UIButton(
            relative_rect=pygame.Rect((150, 50), (140, 40)),
            text="Load Model",
            manager=self.gui_manager,
            container=self.control_panel,
            object_id='#panel_button'
        )

        self.info_label = pygame_gui.elements.UILabel(
            relative_rect=pygame.Rect((10, 100), (280, 30)),
            text='W: Throttle | S: Brake',
            manager=self.gui_manager,
            container=self.control_panel,
            object_id='#info_label'
        )

        self.menu_button = pygame_gui.elements.UIButton(
            relative_rect=pygame.Rect((10, 420), (280, 60)),
            text="Return to Menu",
            manager=self.gui_manager,
            container=self.control_panel,
            object_id='#exit_button'
        )

        self.rl_model = None
        self.model_loader = None
        self.num_of_laps_setter = None

    def load_track(self, filename):
        super().load_track(filename)
        self.model_loader = UIFileSelector(pygame.Rect((0, 0), game_core.screen_dimensions),
                                           self.gui_manager,
                                           "Saved Model",
                                           "load",
                                           callback=lambda filename: self.load_model(filename),
                                           return_mode=MainMenu)

    def update(self, *args):
        self.gui_manager.update(1/game_core.frame_rate)
        if not self.rl_model or r.game_core.is_paused:
            return

        if None not in self.end_times:
            self._reset_cars()

        # Update active cars
        for i, car in enumerate(self.cars):
            if self.is_crashed[i]:
                car.set_car_position(game_core.starting_position, 270)
                car.set_progress(math.floor(car.progress))

            progress, is_lap_completed = car.update_and_get_progress(self.track.track_spine)
            laps_completed = math.floor(progress)
            if is_lap_completed:
                self.lap_times[laps_completed - 1][i] = pygame.time.get_ticks()
            if laps_completed == self.num_of_laps:
                self.end_times[i] = pygame.time.get_ticks()

            car.collision_detection(self.track)

            if car.is_crashed:
                self.is_crashed[i] = True

        sensors = self.cars[1].ray_cast(self.track, 1)
        state = self._get_state_vector(sensors, self.cars[1])
        actions = self.rl_model.get_deterministic_actions(state)

        if not self.is_crashed[0]:
            self.cars[0].update()

        if not self.is_crashed[1]:
            self.cars[1].update(actions)


    def _get_state_vector(self, sensors, car):
        return sensors + [
            car.velocity.magnitude() / car.max_speed,
            car.steer / car.max_steer
        ]

    def _reset_cars(self):
        if self.cars:
            self.start_time = pygame.time.get_ticks()
            self.lap_times = [[None, None]] * self.num_of_laps
            self.end_times = [None, None]

            self.cars[0].throttle_and_braking_meter.kill()
            self.cars[0].steering_slider.kill()
            self.cars[0].speedometer.kill()
            self.game_sprites.remove(self.cars[0], self.cars[1])
            self.cars = [PlayerCar(game_core.starting_position), AICar(game_core.starting_position)]
            self.game_sprites.add(self.cars[0], self.cars[1])
            self.debug_elements["hitboxes"].extend(self.cars[0].hitbox, self.cars[1].hitbox)

    def event_handle(self):
        if self.menu_button in game_core.pressed_buttons:
            game_core.set_game_mode(MainMenu)
        super().event_handle()

    def load_model(self, filename=None):
        save_data = np.load(f"Saved Model/{filename}.npy", allow_pickle=True).item()
        self.rl_model = REINFORCEModel(self.state_size)
        self.rl_model.actor.set_params(save_data['actor_params'])

        self.num_of_laps_setter = UIOptionSelector(pygame.Rect((0, 0), game_core.screen_dimensions),
                                           self.gui_manager,
                                           ["1", "2", "3", "4", "5", "6", "7", "8", "9", "10"],
                                           callback=lambda num_of_laps: self.set_num_of_laps(num_of_laps),
                                           return_mode=MainMenu)


    def set_num_of_laps(self, num_of_laps):
        self.num_of_laps = int(num_of_laps)
        self.start_time = pygame.time.get_ticks()
        self.lap_times = [[None, None]] * num_of_laps


class AICarSim(CarSimulation):
    """AI-powered car simulation with reinforcement learning."""

    def __init__(self, simulation_size=12, state_size=15):
        super().__init__(num_of_cars=simulation_size)
        self.episode_num = 0

        self.title_label = pygame_gui.elements.UILabel(
            relative_rect=pygame.Rect((10, 10), (280, 30)),
            text='AI Training Controls',
            manager=self.gui_manager,
            container=self.control_panel,
            object_id='#panel_title'
        )

        self.episodes_completed_label = pygame_gui.elements.UILabel(
            relative_rect=pygame.Rect((10, 50), (280, 25)),
            text=f'Episode: {self.episode_num}',
            manager=self.gui_manager,
            container=self.control_panel,
            object_id='#info_label'
        )

        self.speed_label = pygame_gui.elements.UILabel(
            relative_rect=pygame.Rect((10, 80), (280, 25)),
            text=f'Speed: {game_core.tick_speedup}x',
            manager=self.gui_manager,
            container=self.control_panel,
            object_id='#info_label'
        )

        self.reset_button = pygame_gui.elements.UIButton(
            relative_rect=pygame.Rect((10, 120), (135, 40)),
            text="Reset Training",
            manager=self.gui_manager,
            container=self.control_panel,
            object_id='#panel_button'
        )

        self.tick_speedup = pygame_gui.elements.UIButton(
            relative_rect=pygame.Rect((155, 120), (135, 40)),
            text="Adjust Speed",
            manager=self.gui_manager,
            container=self.control_panel,
            object_id='#panel_button'
        )

        self.save_ai_button = pygame_gui.elements.UIButton(
            relative_rect=pygame.Rect((10, 175), (280, 45)),
            text="Save Model",
            manager=self.gui_manager,
            container=self.control_panel,
            object_id='#panel_button'
        )

        self.menu_button = pygame_gui.elements.UIButton(
            relative_rect=pygame.Rect((10, 440), (280, 50)),
            text="Return to Menu",
            manager=self.gui_manager,
            container=self.control_panel,
            object_id='#exit_button'
        )


        # AI Configuration
        self.state_size = state_size
        self.reward_size = 9

        # Simulation state
        self.paused = True
        self.actions = np.zeros([self.num_of_cars, 2], np.float32)

        # Progress tracking
        self.stuck_timer = [0] * self.num_of_cars
        self.prev_progress = [0.0] * self.num_of_cars
        self.max_epoch_progress = [0.0] * self.num_of_cars
        self.stuck_limit = 1000
        self.historic_progress = [0]
        self.max_laps = 5

        # Reward tracking
        self.time_step_rewards_breakdown = []
        self.epoch_rewards_breakdown = np.zeros([self.num_of_cars, self.reward_size], float)

        # Model management
        self.performance_history = []
        self.history_window = 20
        self.rollback_threshold = 0.8
        self.best_hist_avg = 0
        self.best_model_params = None



        self.model_type_selector = None
        self.model_loader = None
        self.num_of_laps_setter = None
        self.model_saver = None
        self.model_type = None
        self.rl_model = None

        # Debug setup
        game_core.debug_elements["rays"] = [0] * simulation_size


    def load_track(self, filename):
        super().load_track(filename)
        self.model_type_selector = UIOptionSelector(pygame.Rect((0, 0), game_core.screen_dimensions),
                                           self.gui_manager,
                                           ["REINFORCE", "Monte Carlo Actor Critic (MCAC)"],
                                           callback=lambda model_type: self.set_model_type(model_type),
                                           return_mode=MainMenu)


    def set_model_type(self, model_type):
        self.model_type = model_type
        self.model_loader = UIFileSelector(pygame.Rect((0, 0), game_core.screen_dimensions),
                                           self.gui_manager,
                                           "Saved Model",
                                           "load",
                                           callback=lambda filename: self.load_model(filename))

    def _init_gui(self):
        """Initialize GUI elements including AI-specific controls."""
        super()._init_gui()


    def update(self, *args):
        """Update AI simulation."""
        if self.rl_model is None:
            if game_core.current_model is not None:
                self.load_model(game_core.current_model)
            else:
                self.rl_model = REINFORCEModel(self.state_size)
            self.best_model_params = {'actor': self.rl_model.actor.get_params()}
            if hasattr(self.rl_model, 'critic'):
                self.best_model_params['critic'] = self.rl_model.critic.get_params()

        if all(self.is_crashed):
            self._handle_episode_end()
            return

        self.gui_manager.update(1 / game_core.frame_rate)
        if not game_core.is_paused:
            self._update_ai_step()


    def _handle_episode_end(self):
        """Handle end of episode - logging, rollback check, reset."""
        if len(self.rl_model.rewards) > 0:
            self.episode_num += 1
            self.episodes_completed_label.set_text(f'Episodes completed: {self.episode_num}')
            self._log_episode_results()

            avg_progress = np.mean(self.max_epoch_progress)
            self._cache_performance(avg_progress, self.max_epoch_progress)
            self._check_rollback()

            # Reset reward tracking
            self.epoch_rewards_breakdown = np.zeros([self.num_of_cars, self.reward_size], float)
            self.time_step_rewards_breakdown = []

        self._reinitialise_simulation()

    def _log_episode_results(self):
        """Log episode statistics."""
        print(f"Episode: {self.episode_num}")

        avg_epoch_rewards = np.round(self.epoch_rewards_breakdown.mean(axis=0), 3)
        print(f"Epoch rewards: {avg_epoch_rewards.tolist()}")

        if self.time_step_rewards_breakdown:
            time_step_rewards = np.array(self.time_step_rewards_breakdown)
            avg_time_step_rewards = np.round(time_step_rewards.mean(axis=0), 3)
            print(f"Time step rewards: {avg_time_step_rewards.tolist()}")

        print()

    def _update_ai_step(self):
        """Perform one AI training step."""
        # Get current states
        states = np.zeros((self.num_of_cars, self.state_size), np.float32)

        for i, car in enumerate(self.cars):
            if not self.is_crashed[i]:
                sensors = car.ray_cast(self.track, i)
                states[i] = self._get_state_vector(sensors, car)

        # Get actions from model based on current states
        self.actions, pre_squash = self.rl_model.get_stochastic_actions(states)

        # Apply actions to environment
        for i, car in enumerate(self.cars):
            if not self.is_crashed[i]:
                car.update(self.actions[i])

        # NOW observe the rewards from taking those actions
        rewards = np.zeros(self.num_of_cars, np.float32)

        for i, car in enumerate(self.cars):
            if not self.is_crashed[i]:
                rewards[i] = self._process_car_reward(i, car)

        # Store the correct tuple: (state, action, reward_after_action)
        self.rl_model.update_trajectory(states, pre_squash, rewards)


    def _process_car_reward(self, car_index, car):
        """Process a single car's reward after action is taken."""
        i = car_index

        # Collision detection
        car.collision_detection(self.track)

        # Update progress
        progress, is_lap_finished = car.update_and_get_progress(self.track.track_spine)

        # Check if stuck
        is_stuck = self._check_if_stuck(i, progress)

        if car.is_crashed:
            self.is_crashed[i] = True

        # Get sensors for reward computation
        sensors = car.ray_cast(self.track, i)

        # Compute reward
        reward, rewards_array = car.compute_reward(
            is_stuck,
            is_lap_finished,
            self.prev_progress[i],
            np.mean(self.historic_progress),
            sensors
        )

        # Check if completed max laps
        if car.progress >= self.max_laps:
            self.is_crashed[i] = True
            reward += 20
            rewards_array[-1] += 20

        # Track rewards
        reward_breakdown = np.array(rewards_array)
        clipped_reward = np.clip(reward, -10, 100) / 10
        self.epoch_rewards_breakdown[i] += reward_breakdown
        self.time_step_rewards_breakdown.append(reward_breakdown)

        # Update progress tracking
        self.max_epoch_progress[i] = max(progress, self.max_epoch_progress[i])
        self.prev_progress[i] = progress

        return clipped_reward

    def _check_if_stuck(self, car_index, current_progress):
        """Check if car is stuck and update stuck timer."""
        i = car_index

        if current_progress - self.prev_progress[i] < 0.001:
            self.stuck_timer[i] += 1
        else:
            self.stuck_timer[i] = 0

        if self.stuck_timer[i] > self.stuck_limit:
            self.cars[i].is_crashed = True
            return True

        return False

    def _get_state_vector(self, sensors, car):
        """Construct state vector from sensors and car state."""
        return sensors + [
            car.velocity.magnitude() / car.max_speed,
            car.steer / car.max_steer
        ]

    def event_handle(self):
        """Handle user input events."""
        # Reset simulation

        if self.reset_button in game_core.pressed_buttons:
            self._reinitialise_simulation()

        # Speed control
        if self.tick_speedup in game_core.pressed_buttons:
            self._toggle_speed()

        # Save model
        if self.save_ai_button in game_core.pressed_buttons:
            self.model_saver = UIFileSelector(pygame.Rect((0, 0), game_core.screen_dimensions),
                                              self.gui_manager,
                                              "Saved Model",
                                              "save",
                                              lambda filename: self.save_model(filename))

        super().event_handle()

    def _toggle_speed(self):
        """Toggle simulation speed."""
        if game_core.tick_speedup == 1:
            game_core.tick_speedup = 50
        elif game_core.tick_speedup == 50:
            game_core.tick_speedup = 100
        else:
            game_core.tick_speedup = 1

        self.tick_speedup.set_text(f"Speed: {game_core.tick_speedup}")

    def _check_rollback(self):
        """Check if model should be rolled back due to poor performance."""
        if len(self.performance_history) < self.history_window:
            return

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

            self.performance_history.clear()

    def _cache_performance(self, avg_total, epoch_progresses):
        """Cache performance metrics and update best model if improved."""
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
        if historic_avg > self.best_hist_avg and min(relevant_hist) > 0.8 * historic_avg:
            self.best_hist_avg = historic_avg
            cache_params = {'actor': self.rl_model.actor.get_params()}
            if hasattr(self.rl_model, 'critic'):
                cache_params['critic'] = self.rl_model.critic.get_params()
            self.best_model_params = cache_params

        # Update historic progress
        self.historic_progress.append(max(epoch_progresses))
        if len(self.historic_progress) > self.history_window:
            self.historic_progress.pop(0)

    def _reset_cars(self):
        """Reset all AI cars."""
        self.cars.clear()
        self.stuck_timer = [0] * self.num_of_cars
        self.prev_progress = [0.0] * self.num_of_cars

        for i in range(self.num_of_cars):
            car = AICar(game_core.starting_position)
            self.cars.append(car)
            self.debug_elements["hitboxes"].append(car.hitbox)
            self.debug_elements["AABB"].append(car)
            self.game_sprites.add(car)

    def _reinitialise_simulation(self):
        """Reinitialize simulation and update model parameters."""
        self.rl_model.update_params()
        self.max_epoch_progress = [0.0] * self.num_of_cars
        super()._reinitialise_simulation()

    def save_model(self, filename=None):
        """Save the current model to disk."""
        if filename is None:
            filename = game_core.current_model

        save_data = {
            'model_type': 'MCAC' if hasattr(self.rl_model, 'critic') else 'REINFORCE',
            'actor_params': self.rl_model.actor.get_params(),
            'episode_num': self.episode_num,
            'best_model_params': self.best_model_params
        }

        if save_data['model_type'] == 'MCAC':
            save_data['critic_params'] = self.rl_model.critic.get_params()

        np.save(f"Saved Model/{filename}.npy", save_data, allow_pickle=True)

    def load_model(self, filename=None):
        """Load a model from disk."""
        if filename is None:
            filename = game_core.current_model

        save_data = np.load(f"Saved Model/{filename}.npy", allow_pickle=True).item()

        if save_data['model_type'] == 'REINFORCE':
            self.rl_model = REINFORCEModel(self.state_size)
        else:
            self.rl_model = MCACModel(self.state_size)

        self.rl_model.actor.set_params(save_data['actor_params'])

        if hasattr(self.rl_model, 'critic') and 'critic_params' in save_data:
            self.rl_model.critic.set_params(save_data['critic_params'])

        self.best_model_params = save_data['best_model_params']
        self.episode_num = save_data['episode_num']

"""
Creates an interface that allows the user to create, save and draw custom tracks.
"""


class TrackMakerUI(GameMode):
    """
    Class stores the variables, objects and methods needed to run the track maker.
    """

    def __init__(self):
        super().__init__()

        self.canvas = UITrackCanvas(
            relative_rect=pygame.Rect((0, 0), game_core.screen_dimensions),
            manager=self.gui_manager
        )

        self.title_label = pygame_gui.elements.UILabel(
            relative_rect=pygame.Rect((10, 10), (280, 30)),
            text='Track Editor',
            manager=self.gui_manager,
            container=self.control_panel,
            object_id='#panel_title'
        )

        self.anchor_toggle_button = pygame_gui.elements.UIButton(
            relative_rect=pygame.Rect((10, 50), (140, 40)),
            text="Anchor Mode",
            manager=self.gui_manager,
            container=self.control_panel,
            object_id='#panel_button'
        )
        self.anchor_toggle_button.disable()

        self.width_toggle_button = pygame_gui.elements.UIButton(
            relative_rect=pygame.Rect((150, 50), (140, 40)),
            text="Width Mode",
            manager=self.gui_manager,
            container=self.control_panel,
            object_id='#panel_button'
        )

        self.toggle_handles_button = pygame_gui.elements.UIButton(
            relative_rect=pygame.Rect((10, 100), (140, 40)),
            text="Toggle Handles",
            manager=self.gui_manager,
            container=self.control_panel,
            object_id='#panel_button'
        )

        self.clear_button = pygame_gui.elements.UIButton(
            relative_rect=pygame.Rect((150, 100), (140, 40)),
            text="Clear Track",
            manager=self.gui_manager,
            container=self.control_panel,
            object_id='#panel_button'
        )

        self.save_button = pygame_gui.elements.UIButton(
            relative_rect=pygame.Rect((10, 150), (140, 40)),
            text="Save",
            manager=self.gui_manager,
            container=self.control_panel,
            object_id='#panel_button'
        )

        self.load_button = pygame_gui.elements.UIButton(
            relative_rect=pygame.Rect((150, 150), (140, 40)),
            text="Load",
            manager=self.gui_manager,
            container=self.control_panel,
            object_id='#panel_button'
        )

        self.info_label = pygame_gui.elements.UILabel(
            relative_rect=pygame.Rect((10, 300), (280, 80)),
            text='Click: Add\nBackspace: Delete\nDrag: Move',
            manager=self.gui_manager,
            container=self.control_panel,
            object_id='#info_label'
        )

        self.menu_button = pygame_gui.elements.UIButton(
            relative_rect=pygame.Rect((10, 420), (280, 60)),
            text="Return to Menu",
            manager=self.gui_manager,
            container=self.control_panel,
            object_id='#exit_button'
        )

        self.error_label = pygame_gui.elements.UILabel(
            relative_rect=pygame.Rect(
                (game_core.screen_dimensions[0] / 2 - 125, game_core.screen_dimensions[1] / 2 - 50), (250, 50)),
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
        super().update()
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
                    self.error_label.set_text("Track has overlapping/ kinked walls")
                elif self.canvas.validity_issues == "incomplete":
                    self.error_label.set_text("Track is incomplete")
                elif self.canvas.validity_issues == "too short":
                    self.error_label.set_text("Track is too short")
            else:
                self.save_track()

        if self.load_button in game_core.pressed_buttons:
            self.load_track()

        if self.clear_button in game_core.pressed_buttons:
            self.canvas.kill()
            self.canvas = UITrackCanvas(
                relative_rect=pygame.Rect((0,0), (game_core.screen_dimensions[0], game_core.screen_dimensions[1])),
                manager=self.gui_manager
            )

        if self.menu_button in game_core.pressed_buttons:
            game_core.set_game_mode(MainMenu)

        if self.toggle_handles_button in game_core.pressed_buttons:
            self.canvas.is_handles_enabled = not self.canvas.is_handles_enabled
            self.canvas.selected_point = None
            if self.canvas.is_handles_enabled:
                self.canvas.generate_all_controls()
            else:
                self.canvas.control_points.clear()

        super().event_handle()

    def save_track(self):
        self.file_selector = UIFileSelector(pygame.Rect((0, 0), game_core.screen_dimensions),
                                            self.gui_manager,
                                            "Track",
                                            "save",
                                            lambda filename: self.write_track_data(filename))

    def load_track(self):
        self.file_selector = UIFileSelector(pygame.Rect((0, 0), game_core.screen_dimensions),
                                            self.gui_manager,
                                            "Track",
                                            "load",
                                            lambda filename: self.load_track_data(filename))

    def write_track_data(self, filename=game_core.current_track):
        if not self.canvas.is_handles_enabled:
            self.canvas.generate_all_controls()
        data = {
            "anchors": [(point.x, point.y) for point in self.canvas.anchor_points],
            "controls": [(point.x, point.y) for point in self.canvas.control_points],
            "widths": {str(k): v for k, v in self.canvas.widths_dict.items()}
        }
        with open(f"Track/{filename}.json", "w") as file:
            json.dump(data, file, indent=2)

    def load_track_data(self, filename=game_core.current_track):
        self.canvas.anchor_points, self.canvas.control_points, self.canvas.widths_dict = r.load_bezier_track(filename)

