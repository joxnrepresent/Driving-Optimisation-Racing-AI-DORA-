import math
import os
from nea import resources as r
import re
import pygame
import numpy as np

from nea.track import SpatialHashGrid
from pygame.math import Vector2
from pygame_gui.core import UIElement
from pygame_gui.elements import (UIPanel, UILabel, UIButton, UISelectionList, UITextEntryLine,
                                 UIDropDownMenu, UITextBox, UIScrollingContainer)
import pygame_gui

"""
Custom GUI elements for racing simulation. Custom elements extend pygame_gui UIElement class to carry out more
specialised tasks

Classes:
    UIModelCreator: AI model hyperparameter and training environment configuration interface
    UIEndScreen: Race completion results display
    UIOptionSelector: Option selection dialogue
    UIFileBrowser: Used to save and load track and model files
    UIGaugeMeter: Gauge meter (used as speedometer)
    UITrackCanvas: Track editor canvas with bezier curve tool
"""

class UIModelCreator(UIElement):
    """
    RL model hyperparameter and environment configuration dialogue.

    Provides text input fields or dropdown menus for configuring REINFORCE or MCAC models with validation.
    Supports custom layer architectures and per-layer activation functions.

    Attributes:
        model_type (str): Model type (determines whether critic configurations are provided)
        return_mode (GameMode): Instance of Game Mode to return to
        panel (UIPanel): Main dialogue panel
        scroll_container (UIScrollingContainer): Container holding all configuration fields.
        [various input fields]: Text entry and dropdown fields for individual hyperparameters
        create_button (UIButton): Confirm and create model
        cancel_button (UIButton): Cancel configuration
    """

    def __init__(self, relative_rect, manager, model_type, callback, return_mode=None):
        """
        Initialise model configuration dialogue.

        Args:
            relative_rect (Rect): Config screen area
            manager (UIManager): GUI manager
            model_type (str): "REINFORCE" or "Monte Carlo Actor Critic (MCAC)"
            callback (callable): Function to call with validated config (creates a model with specified configurations)
            return_mode (GameMode): Mode to return to on cancel
        """
        super().__init__(relative_rect, manager, container=None, starting_height=999, layer_thickness=1)

        r.game_core.is_paused = True
        self.model_type = model_type
        self.callback = callback
        self.return_mode = return_mode

        self.image = pygame.Surface(relative_rect.size, pygame.SRCALPHA)
        self.image.fill((30, 30, 30, 120))

        panel_w = 500
        panel_h = 650
        self.panel = UIPanel(
            relative_rect=pygame.Rect(
                ((relative_rect.width - panel_w) / 2,
                 (relative_rect.height - panel_h) / 2),
                (panel_w, panel_h)
            ),
            manager=manager,
            starting_height=1000,
            object_id='#selector_panel'
        )


        self.title_label = UILabel(
            relative_rect=pygame.Rect((panel_w / 2 - 200, 10), (400, 30)),
            text=f"Configure {self.model_type} Model",
            manager=manager,
            container=self.panel,
            object_id='#title'
        )

        self.scroll_container = UIScrollingContainer(
            relative_rect=pygame.Rect((0, 40), (panel_w - 20, panel_h - 150)),
            manager=manager,
            container=self.panel,
        )

        y = 10
        row_height = 40
        x = 50
        label_width = 180
        text_input_width = 200

        UILabel(
            relative_rect=pygame.Rect((x, y), (label_width, 25)),
            text="Number of Cars:",
            manager=manager,
            container=self.scroll_container,
            object_id='#info_label'
        )
        self.num_of_cars_input = UITextEntryLine(
            relative_rect=pygame.Rect((x + label_width, y), (text_input_width, 30)),
            manager=manager,
            container=self.scroll_container
        )
        self.num_of_cars_input.set_text("12")
        y += row_height

        UILabel(
            relative_rect=pygame.Rect((x, y), (label_width, 25)),
            text="Gamma (Discount):",
            manager=manager,
            container=self.scroll_container,
            object_id='#info_label'
        )
        self.gamma_input = UITextEntryLine(
            relative_rect=pygame.Rect((x + label_width, y), (text_input_width, 30)),
            manager=manager,
            container=self.scroll_container
        )
        self.gamma_input.set_text("0.99")
        y += row_height

        UILabel(
            relative_rect=pygame.Rect((x, y), (label_width, 25)),
            text="Entropy Bonus:",
            manager=manager,
            container=self.scroll_container,
            object_id='#info_label'
        )
        self.entropy_input = UITextEntryLine(
            relative_rect=pygame.Rect((x + label_width, y), (text_input_width, 30)),
            manager=manager,
            container=self.scroll_container
        )
        self.entropy_input.set_text("0.005")
        y += row_height

        UILabel(
            relative_rect=pygame.Rect((x, y), (label_width, 25)),
            text="L2 Lambda:",
            manager=manager,
            container=self.scroll_container,
            object_id='#info_label'
        )
        self.l2_lambda_input = UITextEntryLine(
            relative_rect=pygame.Rect((x + label_width, y), (text_input_width, 30)),
            manager=manager,
            container=self.scroll_container
        )
        self.l2_lambda_input.set_text("0.01")
        y += row_height

        UILabel(
            relative_rect=pygame.Rect((x, y), (label_width, 25)),
            text="Init Log Std:",
            manager=manager,
            container=self.scroll_container,
            object_id='#info_label'
        )
        self.actor_init_log_std_input = UITextEntryLine(
            relative_rect=pygame.Rect((x + label_width, y), (text_input_width, 30)),
            manager=manager,
            container=self.scroll_container
        )
        self.actor_init_log_std_input.set_text("0.3")
        y += row_height

        UILabel(
            relative_rect=pygame.Rect((x, y), (label_width, 25)),
            text="Actor learning Rate:",
            manager=manager,
            container=self.scroll_container,
            object_id='#info_label'
        )
        self.actor_learning_rate_input = UITextEntryLine(
            relative_rect=pygame.Rect((x + label_width, y), (text_input_width, 30)),
            manager=manager,
            container=self.scroll_container
        )
        self.actor_learning_rate_input.set_text("0.003")
        y += row_height

        UILabel(
            relative_rect=pygame.Rect((x, y), (label_width, 25)),
            text="Actor adam Beta1:",
            manager=manager,
            container=self.scroll_container,
            object_id='#info_label'
        )
        self.actor_adam_beta1_input = UITextEntryLine(
            relative_rect=pygame.Rect((x + label_width, y), (text_input_width, 30)),
            manager=manager,
            container=self.scroll_container
        )
        self.actor_adam_beta1_input.set_text("0.9")
        y += row_height

        UILabel(
            relative_rect=pygame.Rect((x, y), (label_width, 25)),
            text="Actor adam Beta2:",
            manager=manager,
            container=self.scroll_container,
            object_id='#info_label'
        )
        self.actor_adam_beta2_input = UITextEntryLine(
            relative_rect=pygame.Rect((x + label_width, y), (text_input_width, 30)),
            manager=manager,
            container=self.scroll_container
        )
        self.actor_adam_beta2_input.set_text("0.999")
        y += row_height

        UILabel(
            relative_rect=pygame.Rect((x, y), (label_width, 25)),
            text="Actor Layers:",
            manager=manager,
            container=self.scroll_container,
            object_id='#info_label'
        )
        self.actor_layers_input = UITextEntryLine(
            relative_rect=pygame.Rect((x + label_width, y), (text_input_width, 30)),
            manager=manager,
            container=self.scroll_container
        )
        self.actor_layers_input.set_text("32,32,64,64,64,32,32")
        y += row_height

        UILabel(
            relative_rect=pygame.Rect((x, y), (label_width, 25)),
            text="Actor Activation:",
            manager=manager,
            container=self.scroll_container,
            object_id='#info_label'
        )
        self.actor_activation_dropdown = UIDropDownMenu(
            options_list=[
                'elu',
                'relu',
                'tanh',
                'linear',
                'Custom'
            ],
            starting_option='elu',
            relative_rect=pygame.Rect((x + label_width, y), (text_input_width, 30)),
            manager=manager,
            container=self.scroll_container
        )

        y += row_height

        self.actor_activation_custom_label = UILabel(
            relative_rect=pygame.Rect((x, y), (label_width + 20, 25)),
            text="Custom Actor Activations:",
            manager=manager,
            container=self.scroll_container,
            object_id='#info_label'
        )
        self.actor_activation_custom_input = UITextEntryLine(
            relative_rect=pygame.Rect((x + label_width + 20, y), (text_input_width, 30)),
            manager=manager,
            container=self.scroll_container
        )
        self.actor_activation_custom_input.set_text("elu,relu,tanh")
        self.actor_activation_custom_label.hide()
        self.actor_activation_custom_input.hide()
        y += row_height

        # MCAC-specific fields
        if self.model_type == "Monte Carlo Actor Critic (MCAC)":
            UILabel(
                relative_rect=pygame.Rect((x, y), (label_width, 25)),
                text="Init Critic Log Std:",
                manager=manager,
                container=self.scroll_container,
                object_id='#info_label'
            )
            self.critic_init_log_std_input = UITextEntryLine(
                relative_rect=pygame.Rect((x + label_width, y), (text_input_width, 30)),
                manager=manager,
                container=self.scroll_container
            )
            self.critic_init_log_std_input.set_text("0.3")
            y += row_height

            UILabel(
                relative_rect=pygame.Rect((x, y), (label_width, 25)),
                text="Critic learning Rate:",
                manager=manager,
                container=self.scroll_container,
                object_id='#info_label'
            )
            self.critic_learning_rate_input = UITextEntryLine(
                relative_rect=pygame.Rect((x + label_width, y), (text_input_width, 30)),
                manager=manager,
                container=self.scroll_container
            )
            self.critic_learning_rate_input.set_text("0.003")
            y += row_height

            UILabel(
                relative_rect=pygame.Rect((x, y), (label_width, 25)),
                text="Critic adam Beta1:",
                manager=manager,
                container=self.scroll_container,
                object_id='#info_label'
            )
            self.critic_adam_beta1_input = UITextEntryLine(
                relative_rect=pygame.Rect((x + label_width, y), (text_input_width, 30)),
                manager=manager,
                container=self.scroll_container
            )
            self.critic_adam_beta1_input.set_text("0.9")
            y += row_height

            UILabel(
                relative_rect=pygame.Rect((x, y), (label_width, 25)),
                text="Critic adam Beta2:",
                manager=manager,
                container=self.scroll_container,
                object_id='#info_label'
            )
            self.critic_adam_beta2_input = UITextEntryLine(
                relative_rect=pygame.Rect((x + label_width, y), (text_input_width, 30)),
                manager=manager,
                container=self.scroll_container
            )
            self.critic_adam_beta2_input.set_text("0.999")
            y += row_height

            UILabel(
                relative_rect=pygame.Rect((x, y), (label_width, 25)),
                text="Critic Layers:",
                manager=manager,
                container=self.scroll_container,
                object_id='#info_label'
            )
            self.critic_layers_input = UITextEntryLine(
                relative_rect=pygame.Rect((x + label_width, y), (text_input_width, 30)),
                manager=manager,
                container=self.scroll_container
            )
            self.critic_layers_input.set_text("16,16")
            y += row_height
            UILabel(
                relative_rect=pygame.Rect((x, y), (label_width, 25)),
                text="Critic Activation:",
                manager=manager,
                container=self.scroll_container,
                object_id='#info_label'
            )
            self.critic_activation_dropdown = UIDropDownMenu(
                options_list=[
                    'tanh',
                    'relu',
                    'elu',
                    'linear',
                    'Custom'
                ],
                starting_option='tanh',
                relative_rect=pygame.Rect((x + label_width, y), (text_input_width, 30)),
                manager=manager,
                container=self.scroll_container
            )
            y += row_height

            self. critic_activation_custom_label = UILabel(
                relative_rect=pygame.Rect((x, y), (label_width + 30, 25)),
                text="Custom Critic Activations:",
                manager=manager,
                container=self.scroll_container,
                object_id='#info_label'
            )
            self.critic_activation_custom_input = UITextEntryLine(
                relative_rect=pygame.Rect((x + label_width + 20, y), (text_input_width, 30)),
                manager=manager,
                container=self.scroll_container
            )
            self.critic_activation_custom_input.set_text("tanh,tanh")
            self.critic_activation_custom_label.hide()
            self.critic_activation_custom_input.hide()
            y += row_height

        self.create_button = UIButton(
            relative_rect=pygame.Rect((panel_w / 4 - 75, panel_h - 60), (150, 50)),
            text="Create Model",
            manager=manager,
            container=self.panel,
            object_id='#confirm_button'
        )

        self.cancel_button = UIButton(
            relative_rect=pygame.Rect((panel_w * 3 / 4 - 75, panel_h - 60), (150, 50)),
            text="Cancel",
            manager=manager,
            container=self.panel,
            object_id='#cancel_button'
        )

        self.error_label = UILabel(
            relative_rect=pygame.Rect((50, panel_h - 100), (panel_w - 100, 30)),
            text="",
            manager=manager,
            container=self.panel,
            object_id='#error_label'
        )

        self.scroll_container.set_scrollable_area_dimensions(
            (panel_w - 20, y + 20)
        )

    def get_config_from_inputs(self):
        """
        Extract and validate configuration from all input fields. Any invalid configuration displays an appropriate
        error message to the user.

        Returns:
            dict: Validated model configurations, or None if validation fails
        """
        try:
            config = {}
            # Parse hyperparameters
            config['num_of_cars'] = int(self.num_of_cars_input.get_text())
            config['gamma'] = float(self.gamma_input.get_text())
            config['l2_lambda'] = float(self.l2_lambda_input.get_text())
            config['entropy'] = float(self.entropy_input.get_text())
            config['actor_init_log_std'] = float(self.actor_init_log_std_input.get_text())
            config['actor_learning_rate'] = float(self.actor_learning_rate_input.get_text())
            config['actor_adam_beta1'] = float(self.actor_adam_beta1_input.get_text())
            config['actor_adam_beta2'] = float(self.actor_adam_beta2_input.get_text())

            # Extract actor layer settings
            actor_layer_settings = self.actor_layers_input.get_text().strip().split(',')
            config['actor_hidden_layers'] = []
            for layer in actor_layer_settings:
                layer = layer.strip()
                if layer:
                    config['actor_hidden_layers'].append(int(layer))

            # Extract actor activation function settings
            actor_activation_choice = self.actor_activation_dropdown.selected_option[0]
            if actor_activation_choice == 'Custom':
                actor_activation_text = self.actor_activation_custom_input.get_text()
                config['actor_activations'] = []
                for activation in actor_activation_text.split(','):
                    activation = activation.strip()
                    if activation:
                        config['actor_activations'].append(activation)
                if len(config['actor_activations']) != len(config['actor_hidden_layers']):
                    raise ValueError("Actor activations must match hidden layers")
                config['actor_activations'].append('linear')
            else:
                num_actor_layers = len(config['actor_hidden_layers'])
                config['actor_activations'] = [actor_activation_choice] * num_actor_layers + ['linear']

            # Validate ranges of numerical hyperparameters. Return appropriate errors for invalid ranges.
            if config['num_of_cars'] <= 0 or config['num_of_cars'] > 20 :
                raise ValueError("Number of cars must be a number between 1 and 20")

            if not 0 <= config['gamma'] <= 1:
                raise ValueError("Gamma must be between 0 and 1")

            if not 0 <= config['entropy'] < 1:
                raise ValueError("Entropy must be between 0 and 1")

            if not 0 <= config['l2_lambda'] < 1:
                raise ValueError("L2 Regularisation Lambda must be between 0 and 1")

            if not -5 <= config['actor_init_log_std'] < 5:
                raise ValueError("Actor init log std must be between -5 and 5")

            if not 0 <= config['actor_learning_rate'] < 1:
                raise ValueError("Actor learning rate must be between 0 and 1")

            if not 0 <= config['actor_adam_beta1'] < 1:
                raise ValueError("Actor Adam Beta1 must be between 0 and 1")

            if not 0 <= config['actor_adam_beta2'] < 1:
                raise ValueError("Actor Adam Beta2 must be between 0 and 1")

            # Validate layer sizes.
            if not config['actor_hidden_layers'] or any(x <= 0 for x in config['actor_hidden_layers']):
                raise ValueError("Actor layers must contain positive integers")

            # MCAC specific hyperparameters:
            if self.model_type == "Monte Carlo Actor Critic (MCAC)":
                # Parse numeric parameters
                config['critic_init_log_std'] = float(self.critic_init_log_std_input.get_text())
                config['critic_learning_rate'] = float(self.critic_learning_rate_input.get_text())
                config['critic_adam_beta1'] = float(self.critic_adam_beta1_input.get_text())
                config['critic_adam_beta2'] = float(self.critic_adam_beta2_input.get_text())
                critic_layer_settings = self.critic_layers_input.get_text().strip().split(',')
                config['critic_hidden_layers'] = []
                for layer in critic_layer_settings:
                    layer = layer.strip()
                    if layer:
                        config['critic_hidden_layers'].append(int(layer))

                # Parse activation settings
                critic_activation_choice = self.critic_activation_dropdown.selected_option[0]
                if critic_activation_choice == 'Custom':
                    critic_activation_text = self.critic_activation_custom_input.get_text()
                    config['critic_activations'] = []
                    for activation in critic_activation_text.split(','):
                        activation = activation.strip()
                        if activation:
                            config['critic_activations'].append(activation)

                    if len(config['critic_activations']) != len(config['critic_hidden_layers']):
                        raise ValueError("Critic activations must match hidden layers")
                    config['critic_activations'].append('linear')
                else:
                    num_critic_layers = len(config['critic_hidden_layers'])
                    config['critic_activations'] = [critic_activation_choice] * num_critic_layers + ['linear']

                # Validate critic numeric parameter ranges and display appropriate error message
                if not -5 <= config['critic_init_log_std'] < 5:
                    raise ValueError("Critic init log std must be between -5 and 5")

                if not 0 <= config['critic_learning_rate'] < 1:
                    raise ValueError("Critic learning rate must be between 0 and 1")

                if not 0 <= config['critic_adam_beta1'] < 1:
                    raise ValueError("Critic Adam Beta1 must be between 0 and 1")

                if not 0 <= config['critic_adam_beta2'] < 1:
                    raise ValueError("Critic Adam Beta2 must be between 0 and 1")

                # Validate critic hidden layer sizes and activation functions.
                if not config['critic_hidden_layers'] or any(x <= 0 for x in config['critic_hidden_layers']):
                    raise ValueError("Critic layers must contain positive integers")


            return config

        except ValueError as e:
            # Catch and display error
            self.error_label.set_text(f"Error: {e}")
            return None

    def process_event(self, event):
        super().process_event(event)

        if event.type == pygame_gui.UI_TEXT_ENTRY_CHANGED:
            self.error_label.set_text("")

        if event.type == pygame_gui.UI_DROP_DOWN_MENU_CHANGED:
            self.error_label.set_text("")
            if event.ui_element == self.actor_activation_dropdown:
                if event.text == 'Custom':
                    self.actor_activation_custom_label.show()
                    self.actor_activation_custom_input.show()
                else:
                    self.actor_activation_custom_label.hide()
                    self.actor_activation_custom_input.hide()

            if self.model_type == "Monte Carlo Actor Critic (MCAC)":
                if event.ui_element == self.critic_activation_dropdown:
                    if event.text == 'Custom':
                        self.critic_activation_custom_label.show()
                        self.critic_activation_custom_input.show()
                    else:
                        self.critic_activation_custom_label.hide()
                        self.critic_activation_custom_input.hide()

        if event.type == pygame_gui.UI_BUTTON_PRESSED:
            if event.ui_element == self.create_button:
                config = self.get_config_from_inputs()
                if config:
                    self.kill()
                    self.callback(config)

            elif event.ui_element == self.cancel_button:
                self.kill()
                if self.return_mode:
                    r.game_core.set_game_mode(self.return_mode)

    def kill(self):

        r.game_core.is_paused = False

        self.panel.kill()
        self.title_label.kill()
        self.scroll_container.kill()
        self.num_of_cars_input.kill()
        self.gamma_input.kill()
        self.entropy_input.kill()
        self.actor_learning_rate_input.kill()
        self.actor_adam_beta1_input.kill()
        self.actor_adam_beta2_input.kill()
        self.actor_init_log_std_input.kill()
        self.l2_lambda_input.kill()
        self.actor_layers_input.kill()
        self.actor_activation_dropdown.kill()
        self.actor_activation_custom_label.kill()
        self.actor_activation_custom_input.kill()

        # MCAC-specific inputs
        if self.model_type == "Monte Carlo Actor Critic (MCAC)":
            self.critic_init_log_std_input.kill()
            self.critic_layers_input.kill()
            self.critic_activation_dropdown.kill()
            self.critic_activation_custom_input.kill()
            self.critic_activation_custom_label.kill()
            self.critic_learning_rate_input.kill()
            self.critic_adam_beta1_input.kill()
            self.critic_adam_beta2_input.kill()

        self.create_button.kill()
        self.cancel_button.kill()
        self.error_label.kill()

        super().kill()

class UIEndScreen(UIElement):
    """
    End screen displayed at end of a race.

    Displays individual lap times of Player and AI (if AI is racing).
    User can return to menu, or race again.

    Attributes:
        panel (UIPanel): Results panel
        content_box (UITextBox): Lap time and race time details
        restart_button (UIButton): Restart race
        menu_button (UIButton): Return to menu
    """


    def __init__(self, relative_rect, manager, callback, title, subtitle, content, return_mode=None):
        """
        Initialise end screen dialogue

        Args:
            relative_rect (Rect): End screen area
            manager (UIManager): GUI manager
            callback (callable): Function to call when confirmed by user (Reinitialise race)
            title, subtitle (str): Titles to be displayed
            return_mode (GameMode): Mode to return to on cancel
        """
        super().__init__(relative_rect,manager,container=None,starting_height=999,layer_thickness=1)

        r.game_core.is_paused = True

        self.callback = callback
        self.return_mode = return_mode


        self.image = pygame.Surface(relative_rect.size, pygame.SRCALPHA)
        self.image.fill((30, 30, 30, 140))

        panel_w, panel_h = 500, 320
        self.panel = UIPanel(
            relative_rect=pygame.Rect(
                ((relative_rect.width - panel_w) / 2,
                 (relative_rect.height - panel_h) / 2),
                (panel_w, panel_h)
            ),
            manager=manager,
            starting_height=1000,
            object_id="#end_screen_panel"
        )

        self.title_label = UILabel(
            relative_rect=pygame.Rect((0, 10), (panel_w, 40)),
            text=title,
            manager=manager,
            container=self.panel,
            object_id="#end_screen_title"
        )

        self.subtitle_label = UILabel(
            relative_rect=pygame.Rect((0, 35), (panel_w, 30)),
            text=subtitle,
            manager=manager,
            container=self.panel,
            object_id="#end_screen_subtitle"
        )
        self.content_box = UITextBox(
            html_text=content.replace("\n", "<br>"),
            relative_rect=pygame.Rect((20, 60), (panel_w - 40, 150)),
            manager=manager,
            container=self.panel,
            object_id="#end_screen_content"
        )

        self.restart_button = UIButton(
            relative_rect=pygame.Rect((40, 230), (180, 55)),
            text="Race Again",
            manager=manager,
            container=self.panel,
            object_id="#confirm_button"
        )

        self.menu_button = UIButton(
            relative_rect=pygame.Rect((280, 230), (180, 55)),
            text="Back to Menu",
            manager=manager,
            container=self.panel,
            object_id="#cancel_button"
        )

    def process_event(self, event):
        super().process_event(event)

        if event.type == pygame_gui.UI_BUTTON_PRESSED:
            if event.ui_element == self.restart_button:
                self.kill()
                self.callback("restart")

            elif event.ui_element == self.menu_button:
                self.kill()
                self.callback("menu")

    def kill(self):
        r.game_core.is_paused = False

        self.panel.kill()
        self.title_label.kill()
        self.subtitle_label.kill()
        self.content_box.kill()
        self.restart_button.kill()
        self.menu_button.kill()

        super().kill()


class UIOptionSelector(UIElement):
    """
    Dropdown option selection dialogue.

    User can select one option from the list provided

    Attributes:
        options (list[str]): Allowed options
        callback (callable): Function to call to initialise environment parameter with selected option
        selected_option (str): Currently selected option
        panel (UIPanel): Main container panel
        option_list (UIDropDownMenu): Option dropdown
        ok_button (UIButton): Confirm selection
        cancel_button (UIButton): Cancel dialogue
    """
    def __init__(self, relative_rect, manager, options, callback, title="Select an option", return_mode=None):
        """
        Initialise option selector.

        Args:
            relative_rect (Rect): Option selector screen area
            manager (UIManager): GUI manager
            options (list[str]): Options for the user to choose from
            callback (callable): Function to call with the selected option (initialise required parameter)
            return_mode (GameMode): Mode to return to on cancel
        """
        super().__init__(relative_rect, manager, container=None, starting_height=999, layer_thickness=1)

        r.game_core.is_paused = True

        self.options = options
        self.callback = callback
        self.return_mode = return_mode
        self.selected_option = options[0] if options else None

        self.image = pygame.Surface(relative_rect.size, pygame.SRCALPHA)
        self.image.fill((30, 30, 30, 120))

        panel_w, panel_h = 400, 250
        self.panel = UIPanel(
            relative_rect=pygame.Rect(
                ((relative_rect.width - panel_w) / 2,
                 (relative_rect.height - panel_h) / 2),
                (panel_w, panel_h)
            ),
            manager=manager,
            starting_height=1000,
            object_id="#selector_panel"
        )

        self.title_label = UILabel(
            relative_rect=pygame.Rect((0, 10), (panel_w, 30)),
            text=title,
            manager=manager,
            container=self.panel,
            object_id="#selector_title"
        )

        self.option_list = UIDropDownMenu(
            options_list=options,
            starting_option=self.selected_option,
            relative_rect=pygame.Rect((50, 70), (300, 40)),
            manager=manager,
            container=self.panel,
            object_id="#selector_list"
        )

        self.ok_button = UIButton(
            relative_rect=pygame.Rect((40, 150), (140, 50)),
            text="OK",
            manager=manager,
            container=self.panel,
            object_id="#confirm_button"
        )

        self.cancel_button = UIButton(
            relative_rect=pygame.Rect((220, 150), (140, 50)),
            text="Cancel",
            manager=manager,
            container=self.panel,
            object_id="#cancel_button"
        )

    def process_event(self, event):
        super().process_event(event)

        if event.type == pygame_gui.UI_DROP_DOWN_MENU_CHANGED:
            if event.ui_element == self.option_list:
                self.selected_option = event.text

        if event.type == pygame_gui.UI_BUTTON_PRESSED:
            if event.ui_element == self.ok_button:
                self.kill()
                self.callback(self.selected_option)

            elif event.ui_element == self.cancel_button:
                self.kill()
                if self.return_mode:
                    r.game_core.set_game_mode(self.return_mode)

    def kill(self):
        r.game_core.is_paused = False

        self.panel.kill()
        self.title_label.kill()
        self.option_list.kill()
        self.ok_button.kill()
        self.cancel_button.kill()

        super().kill()

class UIFileBrowser(UIElement):
    """
    File browser with save/load modes.

    Lists relevant files from directory, validates filenames, and handles overwrite confirmation.

    Attributes:
        directory (str): Directory to browse ("Tracks" or "Models")
        mode (str): Operation ("save" or "load")
        confirm_overwrite (bool): Overwrite confirmation
        panel (UIPanel): Main container panel
        file_list (UISelectionList): List of relevant files
        text_entry (UITextEntryLine): Text input field to enter file name
        ok_button (UIButton): Confirm selection
        cancel_button (UIButton): Cancel browser
    """

    def __init__(self, relative_rect, manager, directory, mode,
                 callback= lambda directory, filename: print(directory, filename), return_mode = None):

        """
        Initialise file browser

        Args:
            relative_rect (Rect): File browser area
            manager (UIManager): GUI manager
            directory (str): The directory being accessed ("Track" or "Model"),
            mode (str): Browser mode ("save", "load REINFORCE", "load Monte Carlo Actor Critic (MCAC)" or
                                            "load raceable")
            callback (callable): Function to call with selected file (save to directory, or load to environment)
            return_mode (GameMode): Mode to return to on cancel
        """
        super().__init__(relative_rect, manager, container=None,
                         starting_height=999, layer_thickness=1)

        r.game_core.is_paused = True
        self.directory = directory
        self.mode = mode
        self.callback = callback
        self.return_mode = return_mode
        self.confirm_overwrite = False


        self.image = pygame.Surface(relative_rect.size, pygame.SRCALPHA)
        self.image.fill((30, 30, 30, 100))

        panel_w = 500
        panel_h = 450
        self.panel = UIPanel(
            relative_rect=pygame.Rect(
                ((relative_rect.width - panel_w) / 2, (relative_rect.height-panel_h) / 2 ),
                (panel_w, panel_h)
            ),
            manager=manager,
            starting_height=1000,
            object_id='#selector_panel'
        )

        self.title_label = UILabel(
            relative_rect=pygame.Rect((panel_w/2 - 50,  10), (100, 20)),
            text= f"Select {self.directory}",
            manager=manager,
            container=self.panel,
            object_id='#selector_title'
        )

        self.file_list = UISelectionList(
            relative_rect=pygame.Rect((panel_w/2 - 200, 50), (400, 250)),
            item_list=self.get_file_list(),
            manager=manager,
            container=self.panel,
            object_id='#selector_list'
        )

        self.entry_label = UILabel(
            relative_rect=pygame.Rect((panel_w/4 - 45, 315), (150, 20)),
            text=f"Enter {self.directory} name",
            manager=manager,
            container=self.panel,
            object_id='#info_label'
        )

        self.text_entry = UITextEntryLine(
            relative_rect=pygame.Rect((panel_w/1.75 - 60, 310), (200, 30)),
            manager=manager,
            container=self.panel,
            object_id= '#text_box'
        )

        self.error_label = UILabel(
            relative_rect=pygame.Rect((panel_w/2 - 200, 350), (400, 30)),
            text="",
            manager=manager,
            container=self.panel,
            object_id='#error_label'
        )

        self.ok_button = UIButton(
            relative_rect=pygame.Rect((panel_w/4 - 75, 380), (150, 50)),
            text= "OK",
            manager=manager,
            container=self.panel,
            object_id='#confirm_button'
        )

        self.cancel_button = UIButton(
            relative_rect=pygame.Rect((panel_w*3/4 - 75, 380),
                                      (150, 50)),
            text="Train New Model" if not self.return_mode and self.mode.__contains__(" ") else "Cancel",
            manager=manager,
            container=self.panel,
            object_id='#cancel_button'
        )


    def get_file_list(self):
        """
        Get list of valid files based on mode
        for loading model: list only selected model type
        for track: list only raceable tracks if mode is "load raceable".

        Returns:
            list: List of valid file names
        """
        extension = '.json' if self.directory == 'Tracks' else '.npy'
        files = [f[:-len(extension)] for f in os.listdir(self.directory)]
        cleaned_files = []
        if self.mode == 'load REINFORCE' or self.mode == 'load Monte Carlo Actor Critic (MCAC)':
            for file in files:
                try:
                    save_data = np.load(f"Models/{file}.npy", allow_pickle=True).item()
                    if ((self.mode == 'load REINFORCE' and save_data["model_type"] == "REINFORCE") or
                            (self.mode == 'load Monte Carlo Actor Critic (MCAC)' and save_data["model_type"] == "MCAC")):
                        cleaned_files.append(file)
                except:
                    print(f"Corrupted file {file}")

        elif self.mode == "load raceable":
            for file in files:
                try:
                    save_data = np.load(f"Models/{file}.npy", allow_pickle=True).item()
                    if save_data['is_model_raceable']:
                        cleaned_files.append(file)
                except:
                    print(f"Corrupted file {file}")

        else:
            cleaned_files = files
        return cleaned_files if cleaned_files else [f'No {self.directory}']

    def validated_save_filename(self, filename):
        """
        Use regex to validate save filename. If file exists, confirm overwrite.

        Args:
            filename (str): Filename to validate

        Returns:
            str: Validated filename
        """
        filename = filename.strip()
        extension = '.json' if self.directory == 'Tracks' else '.npy'
        filepath = f"{self.directory}/{filename}{extension}"

        if not filename:
            self.error_label.set_text("Filename cannot be empty")
            return None

        if ".." in filename or "/" in filename or "\\" in filename:
            self.error_label.set_text("Invalid characters in filename")
            return None

        if not re.fullmatch(r"[A-Za-z0-9 _\-]+", filename):
            self.error_label.set_text("Only letters, numbers, spaces, _ and - allowed")
            return None

        if os.path.exists(filepath):
            if not self.confirm_overwrite:
                self.error_label.set_text("File exists. Press OK again to overwrite.")
                self.confirm_overwrite = True
                return None

        return filename


    def get_load_file_path(self, filename):
        """
        Get filepath to load model/track.
        Args:
            filename (str): Filename to load
        Returns:
            str: filepath to load model
        """
        extension = '.json' if self.directory == 'Tracks' else '.npy'
        filepath = f"{self.directory}/{filename}{extension}"
        valid_files = []
        for item in self.file_list.item_list:
            valid_files.append(item['text'])

        if not os.path.exists(filepath):
            self.error_label.set_text("File not found")
            return
        if  filename not in valid_files:
            self.error_label.set_text("Could not load file (File is corrupted)")
            return
        return filepath

    def process_event(self, event):
        super().process_event(event)

        if event.type == pygame_gui.UI_SELECTION_LIST_NEW_SELECTION:
            if event.ui_element == self.file_list:
                selected = event.text
                if selected != 'No files found':
                    self.text_entry.set_text(selected)
                    self.error_label.set_text("")

        if event.type == pygame_gui.UI_TEXT_ENTRY_CHANGED:
            self.confirm_overwrite = False
            self.error_label.set_text("")

        if event.type == pygame_gui.UI_BUTTON_PRESSED:
            if event.ui_element == self.ok_button:
                filename = self.text_entry.get_text().strip()

                if self.mode.__contains__("save"):
                    validated = self.validated_save_filename(filename)
                    if validated:
                        if self.directory == "Tracks":
                            r.game_core.current_track = filename
                        else:
                            r.game_core.current_model = filename
                        self.kill()
                        self.callback(validated)

                elif self.mode.__contains__("load"):
                    filepath = self.get_load_file_path(filename)
                    if filepath:
                        if self.directory == "Tracks":
                            r.game_core.current_track = filename
                        else:
                            r.game_core.current_model = filename
                        self.kill()
                        self.callback(filename)


            elif event.ui_element == self.cancel_button:
                self.kill()
                if self.return_mode:
                    r.game_core.set_game_mode(self.return_mode)
                else:
                    self.callback(filename=None)

    def kill(self):
        r.game_core.is_paused = False
        self.panel.kill()
        self.title_label.kill()
        self.file_list.kill()
        self.entry_label.kill()
        self.text_entry.kill()
        self.error_label.kill()
        self.ok_button.kill()
        self.cancel_button.kill()
        super().kill()

class UIGaugeMeter(UIElement):
    """
    Speedometer display.

    Displays numeric values as rotating needle on semicircular gauge.

    Attributes:
        min_value, max_value, value (float): The minimum, maximum and current value of the gauge
        fill_colour (str): Background colour
        dial_colour (str): Dial colour
        border_colour (str): Border colour
        border_width (int): Border thickness
        dial_thickness (int): DIal thickness
    """
    def __init__(self, relative_rect, manager,
                 min_value=0, max_value=300, starting_value=0,
                 fill_colour="black", dial_colour='red',
                 border_colour="grey", border_width=3, dial_thickness=2, container = None):
        """
        Initialise gauge meter.

        Args:
            relative_rect (Rect): Gauge area
            manager (UIManager): GUI manager
            min_value, max_value (float): Min and max values
            starting_value (float): Initial value
            [styling attributes] (str/int): Parameters for styling the meter.
            container (UIElement): Parent container
        """
        super().__init__(relative_rect, manager, container = container, starting_height=0, layer_thickness=1)

        self.min_value = min_value
        self.max_value = max_value
        self.value = starting_value
        self.fill_colour = fill_colour
        self.dial_colour = dial_colour
        self.border_colour = border_colour
        self.border_width = border_width
        self.dial_thickness = dial_thickness

        self.image = pygame.Surface((relative_rect.width + border_width,
                                     relative_rect.height + border_width), pygame.SRCALPHA)
        self.rebuild()

    def rebuild(self):
        self.image.fill(self.fill_colour)
        width, height = self.relative_rect.size
        center = (width // 2, height)
        radius = height

        arc_rect = pygame.Rect(0, 0, radius * 2, radius * 2)
        arc_rect.midbottom = center
        pygame.draw.arc(self.image, self.border_colour, arc_rect, math.pi, 2* math.pi , self.border_width)
        self.image = pygame.transform.flip(self.image, False, True)

        angle = self.get_angle()
        end_pos = (
            center[0] + (radius - self.border_width*2) * math.cos(angle),
            center[1] + (radius - self.border_width*2) * math.sin(angle)
        )
        pygame.draw.line(self.image, self.dial_colour, center, end_pos, self.dial_thickness)


    def get_angle(self):
        """
        Get dial angle from relative value

        Returns:
            float: Dial angle
        """
        relative_value = (self.value - self.min_value) / (self.max_value - self.min_value)
        return math.pi + relative_value * math.pi


    def update_value(self, value):
        """
        Clamps value within range

        Args:
            value (float): Value to clamp
        """
        self.value = r.clamp_value(value, self.min_value, self.max_value)
        self.rebuild()

class UITrackCanvas(UIElement):
    """
    Track editor with bezier curve tool.

    Allows creation of unique track shapes using bezier curve with automatic control point generation.
    Can optionally enable curvature control handles.
    Handles validation of track shapes to ensure no overlaps/kinks.

    Attributes:
        shg_grid (SpatialHashGrid): Spatial hash for collision detection
        anchor_points (list[Vector2]): List of bezier anchor points
        control_points (list[Vector2]): List of bezier control points
        widths_dict (dict): Width values at fractional positions
        track_spine (list[Vector2]): Generated centerline
        max_handle_length (int): Maximum curvature handle length
        mode (str): Current mode ("anchor" or "width")
        is_dragging (bool): Whether user is currently dragging a points
        is_track_complete (bool): Flag for if track forms closed loop
        validity_issues (str): Current validity issue preventing saving
        selected_point (tuple): (type, index) of selected point
    """
    def __init__(self, relative_rect, manager, is_handles_enabled):
        """
        Initialise track canvas

        Args:
            relative_rect (Rect): Canvas area
            manager (UIManager): GUI manager
            is_handles_enabled (bool): Initial state of handles
        """
        super().__init__(relative_rect, manager, container=None, starting_height=0, layer_thickness=1)

        self.image = pygame.Surface((relative_rect.width + 10, relative_rect.height + 10))
        self.shg_grid = SpatialHashGrid(20)
        self.border_width = 20
        self.anchor_points = []
        self.control_points = []
        self.widths_dict = {0.0:60}
        self.track_spine = []

        self.point_size = 5
        self.max_handle_length = 100

        self.mode = "anchor"
        self.is_handles_enabled = is_handles_enabled
        self.is_dragging = False
        self.is_track_complete = False
        self.validity_issues = None
        self.is_click_buffer = False
        self.selected_point = None

        self.rebuild([],[])

    def process_event(self, event):
        if (event.type in (pygame.MOUSEBUTTONDOWN, pygame.MOUSEBUTTONUP, pygame.MOUSEMOTION)):
            relative_x, relative_y = event.pos[0] - self.rect.x, event.pos[1] - self.rect.y
            if (self.rect.collidepoint(event.pos) and
                self.border_width <= relative_x <= self.rect.width - self.border_width  and
                self.border_width  <= relative_y <= self.rect.height - self.border_width):
                relative_mouse_pos = Vector2(event.pos[0] - self.rect.x, event.pos[1] - self.rect.y)

                if event.type == pygame.MOUSEBUTTONDOWN and event.button == 1:
                    self.process_mouse_down(relative_mouse_pos)

                if event.type == pygame.MOUSEMOTION:
                    self.process_mouse_motion(relative_mouse_pos)

                if event.type == pygame.MOUSEBUTTONUP and event.button == 1:
                    self.process_mouse_up()

        elif event.type == pygame.KEYDOWN and event.key == pygame.K_BACKSPACE:
            if not self.selected_point or self.selected_point[0] == 'c' or len(self.anchor_points) <= 2:
                return
            self.handle_point_delete()

        outer_wall_points, inner_wall_points = self.generate_track()
        self.rebuild(outer_wall_points, inner_wall_points)

    def process_mouse_down(self, relative_mouse_pos):
        """
        Handle mouse left clicks:
        if point selected, start dragging
        else insert/ create anchor/ width point (depending on click location and mode)

        Args:
            relative_mouse_pos (Vector2): Relative mouse position
        """
        self.is_click_buffer = True
        is_point_selected = self.point_selection_handling(relative_mouse_pos)

        if is_point_selected:
            if self.selected_point:
                self.is_dragging = True
        else:
            if self.mode == "anchor":
                insertion_index = self.handle_anchor_insertion(relative_mouse_pos)
                if insertion_index is not None:
                    insertion_index += 1
                    self.insert_anchor(relative_mouse_pos, insertion_index)
                else:
                    self.create_new_anchor(relative_mouse_pos)
                self.check_track_completion()
            if self.mode == "width":
                self.create_width_point(relative_mouse_pos)

    def process_mouse_motion(self, relative_mouse_pos):
        """
        Handle mouse movement (drag points if a point is selected)

        Args:
            relative_mouse_pos (Vector2): Relative mouse position
        """
        if self.is_dragging:
            point_index = self.selected_point[1]
            if self.selected_point[0] == 'a':
                self.handle_anchor_point_movement(relative_mouse_pos, point_index)
                self.check_track_completion()
            elif self.selected_point[0] == 'c':
                self.handle_control_point_movement(relative_mouse_pos, point_index)

    def process_mouse_up(self):
        """
        Reset flags when mouse button is released
        """
        self.is_click_buffer = False
        self.is_dragging = False

    def point_selection_handling(self, mouse_pos):
        """
        Select anchor/width point if clicked

        Args:
            mouse_pos (Vector2): Relative mouse position
        """
        if self.mode == "anchor":
            for i, point in enumerate(self.anchor_points):
                if (mouse_pos - point).length() <= self.point_size:
                    self.selected_point = ('a', i)
                    self.is_point_selected = True
                    return True

            for i, point in enumerate(self.control_points):
                if (mouse_pos - point).length() <= self.point_size:
                    self.selected_point = ('c', i)
                    self.is_point_selected = True
                    return True

        elif self.mode == "width":
            if not self.track_spine:
                return False
            width_point_positions = list(self.widths_dict.keys())
            for position in width_point_positions:
                width_point_index = max(min(math.floor(position * len(self.track_spine)), len(self.track_spine)-1),0)
                point = self.track_spine[width_point_index]
                if (mouse_pos - point).length() <= self.point_size:
                    self.selected_point = ('w', position)
                    self.is_point_selected = True
                    return True

        if self.selected_point:
            self.selected_point = None
            return True
        return False

    def handle_point_delete(self):
        """
        Delete selected point
        If anchor point deleted, delete corresponding control points
        """
        i = self.selected_point[1]
        if self.selected_point[0] == 'a':

            self.anchor_points.pop(i)
            self.selected_point = None

            if not self.is_handles_enabled:
                return
            if 0 < i < len(self.anchor_points) - 1:
                self.control_points.pop(2 * i)
                self.control_points.pop(2 * i - 1)
            else:
                if i == 0:
                    self.control_points.pop(0)
                    self.control_points.pop(0)
                else:
                    self.control_points.pop(-1)
                    self.control_points.pop(-1)

        else:
            self.widths_dict.pop(i)

        self.selected_point = None

    def handle_anchor_insertion(self, mouse_pos):
        for i in range(len(self.anchor_points) - 1):
            a1, a2 = self.anchor_points[i], self.anchor_points[i + 1]
            if self.is_handles_enabled:
                c1, c2 = self.control_points[2 * i], self.control_points[2 * i + 1]
                spine_points = r.generate_bezier_track_spine([a1, a2], [c1, c2])
            else:
                spine_points = r.generate_catmull_rom_track_spine([a1, a2], self.is_track_complete)

            for j in range(len(spine_points) - 1):
                segment = (spine_points[j], spine_points[j + 1])

                dist = r.point_segment_distance(mouse_pos, segment)
                if dist <= 40:
                    return i
        return None

    def create_new_anchor(self, mouse_pos):
        self.anchor_points.append(mouse_pos)
        if len(self.anchor_points) > 1:
            p0 = self.anchor_points[-3] if len(self.anchor_points) > 2 else self.anchor_points[-2]
            p1 = self.anchor_points[-2]
            p2 = self.anchor_points[-1]
            p3 = mouse_pos
            if self.is_handles_enabled:
                self.control_points.extend(self.get_bezier_points(p0, p1, p2, p3))
                if len(self.anchor_points) > 2:
                    anchor = self.anchor_points[-2]
                    handle_length = (self.control_points[-2] - anchor).length()
                    direction = (self.control_points[-2] - anchor).normalize()

                    updated_handle_point = anchor - handle_length * direction
                    self.control_points[-3] = updated_handle_point

    def insert_anchor(self, mouse_pos, insert_index):
        self.anchor_points.insert(insert_index, mouse_pos)
        self.control_points = []
        if not self.is_handles_enabled:
            return
        for i in range(len(self.anchor_points) - 1):
            p1 = self.anchor_points[i]
            p2 = self.anchor_points[i + 1]
            p0 = self.anchor_points[i - 1] if i - 1 >= 0 else p1
            p3 = self.anchor_points[i + 2] if i + 2 < len(self.anchor_points) else p2
            b1, b2 = self.get_bezier_points(p0, p1, p2, p3)
            self.control_points.append(b1)
            self.control_points.append(b2)

        if self.is_track_complete and len(self.control_points) >= 2:
            handle_point = self.anchor_points[0]
            mirror_handle_translation = self.control_points[0] - handle_point
            self.control_points[-1] = handle_point - mirror_handle_translation

    def create_width_point(self, mouse_pos):
        if len(self.track_spine) > 1:
            for i in range(len(self.track_spine) - 1):
                segment = (self.track_spine[i], self.track_spine[i+1])
                dist = r.point_segment_distance(mouse_pos, segment)
                if dist <= 2:

                    # Create points if the track spine point is close enough
                    track_proportion = i/(len(self.track_spine)-1)
                    self.widths_dict[track_proportion] = 60
                    self.selected_point = ('w', track_proportion)
                    break

    def handle_anchor_point_movement(self, mouse_pos, point_index):
        """
        Move anchor point and associated control points

        Args:
            mouse_pos(Vector2): (x, y) coordinate
            point_index(int): point index in anchors list
        """
        if self.anchor_points[0] != self.anchor_points[-1]:
            self.is_track_complete = False
        translate_vector = mouse_pos - self.anchor_points[point_index]
        self.anchor_points[point_index] = mouse_pos
        if not self.is_handles_enabled:
            return
        if point_index * 2 < len(self.control_points):
            self.control_points[point_index * 2] += translate_vector
        if point_index != 0:
            self.control_points[point_index * 2 - 1] += translate_vector

    def handle_control_point_movement(self, mouse_pos, point_index):
        """
        Move control point and paired handle point

        Args:
            mouse_pos(Vector2): (x, y) coordinate
            point_index(int): point index in controls list
        """
        if not self.is_handles_enabled:
            return
        corresponding_anchor_index = point_index // 2 if point_index % 2 == 0 else point_index // 2 + 1
        anchor_point = self.anchor_points[corresponding_anchor_index]
        clamped_translation = (mouse_pos - anchor_point).clamp_magnitude(self.max_handle_length)
        if clamped_translation.length_squared() < (2*self.point_size) ** 2:
            return
        moved_point = anchor_point + clamped_translation
        self.control_points[point_index] = moved_point
        mirror_point_index = None
        if point_index != 0 and point_index != len(self.control_points) - 1:
            if point_index % 2 == 0:
                mirror_point_index = point_index - 1
            else:
                mirror_point_index = point_index + 1
        if self.is_track_complete:
            if point_index == 0:
                self.control_points[-1] = anchor_point - clamped_translation
            elif point_index == len(self.control_points) - 1:
                self.control_points[0] = anchor_point - clamped_translation

        if mirror_point_index:
            self.control_points[mirror_point_index] = anchor_point - clamped_translation

    def check_track_completion(self):
        """
        Check if first anchor is close to last anchor for track completion
        """
        if len(self.anchor_points) < 3:
            return False

        first_point = self.anchor_points[0]
        last_point = self.anchor_points[-1]
        moving_anchor = self.anchor_points[-1]

        if (moving_anchor != last_point and moving_anchor != first_point
            or (last_point - first_point).length() > self.point_size * 5):
            return False
        self.anchor_points[-1] = first_point
        self.is_track_complete = True
        if not self.is_handles_enabled:
            return
        mirror_handle_translation = self.control_points[0] - first_point
        self.control_points[-1] = first_point - mirror_handle_translation

        return True

    def generate_track(self):
        """
        Generate track spine and track walls from anchors and control points data
        Generate track using de casteljau's algorithm or catmull-rom interpolation depending on if handles is enabled

        Returns:
             tuple(list[Vector2], list[Vector2]): outer_wall_points, inner_wall_points
        """
        if self.track_spine and not self.widths_dict.__contains__(1.0):
            self.widths_dict[1.0] = self.widths_dict[0.0]
        outer_wall_points, inner_wall_points = r.generate_track_walls(self.track_spine, self.widths_dict)
        if self.is_track_complete and outer_wall_points and inner_wall_points:
            outer_wall_points.append(outer_wall_points[0])
            inner_wall_points.append(inner_wall_points[0])
            self.widths_dict.pop(1.0)
        if self.is_handles_enabled:
            self.track_spine = r.generate_bezier_track_spine(self.anchor_points, self.control_points)
        else:
            self.track_spine = r.generate_catmull_rom_track_spine(self.anchor_points, self.is_track_complete)
        return outer_wall_points, inner_wall_points

    def rebuild(self, outer_wall_points, inner_wall_points):
        self.image.fill((152, 152, 152))
        self.draw_track(outer_wall_points, inner_wall_points)
        if len(self.anchor_points) != 0:
            if self.mode == "anchor":
                self.draw_anchors_and_handles()
            else:
                self.draw_width_points()

        validity_overlay = self.generate_validity_overlay(outer_wall_points, inner_wall_points)
        self.image.blit(validity_overlay, (0, 0))

        pygame.draw.rect(self.image, "black", pygame.Rect((0, 0), (
        self.image.get_width() - self.border_width/2, self.image.get_height())), width=self.border_width)

    def draw_track(self, outer_wall_points, inner_wall_points):
        r.draw_alternating_line_segments(self.image, self.track_spine)
        for i in range(len(outer_wall_points)-1):
            r.draw_line(self.image, "purple", outer_wall_points[i], outer_wall_points[i+1])
        for i in range(len(inner_wall_points)-1):
            r.draw_line(self.image, "purple", inner_wall_points[i], inner_wall_points[i+1])

    def draw_anchors_and_handles(self):
        for anchor in self.anchor_points:
            if anchor == self.anchor_points[0]:
                pygame.draw.circle(self.image, color="Red", center=anchor, radius=self.point_size + 1)
            elif anchor == self.anchor_points[-1]:
                pygame.draw.circle(self.image, color="black", center=anchor, radius=self.point_size + 1)
            else:
                pygame.draw.circle(self.image, color="green", center=anchor, radius=self.point_size)

        for i, handle in enumerate(self.control_points):
            if i % 2 == 0 and i != 0:
                pygame.draw.line(self.image, "dark grey", handle, self.control_points[i - 1], 2)
            if i == 0:
                pygame.draw.line(self.image, "dark grey", handle, self.anchor_points[0], 2)
            if i == len(self.control_points) - 1:
                pygame.draw.line(self.image, "dark grey", handle, self.anchor_points[-1], 2)

            pygame.draw.circle(self.image, color="orange", center=handle, radius=self.point_size)

    def draw_width_points(self):
        width_point_positions = list(self.widths_dict.keys())
        for width_point_position in width_point_positions:
            width_point_index = math.floor(
                max(min(width_point_position * len(self.track_spine), len(self.track_spine) - 1), 0))
            pygame.draw.circle(self.image, color="pink", center=self.track_spine[int(width_point_index)],
                               radius=self.point_size)

    def generate_validity_overlay(self, outer_wall_points, inner_wall_points):
        """
        Create an overlay to highlight selected point blue and highlight invalid regions red

        Args:
            outer_wall_points, inner_wall_points (list(Vector2)): wall points
        Returns:
            validity_overlay (pygame.Surface): overlay with highlight
        """
        validity_overlay = pygame.Surface(self.image.get_size(), pygame.SRCALPHA)
        if self.selected_point:
            point_index = self.selected_point[1]
            point_type = self.selected_point[0]
            if point_type == 'a':
                point = self.anchor_points[point_index]
            elif point_type == 'c':
                point = self.control_points[point_index]
            else:
                width_point_index = math.floor(
                    max(min(point_index * len(self.track_spine), len(self.track_spine) - 1), 0))
                point = self.track_spine[int(width_point_index)]
            pygame.draw.circle(validity_overlay, color=(137, 207, 240, 200), center=point, radius=self.point_size + 2)

        invalid_walls = self.get_validity_issues(outer_wall_points, inner_wall_points)
        for wall in invalid_walls:
            r.draw_line(validity_overlay, (255, 0, 0, 150), wall[0], wall[1], 15)
        return validity_overlay

    def hash_grid(self, outer_wall_points, inner_wall_points):
        """
        Rehash walls to grid data structure

        Args:
            outer_wall_points, inner_wall_points (list(Vector2)): wall points
        """
        self.shg_grid.clear_grid()
        hash_points = outer_wall_points + inner_wall_points
        for i in range(len(hash_points) - 1):
            if i != len(outer_wall_points) - 1:
                self.shg_grid.hash_segment((hash_points[i], hash_points[i + 1]), i)

    def generate_all_controls(self):
        """
        Generate all controls from catmull rom anchors using catmull rom to bezier conversion
        """
        self.control_points.clear()
        if len(self.anchor_points) < 2:
            return

        for i in range(len(self.anchor_points)  - 1):
            p1 = self.anchor_points[i]
            p2 = self.anchor_points[i + 1]

            if self.is_track_complete:
                # wrap neighbors for closed track:
                p0 = self.anchor_points[i - 1] if i - 1 >= 0 else self.anchor_points[-2]
                p3 = self.anchor_points[i + 2] if i + 2 < len(self.anchor_points)  else self.anchor_points[1]
            else:
                # keep current behaviour for open track
                p0 = self.anchor_points[i - 1] if i - 1 >= 0 else p1
                p3 = self.anchor_points[i + 2] if i + 2 < len(self.anchor_points)  else p2

            b1, b2 = self.get_bezier_points(p0, p1, p2, p3)
            self.control_points.append(b1)
            self.control_points.append(b2)

        # ensure the handle is mirrored at connection point for closed tracks
        if self.is_track_complete and len(self.control_points) >= 2:
            handle_point = self.anchor_points[0]  # anchor at seam
            mirror_handle_translation = self.control_points[0] - handle_point
            self.control_points[-1] = handle_point - mirror_handle_translation

    @staticmethod
    def get_bezier_points(p0, p1, p2, p3):
        """
        Catmull rom to bezier conversion

        Args:
            p0, p1, p2, p3 (Vector2): Catmull rom anchor points
        Returns:
            b1, b2 (Vector2): Bezier control points
        """
        b1 = p1 + (p2 - p0) / 6
        b2 = p2 - (p3 - p1) / 6
        return b1, b2

    def get_validity_issues(self, outer_wall_points, inner_wall_points):
        """
        Set appropriate error type by checking reason for invalidity

        Args:
            outer_wall_points, inner_wall_points (list(Vector2)): wall points
        """
        self.hash_grid(outer_wall_points, inner_wall_points)

        def check_wall_validity(wall_points, hashed_grid_start_index):
            hashed_points = outer_wall_points + inner_wall_points
            invalid_segments = []
            for i in range(len(wall_points) - 1):
                segment = (wall_points[i], wall_points[i + 1])
                collisions = self.shg_grid.return_all_collisions(segment, hashed_points)
                hashed_segment_index = hashed_grid_start_index + i

                for collision_index in collisions:
                    distance = min(abs(collision_index - hashed_segment_index),
                                   len(wall_points) - 1 - abs(collision_index - hashed_segment_index))

                    # Checking for collision between walls that are not adjacent/very close (preventing false positives)
                    if ((0< collision_index < len(hashed_points)-1 and 0< i < len(hashed_points)-1)and
                            (distance > 4)):
                        invalid_segments.append((hashed_points[collision_index], hashed_points[collision_index + 1]))

                # Checking for kinks in the track walls
                if Vector2(segment[0] - segment[1]).length_squared() > 4000:
                    invalid_segments.append(segment)

            return invalid_segments

        invalid_outer_walls = check_wall_validity(outer_wall_points, 0)
        invalid_inner_walls = check_wall_validity(inner_wall_points, len(outer_wall_points))

        if len(invalid_inner_walls + invalid_outer_walls) > 0:
            self.validity_issues = "invalid walls"
        elif not self.is_track_complete:
            self.validity_issues = "incomplete"
        elif len(outer_wall_points) < 0 or len(inner_wall_points) < 0:
            self.validity_issues = "too short"
        else:
            self.validity_issues = None
        return invalid_outer_walls + invalid_inner_walls

    def kill(self):
        self.anchor_points.clear()
        self.control_points.clear()
        self.track_spine.clear()
        self.widths_dict.clear()
        self.shg_grid.clear_grid()
        self.selected_point = None
        self.is_dragging = False
        self.is_track_complete = False
        self.validity_issues = None

        super().kill()
