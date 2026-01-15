import math
from operator import index
from typing import Union
import random
import json
from fontTools.varLib.errors import NotANone
import os
import resources as r
import pygame

from track import SpatialHashGrid
from pygame.math import Vector2
from pygame_gui.core import UIElement
from pygame_gui.elements import UIPanel,UILabel,UIButton,UISelectionList,UITextEntryLine, UIDropDownMenu
import pygame_gui
import re
"""
This module contains custom GUI elements not included in the pygame_gui library
"""

class UIOptionSelector(UIElement):
    def __init__(self, relative_rect, manager, options, callback, title="Select an option", return_mode=None):
        super().__init__(relative_rect, manager, container=None, starting_height=999, layer_thickness=1)

        # Pause game
        r.game_core.is_paused = True

        self.options = options
        self.callback = callback
        self.return_mode = return_mode
        self.selected_option = options[0] if options else None

        # Semi-transparent background
        self.image = pygame.Surface(relative_rect.size, pygame.SRCALPHA)
        self.image.fill((30, 30, 30, 120))

        # Panel
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

        # Title label
        self.title_label = UILabel(
            relative_rect=pygame.Rect((0, 10), (panel_w, 30)),
            text=title,
            manager=manager,
            container=self.panel,
            object_id="#selector_title"
        )

        # Dropdown list (similar to file_list in FileSelector)
        self.option_list = UIDropDownMenu(
            options_list=options,
            starting_option=self.selected_option,
            relative_rect=pygame.Rect((50, 70), (300, 40)),
            manager=manager,
            container=self.panel,
            object_id="#selector_list"
        )

        # OK button
        self.ok_button = UIButton(
            relative_rect=pygame.Rect((40, 150), (140, 50)),
            text="OK",
            manager=manager,
            container=self.panel,
            object_id="#confirm_button"
        )

        # Cancel button
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

class UIFileSelector(UIElement):
    def __init__(self, relative_rect, manager, directory, mode,
                 callback= lambda directory, filename: print(directory, filename), return_mode = None):
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
            relative_rect=pygame.Rect((panel_w/4 - 45, 315), (130, 20)),
            text=f"Enter {self.directory} name",
            manager=manager,
            container=self.panel,
            object_id='#info_label'
        )

        self.text_entry = UITextEntryLine(
            relative_rect=pygame.Rect((panel_w/1.75 - 70, 310), (200, 30)),
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
            text="Train New Model" if not self.return_mode else "Cancel",
            manager=manager,
            container=self.panel,
            object_id='#cancel_button'
        )

    def get_file_list(self):
        extension = '.json' if self.directory == 'Track' else '.npy'
        files = [f[:-len(extension)] for f in os.listdir(self.directory)]

        return files if files else [f'No saved {self.directory}s']

    def validated_save_filename(self, filename):
        filename = filename.strip()
        extension = '.json' if self.directory == 'Track' else '.npy'
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
        extension = '.json' if self.directory == 'Track' else '.npy'
        filepath = f"{self.directory}/{filename}{extension}"
        if not os.path.exists(filepath):
            self.error_label.set_text("File not found")
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

                if self.mode == "save":
                    validated = self.validated_save_filename(filename)
                    if validated:
                        if self.directory == "Track":
                            r.game_core.current_track = filename
                        else:
                            r.game_core.current_model = filename
                        self.kill()
                        self.callback(validated)


                elif self.mode == "load":
                    filepath = self.get_load_file_path(filename)
                    if filepath:
                        if self.directory == "Track":
                            r.game_core.current_track = filename
                        else:
                            r.game_core.current_model = filename
                        self.kill()
                        self.callback(filename)


            elif event.ui_element == self.cancel_button:
                self.kill()
                if self.return_mode:
                    r.game_core.set_game_mode(self.return_mode)
                

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
    This class is for a meter that displays values as a gauge needle rotating (like a speedometer/rpm meter)
    """
    def __init__(self, relative_rect, manager,
                 min_value=0, max_value=300, starting_value=0,
                 fill_colour="grey", dial_colour='red',
                 border_colour="black", border_width=3, dial_thickness=2):
        super().__init__(relative_rect, manager, container=None, starting_height=0, layer_thickness=1)

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

    # Redraws updated version of meter
    def rebuild(self):
        self.image.fill("grey")
        width, height = self.relative_rect.size
        center = (width // 2, height)
        radius = height

        arc_rect = pygame.Rect(0, 0, radius * 2, radius * 2)
        arc_rect.midbottom = center
        pygame.draw.arc(self.image, self.border_colour, arc_rect, math.pi, 2* math.pi , self.border_width)
        self.image = pygame.transform.flip(self.image, False, True)

        angle = self._get_angle()
        end_pos = (
            center[0] + (radius - self.border_width*2) * math.cos(angle),
            center[1] + (radius - self.border_width*2) * math.sin(angle)
        )
        pygame.draw.line(self.image, self.dial_colour, center, end_pos, self.dial_thickness)

    # Returns angle value based on relative value (pi is added to make dial go from left to right)
      
    def _get_angle(self):
        relative_value = (self.value - self.min_value) / (self.max_value - self.min_value)
        return math.pi + relative_value * math.pi

    # Clamps value within the min and max range and calls rebuild
      
    def update_value(self, value):
        self.value = r.clamp_value(value, self.min_value, self.max_value)
        self.rebuild()

class UITrackCanvas(UIElement):
    def __init__(self, relative_rect, manager):
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
        self.is_handles_enabled = True
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
                    self.is_click_buffer = True

                    is_point_selected = self.point_selection_handling(relative_mouse_pos)

                    if is_point_selected:
                        if self.selected_point:
                            self.is_dragging = True
                    else:
                        if self.mode == "anchor":
                            insertion_index = self.check_anchor_insertion(relative_mouse_pos)
                            if insertion_index is not None:
                                insertion_index += 1
                                self.insert_anchor(relative_mouse_pos, insertion_index)
                            else:
                                self.create_new_anchor(relative_mouse_pos)
                            self.check_track_completion()
                        if self.mode == "width":
                           self.create_width_point(relative_mouse_pos)

                if event.type == pygame.MOUSEMOTION:
                    if self.is_dragging:
                        point_index = self.selected_point[1]
                        if self.selected_point[0] == 'a':
                            self.handle_anchor_point_movement(relative_mouse_pos, point_index)
                            self.check_track_completion(point_index)
                        elif self.selected_point[0] == 'c':
                            self.handle_control_point_movement(relative_mouse_pos, point_index)
                        # else:
                        #     self.handle_width_point_movement(relative_mouse_pos, point_index)
                if event.type == pygame.MOUSEBUTTONUP and event.button == 1:
                    self.is_click_buffer = False
                    self.is_dragging = False

        elif event.type == pygame.KEYDOWN and event.key == pygame.K_BACKSPACE:
            if not self.selected_point or self.selected_point[0] == 'c' or len(self.anchor_points) <= 2:
                return

            i = self.selected_point[1]

            if self.selected_point[0] == 'a':

                self.anchor_points.pop(i)
                self.selected_point = None

                if not self.is_handles_enabled:
                    return
                if 0 < i < len(self.anchor_points)-1:
                    self.control_points.pop(2*i)
                    self.control_points.pop(2*i-1)
                else:
                    if i == 0:
                        self.control_points.pop(0)
                        self.control_points.pop(0)
                    else:
                        self.control_points.pop(-1)
                        self.control_points.pop(-1)

            else:
                self.widths_dict.pop(i)

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
        self.rebuild(outer_wall_points, inner_wall_points)

    def point_selection_handling(self, mouse_pos):
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

    def check_anchor_insertion(self, mouse_pos):
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
                if dist <= 30:
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
        if not self.is_handles_enabled:
            return
        self.control_points = []
        for i in range(len(self.anchor_points) - 1):
            p1 = self.anchor_points[i]
            p2 = self.anchor_points[i + 1]
            p0 = self.anchor_points[i - 1] if i - 1 >= 0 else p1
            p3 = self.anchor_points[i + 2] if i + 2 < len(self.anchor_points) else p2
            if not self.is_handles_enabled:
                return
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
                    track_proportion = i/(len(self.track_spine)-1)
                    self.widths_dict[track_proportion] = 60
                    self.selected_point = ('w', track_proportion)
                    break

    def handle_anchor_point_movement(self, mouse_pos, point_index):
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

    def handle_width_point_movement(self, mouse_pos, point_position):
        width_point_index = math.floor(max(min(point_position * len(self.track_spine), len(self.track_spine) - 1), 0))
        min_distance = 100000
        closest_index = width_point_index
        for i in range(width_point_index-20, width_point_index+20):
            mouse_distance = (mouse_pos - self.track_spine[i]).length_squared()
            if mouse_distance < min_distance:
                closest_index = i
                min_distance = mouse_distance
        new_position = closest_index/(len(self.track_spine)-1)
        self.widths_dict[new_position] = self.widths_dict[point_position]
        self.widths_dict.pop(point_position)

    def check_track_completion(self, moving_anchor_index = -1):

        if len(self.anchor_points) < 3:
            return False

        first_point = self.anchor_points[0]
        last_point = self.anchor_points[-1]
        moving_anchor = self.anchor_points[moving_anchor_index]

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

    def rebuild(self, outer_wall_points, inner_wall_points):
        width_point_positions = list(self.widths_dict.keys())
        self.image.fill((152, 152, 152))
        r.draw_alternating_line_segments(self.image, self.track_spine)


        for i in range(len(outer_wall_points)-1):
            r.draw_line(self.image, "purple", outer_wall_points[i], outer_wall_points[i+1])
        for i in range(len(inner_wall_points)-1):
            r.draw_line(self.image, "purple", inner_wall_points[i], inner_wall_points[i+1])

        if len(self.anchor_points) != 0:
            if self.mode == "anchor":
                for anchor in self.anchor_points:
                    if anchor == self.anchor_points[0]:
                        pygame.draw.circle(self.image, color="Red", center=anchor, radius=self.point_size + 1)
                    elif anchor == self.anchor_points[-1]:
                        pygame.draw.circle(self.image, color="black", center=anchor, radius=self.point_size + 1)
                    else:
                        pygame.draw.circle(self.image, color= "green", center = anchor, radius = self.point_size)

                for i, handle in enumerate(self.control_points):
                    if i%2 == 0 and i != 0:
                        pygame.draw.line(self.image, "dark grey", handle, self.control_points[i-1], 2)
                    if i == 0:
                        pygame.draw.line(self.image, "dark grey", handle, self.anchor_points[0], 2)
                    if i == len(self.control_points) - 1:
                        pygame.draw.line(self.image, "dark grey", handle, self.anchor_points[-1], 2)

                    pygame.draw.circle(self.image, color="orange", center=handle, radius=self.point_size)
            else:
                for width_point_position in width_point_positions:
                    width_point_index = math.floor(max(min(width_point_position * len(self.track_spine), len(self.track_spine) - 1), 0))
                    pygame.draw.circle(self.image, color="pink", center=self.track_spine[int(width_point_index)], radius=self.point_size)



        overlay = pygame.Surface(self.image.get_size(), pygame.SRCALPHA)
        if self.selected_point:
            point_index = self.selected_point[1]
            point_type = self.selected_point[0]
            if point_type == 'a':
                point = self.anchor_points[point_index]
            elif point_type == 'c':
                point = self.control_points[point_index]
            else:
                width_point_index = math.floor(max(min(point_index * len(self.track_spine), len(self.track_spine) - 1), 0))
                point = self.track_spine[int(width_point_index)]
            pygame.draw.circle(overlay, color=(137, 207, 240, 200), center=point, radius= self.point_size + 2)

        invalid_walls = self.check_validity(outer_wall_points, inner_wall_points)
        for wall in invalid_walls:
            r.draw_line(overlay, (255, 0, 0, 150), wall[0], wall[1], 15)
        self.image.blit(overlay, (0, 0))

        pygame.draw.rect(self.image, "black", pygame.Rect((0, 0), (
        self.image.get_width() - self.border_width/2, self.image.get_height() - self.border_width/2)), width=self.border_width)

    def hash_grid(self, outer_wall_points, inner_wall_points):
        self.shg_grid.clear_grid()
        hash_points = outer_wall_points + inner_wall_points
        for i in range(len(hash_points) - 1):
            if i != len(outer_wall_points) - 1:
                self.shg_grid.hash_segment((hash_points[i], hash_points[i + 1]), i)


    def generate_all_controls(self):
        self.control_points.clear()
        for i in range(len(self.anchor_points) -1):
            p0 = self.anchor_points[i-1] if i > 0 else self.anchor_points[i]
            p1 = self.anchor_points[i]
            p2 = self.anchor_points[i+1]
            p3 = self.anchor_points[i+2] if i < len(self.anchor_points) - 2 else self.anchor_points[i+1]
            self.control_points.extend(self.get_bezier_points(p0, p1, p2, p3))


    @staticmethod
    def get_bezier_points(p0, p1, p2, p3):
        b1 = p1 + (p2 - p0) / 6
        b2 = p2 - (p3 - p1) / 6
        return b1, b2

    def check_validity(self, outer_wall_points, inner_wall_points):
        self.hash_grid(outer_wall_points, inner_wall_points)

        def check_wall_validity(wall_points, wall_start_index):
            hashed_points = outer_wall_points + inner_wall_points
            invalid_segments = []
            for i in range(len(wall_points) - 1):
                segment = (wall_points[i], wall_points[i + 1])
                collisions = self.shg_grid.return_all_collisions(segment, hashed_points)

                actual_segment_index = wall_start_index + i

                for collision_index in collisions:
                    distance = min(abs(collision_index - actual_segment_index), len(wall_points) - 1 - abs(collision_index - actual_segment_index))
                    if ((0< collision_index < len(hashed_points)-1 and 0< i < len(hashed_points)-1)and
                            (distance > 4)):
                        invalid_segments.append((hashed_points[collision_index], hashed_points[collision_index + 1]))

                if Vector2(segment[0] - segment[1]).length_squared() > 4000:
                    invalid_segments.append(segment)
            return invalid_segments

        invalid_outer_walls = check_wall_validity(outer_wall_points, 0)
        invalid_inner_walls = check_wall_validity(inner_wall_points, len(outer_wall_points))

        if len(invalid_inner_walls + invalid_outer_walls) > 0:
            self.validity_issues = "invalid walls"
        elif not self.is_track_complete:
            self.validity_issues = "incomplete"
        elif len(outer_wall_points) > 300 or len(inner_wall_points) > 300:
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


# class UITrackCanvas(UIElement):
#     def __init__(self, relative_rect, manager, bg_colour = "White"):
#         super().__init__(relative_rect, manager, container=None, starting_height=0, layer_thickness=1)
#         self.bg_colour = bg_colour
#         self.strokes = []
#         self.current_stroke = []
#         self.is_dragging = False
#         self.is_drawing = False
#         self.image = pygame.Surface((self.relative_rect.width, self.relative_rect.height), pygame.SRCALPHA)
#         self.rebuild()
#
#
#     def process_event(self, event):
#         if self.is_drawing:
#             if event.type in (pygame.MOUSEBUTTONUP, pygame.MOUSEBUTTONDOWN, pygame.MOUSEMOTION):
#                 if self.rect.collidepoint(event.pos):
#                     if event.type == pygame.MOUSEBUTTONDOWN:
#                         self.is_dragging = True
#                         self.current_stroke.clear()
#                     elif event.type == pygame.MOUSEMOTION and self.is_dragging:
#                         relative_pos = pygame.Vector2((event.pos[0] - self.rect.x, event.pos[1] - self.rect.y))
#                         self.current_stroke.append(relative_pos)
#                     elif event.type == pygame.MOUSEBUTTONUP:
#                         self.strokes.append(self.current_stroke.copy())
#                         self.current_stroke.clear()
#                         self.is_dragging = False
#             self.rebuild()
#
#     def rebuild(self):
#         self.image.fill((0,0,0,0))
#         r.draw_track_outline(self.image, self.strokes)
#         if self.is_dragging:
#             for i in range(len(self.current_stroke) - 1):
#                 pygame.draw.line(self.image, "Red", self.current_stroke[i], self.current_stroke[i+1], 3)
#
#     def save_drawing(self, filename):
#         with open("Track/"+ filename + ".txt", "w") as file:
#             for stroke in self.strokes:
#                 for point in stroke:
#                     file.write(str(point[0]) + "," + str(point[1]) + ",")
#                 file.write("\n")
#             print("saved")
#
#     def load_drawing(self, filename):
#         self.strokes = r.load_track_from_file(filename)
#         self.rebuild()
#
#     def clear(self):
#         self.strokes.clear()
#         self.current_stroke.clear()
#         self.rebuild()
