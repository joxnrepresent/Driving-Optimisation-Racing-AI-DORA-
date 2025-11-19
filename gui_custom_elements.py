import math
from operator import index
from typing import Union
import random

from fontTools.cu2qu import curves_to_quadratic
from numpy.f2py.crackfortran import lenarraypattern

import resources as r
import pygame
from pygame.math import Vector2
from pygame_gui.core import UIElement
"""
This module contains custom GUI elements not included in the pygame_gui library
"""
#---------------------------------------------------------------------------------------------------------------------#

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
        self.anchor_points = []
        self.control_points = []
        self.point_size = 5
        self.track_spine = []
        self.walls = []
        self.max_handle_length = 100
        self.image = pygame.Surface((relative_rect.width, relative_rect.height))
        self.is_dragging = False
        self.is_click_buffer = False
        self.is_anchor_point_selected = False
        self.drag_point = None
        self.mouse_pos = None
        self.rebuild()

    def process_event(self, event):
        if event.type in (pygame.MOUSEBUTTONDOWN, pygame.MOUSEBUTTONUP, pygame.MOUSEMOTION):
            if self.rect.collidepoint(event.pos):
                relative_mouse_pos = Vector2(event.pos[0] - self.rect.x, event.pos[1] - self.rect.y)
                if event.type == pygame.MOUSEBUTTONDOWN and event.button == 1:
                    self.is_click_buffer = True
                    is_point_clicked = False
                    clicked_point = None
                    for i, point in enumerate(self.anchor_points):
                        if (relative_mouse_pos - point).length() <= self.point_size:
                            is_point_clicked = True
                            self.is_anchor_point_selected = True
                            clicked_point = (i, point)

                    for i, point in enumerate(self.control_points):
                        if (relative_mouse_pos - point).length() <= self.point_size:
                            is_point_clicked = True
                            self.is_anchor_point_selected = False
                            clicked_point = (i, point)

                    if is_point_clicked:
                        self.is_dragging = True
                        self.drag_point = clicked_point
                    else:
                        self.anchor_points.append(relative_mouse_pos)
                        if len(self.anchor_points) > 1:
                            p0 = self.anchor_points[-3] if len(self.anchor_points) >2 else self.anchor_points[-2]
                            p1 = self.anchor_points[-2]
                            p2 = self.anchor_points[-1]
                            p3 = relative_mouse_pos
                            self.control_points.extend(self.get_bezier_points(p0, p1, p2, p3))
                            if len(self.anchor_points) > 2:
                                anchor = self.anchor_points[-2]
                                handle_length = (self.control_points[-2] - anchor).length()
                                direction = (self.control_points[-2] - anchor).normalize()

                                updated_handle_point = anchor - handle_length * direction
                                self.control_points[-3] = updated_handle_point

                if event.type == pygame.MOUSEMOTION:
                    if self.is_dragging:
                        point_index = self.drag_point[0]
                        if self.is_anchor_point_selected:
                            translate_vector = relative_mouse_pos - self.anchor_points[point_index]
                            self.anchor_points[point_index] = relative_mouse_pos
                            if point_index*2 < len(self.control_points):
                                self.control_points[point_index * 2] += translate_vector
                            if point_index != 0:
                                self.control_points[point_index * 2 - 1] += translate_vector

                        else:
                            corresponding_anchor_index = point_index//2 if point_index % 2 == 0 else point_index //2+1
                            anchor_point = self.anchor_points[corresponding_anchor_index]
                            clamped_translation = (relative_mouse_pos - anchor_point).clamp_magnitude(self.max_handle_length)
                            moved_point = anchor_point + clamped_translation
                            self.control_points[point_index] = moved_point
                            mirror_point_index = None
                            if (point_index % 2 == 0
                                    and point_index != 0
                                    and point_index != len(self.control_points) - 1):
                                mirror_point_index = point_index - 1
                            elif (point_index != 0
                                    and point_index != len(self.control_points) - 1):
                                mirror_point_index = point_index + 1
                            if mirror_point_index:
                                self.control_points[mirror_point_index] = anchor_point - clamped_translation
                        self.mouse_pos = None
                    else:
                        self.mouse_pos = relative_mouse_pos

                if event.type == pygame.MOUSEBUTTONUP and event.button == 1:
                    self.is_click_buffer = False
                    self.is_dragging = False
                    self.drag_point = None

            self.rebuild()

    def rebuild(self):
        self.image.fill("White")

        for anchor in self.anchor_points:
            pygame.draw.circle(self.image, color= "Red", center = anchor, radius = self.point_size)

        for i, handle in enumerate(self.control_points):
            pygame.draw.circle(self.image, color="Blue", center=handle, radius=self.point_size)
            if i%2 == 0 and i != 0:
                pygame.draw.line(self.image, "green", handle, self.control_points[i-1], 2)
            if i == 0:
                pygame.draw.line(self.image, "green", handle, self.anchor_points[0], 2)
            if i == len(self.control_points) - 1:
                pygame.draw.line(self.image, "green", handle, self.anchor_points[-1], 2)

        self.track_spine = r.generate_track_spine(self.anchor_points, self.control_points)
        r.draw_alternating_line_segments(self.image, self.track_spine)
        outer_walls, inner_walls = r.generate_track(self.anchor_points, self.control_points)
        self.walls = outer_walls + inner_walls


    @staticmethod
    def catmull_rom(p0, p1, p2, p3, resolution = 100):
        # Caching coefficients of cubic in terms of t (at^3 + bt^2 + ct + d)
        a = -p0 + 3*p1 - 3*p2 + p3
        b = 2*p0 - 5*p1 + 4*p2 - p3
        c = -p0 + p2
        d = 2*p1

        curve_points = []
        for i in range(resolution + 1):
            t = i/resolution
            t2 = t*t
            t3 = t2 * t
            point = 0.5 * (a*t3 + b*t2 + c*t + d)
            curve_points.append(point)

        return curve_points

    @staticmethod
    def get_bezier_points(p0, p1, p2, p3):
        b1 = p1 + (p2 - p0) / 6
        b2 = p2 - (p3 - p1) / 6
        return b1, b2







#
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
#         with open("Tracks/"+ filename + ".txt", "w") as file:
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
