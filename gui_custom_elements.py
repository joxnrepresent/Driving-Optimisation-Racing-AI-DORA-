from collections import defaultdict

import pygame
import pygame_gui
import math

from numpy.ma.core import indices
from pygame import Vector2
from pygame_gui.core import UIElement
import resources as r

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

        self.image = pygame.Surface((self.relative_rect.width + border_width,
                                     self.relative_rect.height + border_width), pygame.SRCALPHA)
        self.rebuild()

    # Redraws updated version of meter
    # -----------------------------------#
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
    # -----------------------------------#
    def _get_angle(self):
        relative_value = (self.value - self.min_value) / (self.max_value - self.min_value)
        return math.pi + relative_value * math.pi

    # Clamps value within the min and max range and calls rebuild
    # -----------------------------------#
    def update_value(self, value):
        self.value = r.clamp_value(value, self.min_value, self.max_value)
        self.rebuild()




class TrackCanvas(UIElement):
    def __init__(self, relative_rect, manager, bg_colour = "White"):
        super().__init__(relative_rect, manager, container=None, starting_height=0, layer_thickness=1)
        self.bg_colour = bg_colour
        self.strokes = []
        self.current_stroke = []
        self.is_dragging = False
        self.is_drawing = False
        self.image = pygame.Surface((self.relative_rect.width, self.relative_rect.height), pygame.SRCALPHA)
        self.rebuild()


    def process_event(self, event):
        if self.is_drawing:
            if event.type in (pygame.MOUSEBUTTONUP, pygame.MOUSEBUTTONDOWN, pygame.MOUSEMOTION):
                if self.rect.collidepoint(event.pos):
                    if event.type == pygame.MOUSEBUTTONDOWN:
                        self.is_dragging = True
                        self.current_stroke.clear()
                    elif event.type == pygame.MOUSEMOTION and self.is_dragging:
                        relative_pos = Vector2((event.pos[0] - self.rect.x, event.pos[1] - self.rect.y))
                        self.current_stroke.append(relative_pos)
                    elif event.type == pygame.MOUSEBUTTONUP:
                        self.strokes.append(self.current_stroke.copy())
                        self.current_stroke.clear()
                        self.is_dragging = False
            self.rebuild()

    def rebuild(self):
        self.image.fill((0,0,0,0))
        r.draw_track(self.image, self.strokes)
        if self.is_dragging:
            for i in range(len(self.current_stroke) - 1):
                pygame.draw.line(self.image, "Red", self.current_stroke[i], self.current_stroke[i+1], 3)

    def save_drawing(self, filename):
        with open(filename + ".txt", "w") as file:
            for stroke in self.strokes:
                for point in stroke:
                    file.write(str(point[0]) + "," + str(point[1]) + ",")
                file.write("\n")
            print("saved")

    def load_drawing(self, filename):
        self.strokes = r.load_track_from_file(filename)
        self.rebuild()

    def clear(self):
        self.strokes.clear()
        self.current_stroke.clear()
        self.rebuild()

class SpatialHashGrid:
    def __init__(self, cell_size = 10):
        self.cell_size = cell_size
        self.cells = defaultdict(list)

    def _get_cell(self, point):
        x, y = point
        return int(x // self.cell_size), int(y // self.cell_size)

    def add_segment(self, segment):
        (x1, y1), (x2, y2) = segment

        min_point = (min(x1, x2), min(y1, y2))
        max_point = (max(x1, x2), max(y1, y2))

        min_cell = self._get_cell(min_point)
        max_cell = self._get_cell(max_point)

        for i in range(min_cell[0], max_cell[0] + 1):
            for j in range(min_cell[1], max_cell[1] + 1):
                self.cells[(i, j)].append(segment)

    def get_cell_segments(self, point):
        cell = self._get_cell(point)
        return self.cells.get(cell, [])


class Track(UIElement):
    def __init__(self, relative_rect, manager, track_name):
        super().__init__(relative_rect, manager, container=None, starting_height=0, layer_thickness=1)
        self.grid = None
        self.walls = r.load_track_from_file(track_name)
        self.wall_segments = []
        points = []
        for wall in self.walls:
            points.extend(wall)
        for i in range(0, len(points) - 1, 2):
            self.wall_segments.append((points[i], points[i + 1]))
        self.image = pygame.Surface((self.relative_rect.width, self.relative_rect.height), pygame.SRCALPHA)
        self.rebuild()

    def generate_spatial_hash_grid(self):
        self.grid = SpatialHashGrid()
        for segment in self.wall_segments:
            self.grid.add_segment(segment)


    def rebuild(self):
        # r.draw_track(self.image, self.walls)
        for seg in self.wall_segments[:5]:
            print("Segment:", seg)

        is_black = True
        self.generate_spatial_hash_grid()
        for cell_segments in self.grid.cells.values():
            colour = "red"
            if is_black:
                colour = "black"
            r.draw_track(self.image, cell_segments, colour)
            is_black = not  is_black





#             pygame.draw.line(self.image, "Red", self.control_points[0], self.control_points[1], 3)
# class UIBezierCanvas(UIElement):
#     def __init__(self, relative_rect, manager):
#         super().__init__(relative_rect, manager, container=None, starting_height=0, layer_thickness=1)
#         self.control_points = []
#
#         self.image = pygame.Surface((self.relative_rect.width, self.relative_rect.height), pygame.SRCALPHA)
#         self.rebuild()
#
#     def process_event(self, event):
#         if event.type  == pygame.MOUSEBUTTONDOWN:
#             if self.rect.collidepoint(event.pos):
#                 relative_pos = Vector2((event.pos[0] - self.rect.x, event.pos[1] - self.rect.y))
#
#                 if len(self.control_points) == 3:
#                     self.control_points.pop(0)
#                 self.control_points.append(relative_pos)
#             self.rebuild()
#
#     def rebuild(self):
#         if len(self.control_points) >2:
#             p0, p1, p2 = self.control_points
#             points = [p0]
#             for i in range(50):
#                 t = i/50
#                 l0 = p0.lerp(p1, t)
#                 l1 = p1.lerp(p2, t)
#                 q0 = l0.lerp(l1, t)
#                 points.append(q0)
#             for i in range(len(points)-2):
#                 pygame.draw.line(self.image, "Red", points[i], points[i+1], 3)
#
#         elif len(self.control_points) == 2:
#             pygame.draw.line(self.image, "Red", self.control_points[0], self.control_points[1], 3)
