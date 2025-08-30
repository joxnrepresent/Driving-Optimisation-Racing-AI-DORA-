from collections import defaultdict
from selectors import SelectSelector

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

class Track(UIElement):
    def __init__(self, relative_rect, manager, track_name):
        super().__init__(relative_rect, manager, container=None, starting_height=0, layer_thickness=1)
        self.grid = None
        self.walls = r.load_track_from_file(track_name)
        self.wall_segments = []
        self.is_black = True
        for wall in self.walls:
            for i in range(len(wall) - 1):
                self.wall_segments.append((wall[i], wall[i + 1]))
        self.image = pygame.Surface((self.relative_rect.width, self.relative_rect.height), pygame.SRCALPHA)
        self.rebuild()

    def rebuild(self):
        self.image.fill((0, 0, 0, 0))
        r.draw_track(self.image, self.walls)

class SpatialHashGrid(Track):

    def __init__(self, relative_rect, manager, track_name):

        self.cells = defaultdict(list)
        super().__init__(relative_rect, manager, track_name)
        for segment_index, segment in enumerate(self.wall_segments):
            self.dda_traverse(segment, segment_index)
        self.rebuild()

        r.DEBUG_ELEMENTS["grid lines"].append(True)

    @staticmethod
    def _get_cell_index(point):
        x, y = point
        return int(math.floor(x / r.SHG_CELL_SIZE)), int(math.floor(y / r.SHG_CELL_SIZE))

    def dda_traverse(self, segment, segment_index = None):
        (x1, y1), (x2, y2) = segment
        dx = x2 - x1
        dy = y2 - y1
        ix, iy = self._get_cell_index((x1, y1))
        ex, ey = self._get_cell_index((x2, y2))

        step_x = 1 if dx > 0 else -1 if dx < 0 else 0
        t_delta_x = (r.SHG_CELL_SIZE / abs(dx)) if dx != 0 else float('inf')
        next_boundary_x = (ix + 1) * r.SHG_CELL_SIZE if step_x == 1 else ix * r.SHG_CELL_SIZE
        t_max_x = (next_boundary_x - x1) / dx if dx != 0 else float("inf")

        step_y = 1 if dy > 0 else -1 if dy < 0 else 0
        t_delta_y = (r.SHG_CELL_SIZE / abs(dy)) if dy != 0 else float('inf')
        next_boundary_y = (iy + 1) * r.SHG_CELL_SIZE if step_y == 1 else iy * r.SHG_CELL_SIZE
        t_max_y = (next_boundary_y - y1) / dy if dy != 0 else float("inf")

        max_steps = (abs(ex - ix) + abs(ey - iy) + 10)
        for i in range(max_steps):

            if segment_index is not None:
                self._add_segment_to_grid(ix, iy, segment_index)
            else:
                hit_point = self.ray_cast(segment, ix, iy)
                if hit_point:
                    return hit_point

            if ix == ex and iy == ey:
                break

            if t_max_x < t_max_y:
                ix += step_x
                t_max_x += t_delta_x
            else:
                iy += step_y
                t_max_y += t_delta_y

    def _add_segment_to_grid(self, ix, iy, segment_index):
        self.cells[(ix, iy)].append(segment_index)

    def get_cell_segments(self, cell_index):
        return self.cells[cell_index]

    def ray_cast(self, ray, ix, iy):
        if (ix, iy) in self.cells:
            for seg_index in self.cells[(ix, iy)]:
                seg = self.wall_segments[seg_index]
                hit_point = r.get_line_segment_intersection(ray, seg)
                if hit_point:
                    return hit_point
        return None

    def rebuild(self):
        self.image.fill((0,0,0,0))
        is_black = True
        for cell_segments in self.cells.values():
            wall_segments = [self.wall_segments[i] for i in cell_segments]
            r.draw_segments(self.image, wall_segments, is_black)
            is_black = not is_black






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
