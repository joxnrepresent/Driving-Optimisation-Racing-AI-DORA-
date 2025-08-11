import pygame
import pygame_gui
import math
from pygame import Vector2
from pygame_gui.core import UIElement
import resources

"""This module contains custom GUI elements not included in the pygame_gui library"""
#---------------------------------------------------------------------------------------------------------------------#

# -----------------------------------#
"""This class is for a meter that displays values as a gauge (like a speedometer/rpm meter)"""
# -----------------------------------#
class UIGaugeMeter(UIElement):
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

        self.image = pygame.Surface((self.relative_rect.width + border_width, self.relative_rect.height + border_width), pygame.SRCALPHA)
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
        self.value = resources.clamp_value(value, self.min_value, self.max_value)
        self.rebuild()

class UICanvas(UIElement):
    def __init__(self, relative_rect, manager, bg_colour = "White"):
        super().__init__(relative_rect, manager, container=None, starting_height=0, layer_thickness=1)
        self.bg_colour = bg_colour
        self.save_points = []
        self.dragging_points = []
        self.dragging = False
        self.image = pygame.Surface((self.relative_rect.width, self.relative_rect.height), pygame.SRCALPHA)
        self.rebuild()

    def process_event(self, event):
        if event.type in (pygame.MOUSEBUTTONUP, pygame.MOUSEBUTTONDOWN, pygame.MOUSEMOTION):
            if self.rect.collidepoint(event.pos):
                if event.type  == pygame.MOUSEBUTTONDOWN:
                    self.dragging = True
                elif event.type == pygame.MOUSEMOTION and self.dragging:
                    relative_pos = Vector2((event.pos[0] - self.rect.x, event.pos[1] - self.rect.y))
                    self.dragging_points.append(relative_pos)
                    self.save_points.append(relative_pos)
                elif event.type == pygame.MOUSEBUTTONUP:
                    self.dragging_points.clear()
                    self.dragging = False

        self.rebuild()

    def rebuild(self, saved_points = None):
        if saved_points is None:
            for i in range(len(self.dragging_points) - 2):
                pygame.draw.line(self.image, "Red", self.dragging_points[i], self.dragging_points[i+1], 3)
        else:
            for i in range(len(saved_points) - 2):
                pygame.draw.line(self.image, "Red", saved_points[i], saved_points[i + 1], 3)

    def save_drawing(self, filename):
        with open(filename + ".txt", "w") as file:
            for point in self.save_points:
                file.write(point + ",")
            print("saved")

    def open_drawing(self, filename):
        try:
            with open(filename+".txt", "w") as file:
                self.save_points = file.read().split(',')
                self.rebuild(self.save_points)
        except FileNotFoundError:
            print("File not found")

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
