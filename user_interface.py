import pygame
import pygame_gui
import math
from pygame_gui.core import UIElement
import resources

"""This module contains custom UI elements not includer in the pygame_gui package"""

"""This class is for a meter that displays values as a gauge (like a speedometer/rpm meter)"""
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

    """Redraws updated version of meter"""
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

    """Returns angle value based on relative value (pi is added to make dial go from left to right)"""
    def _get_angle(self):
        relative_value = (self.value - self.min_value) / (self.max_value - self.min_value)
        return math.pi + relative_value * math.pi

    """Clamps value within the min and max range and calls rebuild"""
    def update_value(self, value):
        self.value = resources.clamp_value(value, self.min_value, self.max_value)
        self.rebuild()

