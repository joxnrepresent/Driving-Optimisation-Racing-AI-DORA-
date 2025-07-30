import math

import pygame

import resources


class Meter:
    def __init__(self, min_value, max_value):
        self.min_value = min_value
        self.max_value = max_value
        self.value = 0
        self.range_of_values = self.max_value - self.min_value
        self.dragging = False

    def update_value(self, value):
        self.value = resources.clamp_value(value, self.min_value, self.max_value)
        return (self.value - self.min_value) / self.range_of_values

class HorizontalMeter(Meter):
    def __init__(self, x, y, width, height, base_colour, knob_colour, min_value, max_value):
        super().__init__(min_value, max_value)
        self.knob_width = 10
        self.base = pygame.Rect(x, y, width, height)
        self.knob = pygame.Rect((x + (width - self.knob_width) // 2), y, self.knob_width, height)
        self.elements = ((self.base, base_colour), (self.knob, knob_colour))

    def update_value(self, value):
        relative_value = super().update_value(value)
        self.knob.x = relative_value * self.base.width + self.base.left


class GaugeMeter(Meter):
    def __init__(self, x, y, radius, max_value, fill_colour, dial_colour, border_width = 3, dial_thickness = 2):
        super().__init__(0, max_value)
        self.x = x
        self.y = y
        self.dial_radius = radius - border_width
        self.base = ["semicircle", (x, y), radius, fill_colour, 0, border_width*2]
        self.dial = ["line", dial_colour, (self.x, self.y), (self.x - self.dial_radius, self.y), dial_thickness]
        self.elements = (self.base, self.dial)

    def update_value(self, value):
        relative_value = super().update_value(value)
        dial_angle = relative_value * 180
        self.dial[3] = (self.x + self.dial_radius * math.cos(math.radians(180 - dial_angle)),
            self.y - self.dial_radius * math.sin(math.radians(180 -dial_angle)))

class Slider(HorizontalMeter):
    def event_handle(self, event):
        if self.knob.collidepoint(event.pos):
            if event.type == pygame.MOUSEBUTTONDOWN and event.button == 1:
               self.dragging = True
            if event.type == pygame.MOUSEBUTTONUP and event.button == 1:
                self.dragging = False

        if self.dragging:
            click_position = resources.clamp_value(event.pos[0], self.base.left, self.base.right)
            relative_position = click_position - self.base.left

            percentage_increment = relative_position / (self.base.right - self.base.left)
            value = round(percentage_increment * self.range_of_values, 2) + self.min_value
            super().update_value(value)