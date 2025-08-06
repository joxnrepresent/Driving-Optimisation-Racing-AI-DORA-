def draw(screen, elements):
    for element in elements:
        if isinstance(element[0], pygame.Rect):
            pygame.draw.rect(screen, element[1], element[0])
        elif isinstance(element[0], pygame.Surface):
            screen.blit(element[0], element[1])
        elif element[0] == "semicircle":
            draw_semicircle(screen, * element[1:])
        elif element[0] == "line":
            pygame.draw.line(screen, * element[1:])

def draw_semicircle(screen, center, radius, fill_colour, start_angle = 0, border_width = 5, border_colour = "Black", resolution = 5 ):
    points = [center]
    for i in range(180*resolution + 1 ):
        angle = start_angle + i/resolution
        x = center[0] + radius * math.cos(math.radians(angle))
        y = center[1] - radius * math.sin(math.radians(angle))
        points.append((x, y))
    pygame.draw.polygon(screen, fill_colour, points)
    pygame.draw.polygon(screen, border_colour, points, border_width)


    DRAG_COEFFICIENT = 0.4257
    ROLLING_RESISTANCE_COEFFICIENT = 12.8


#
# class Meter:
#     def __init__(self, min_value, max_value):
#         self.min_value = min_value
#         self.max_value = max_value
#         self.value = 0
#         self.range_of_values = self.max_value - self.min_value
#         self.dragging = False
#
#     def update_value(self, value):
#         self.value = resources.clamp_value(value, self.min_value, self.max_value)
#         return (self.value - self.min_value) / self.range_of_values
#
# class HorizontalMeter(Meter):
#     def __init__(self, x, y, width, height, base_colour, knob_colour, min_value, max_value):
#         super().__init__(min_value, max_value)
#         self.knob_width = 10
#         self.base = pygame.Rect(x, y, width, height)
#         self.knob = pygame.Rect((x + (width - self.knob_width) // 2), y, self.knob_width, height)
#         self.elements = ((self.base, base_colour), (self.knob, knob_colour))
#
#     def update_value(self, value):
#         relative_value = super().update_value(value)
#         self.knob.x = relative_value * self.base.width + self.base.left

# class Slider(HorizontalMeter):
#     def event_handle(self, event):
#         if self.knob.collidepoint(event.pos):
#             if event.type == pygame.MOUSEBUTTONDOWN and event.button == 1:
#                self.dragging = True
#             if event.type == pygame.MOUSEBUTTONUP and event.button == 1:
#                 self.dragging = False
#
#         if self.dragging:
#             click_position = resources.clamp_value(event.pos[0], self.base.left, self.base.right)
#             relative_position = click_position - self.base.left
#
#             percentage_increment = relative_position / (self.base.right - self.base.left)
#             value = round(percentage_increment * self.range_of_values, 2) + self.min_value
#             super().update_value(value)