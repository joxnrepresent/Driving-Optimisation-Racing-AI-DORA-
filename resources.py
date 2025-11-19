import math
import pygame
import pygame_gui
from collections import defaultdict
from numpy import exp
from pygame import Vector2

"""
This module contains shared resources like constants, game-state variables (singletons), and utility 
functions/methods used throughout the project.
"""
#---------------------------------------------------------------------------------------------------------------------#

class GameCore:
    def __init__(self):
        # Program constants
        self.is_debugging = False
        self.load_model = False
        self.frame_rate = 60
        self.tick_speedup = 1
        self.screen_dimensions = (1200, 750)
        self.screen_fill = "White"
        self.current_track = "testing_track"
        self.meter_pixel_conversion = 10
        self.eps = 1e-9

        # Core program components
        self.main_screen = pygame.display.set_mode(self.screen_dimensions)
        self.clock = pygame.time.Clock()
        self.pressed_keys = set()
        self.pressed_buttons = set()
        self.game_mode = None

        self.gui_manager = None
        self.game_sprites = pygame.sprite.Group()
        self.debug_elements = defaultdict(list)

    def set_game_mode(self, new_mode):
        self.game_sprites.empty()
        self.debug_elements.clear()
        self.gui_manager.clear_and_reset()

        self.game_mode = new_mode()

    def render(self):
        self.main_screen.fill(self.screen_fill)

        # Debug graphics
        if self.is_debugging:
            if self.debug_elements:
                for element_type, elements in self.debug_elements.items():
                    for element in elements:
                        if element_type == "hitboxes":
                            pygame.draw.polygon(self.main_screen, "red", element, 2)
                        elif element_type == "AABB":
                            pygame.draw.rect(self.main_screen, "red", element.rect, 2)
                        elif element_type == "rays":
                            try:
                                for ray in element:
                                    pygame.draw.line(self.main_screen, "red", ray[0], ray[1])
                            except:
                                print(element)
                        elif element_type == "track spine":
                            for i in range(len(element) - 1):
                                draw_line(self.main_screen, "Blue", element[i], element[i + 1])
                        if element_type == "grid lines":
                            if element:
                                draw_grid(self.main_screen)

        # Main game rendering
        self.game_sprites.draw(self.main_screen)
        self.gui_manager.draw_ui(self.main_screen)
        pygame.display.flip()
        self.clock.tick(game_core.frame_rate)

    def cache_events(self, event):
        self.gui_manager.process_events(event)
        if event.type == pygame.QUIT:
            exit()
        if event.type == pygame.KEYDOWN:
            self.pressed_keys.add(event.key)
        if event.type == pygame.KEYUP:
            self.pressed_keys.discard(event.key)
        if event.type == pygame.USEREVENT and  event.user_type == pygame_gui.UI_BUTTON_PRESSED:
            self.pressed_buttons.add(event.ui_element)

    def process_frame(self):
        if self.game_mode:
            self.game_mode.event_handle()
            self.game_mode.update()

game_core = GameCore()


"""Utility functions"""

# Loads an image using the file name (png only)
def set_image(image):
    return pygame.image.load(f'Images/{image}.png').convert_alpha()

# Returns the sign of the input
def sign(x):
    if x > 0:
        return 1
    elif x < 0:
        return -1
    else:
        return 0

# Keeps input value within a range with an upper and lower limit
def clamp_value(value, lower_limit, upper_limit):
    return max(lower_limit, min(value, upper_limit))

# Wraps a value by looping to the lower limit when the upper limit is crossed
def wrap_value(value, lower_limit, upper_limit):
    return ((value - lower_limit) % (upper_limit-lower_limit)) + lower_limit

def load_track_from_file(filename):
    try:
        with open("Tracks/"+filename + ".txt", "r") as file:
            strokes = []
            for line in file:
                coordinates = line.split(',')
                points = []
                for i in range(0, len(coordinates) - 1, 2):
                    points.append((float(coordinates[i]), float(coordinates[i + 1])))
                strokes.append(points)
            return strokes
    except FileNotFoundError:
        print("File not found")
        return None

def draw_track_outline(surface, strokes):
    is_black = True
    for stroke in strokes:
        for i in range(len(stroke) - 1):
            if is_black:
                colour = "black"
            else:
                colour = "red"
            is_black = not is_black
            pygame.draw.line(surface, colour, stroke[i], stroke[i + 1], 3)

def draw_line(surface, colour, p1, p2):
    pygame.draw.line(surface, colour, p1, p2, 3)

def plot_line(surface, colour, p1, p2):
    pygame.draw.circle(surface, colour, p1, 2)
    pygame.draw.circle(surface, colour, p2, 2)


def draw_alternating_line_segments(surface, segments):
    colour = "black"
    is_black = True
    for (p1, p2) in segments:
        if is_black:
            colour = "black"
        else:
            colour = "red"
        pygame.draw.line(surface, colour, p1, p2, 4)
        is_black = not is_black

def draw_grid(surface, color="green", cell_size = 20):
    w,h = surface.get_size()
    s = pygame.Surface((w,h), pygame.SRCALPHA)
    for x in range(0, w, cell_size):
        pygame.draw.line(s, color, (x,0), (x,h))
    for y in range(0, h, cell_size):
        pygame.draw.line(s, color, (0,y), (w,y))
    surface.blit(s, (0,0))

def get_line_segments_intersection(seg1, seg2):
    (x1, y1), (x2, y2) = seg1
    (x3, y3), (x4, y4) = seg2

    #Trying to early reject intersection if AABB's do not intersect
    if (max(x1, x2) < min(x3, x4) or max(x3, x4) < min(x1, x2) or
            max(y1, y2) < min(y3, y4) or max(y3, y4) < min(y1, y2)):
        return None

    denominator = (x2 - x1) * (y4 - y3) - (y2 - y1) * (x4 - x3)
    alpha_numerator = (x3 - x1) * (y4 - y3) - (y3 - y1) * (x4 - x3)
    beta_numerator = (x3 - x1) * (y2 - y1) - (y3 - y1) * (x2 - x1)

    if abs(denominator) < game_core.eps:
        if abs(alpha_numerator) < game_core.eps and abs(beta_numerator) < game_core.eps:

            dot_prod = (x2 - x1) * (x4 - x3) + (y2 - y1) * (y4 - y3)

            overlap_start_x = max(min(x1, x2), min(x3, x4))
            overlap_end_x = min(max(x1, x2), max(x3, x4))
            overlap_start_y = max(min(y1, y2), min(y3, y4))
            overlap_end_y = min(max(y1, y2), max(y3, y4))

            if overlap_start_x <= overlap_end_x and overlap_start_y <= overlap_end_y:

                if min(x3, x4) <= x1 <= max(x3, x4) and min(y3, y4) <= y1 <= max(y3, y4):
                    return x1, y1

                p1 = (overlap_start_x, overlap_start_y)
                p2 = (overlap_end_x, overlap_end_y)
                if dot_prod > 0:
                    return p1
                else:
                    return p2
        return None

    alpha = alpha_numerator/denominator
    beta = beta_numerator/denominator

    if alpha >= 0 and 0 <= beta <= 1:
        x = x1 + alpha * (x2 - x1)
        y = y1 + alpha * (y2 - y1)
        return x, y
    else:
        return None

def transformed_sigmoid(x):
    return (2.0 / (1.0 + exp(-x)))-1

def generate_track_spine(anchor_points, control_points):
    track_spine = []
    for i in range(len(anchor_points) - 1):
        p1, p2 = anchor_points[i], anchor_points[i + 1]
        b1, b2 = control_points[2 * i], control_points[2 * i + 1]
        track_spine.extend(generate_spine_segments(p1, b1, b2, p2))
    return track_spine

def generate_spine_segments(p1, b1,b2,p2, resolution = 50):
    curve_segments = []
    for i in range(resolution + 1):
        t = i/resolution
        points = [p1, b1, b2, p2]
        curve_point = de_casteljau(points, t)
        if t != 0:
            curve_segments.append((cache_point, curve_point))
        cache_point = curve_point
    return curve_segments

def de_casteljau(points, t):
    if len(points) == 1:
        # Base case
        return points[0]

    # Recursive case
    new_points = []
    for i in range(len(points) - 1):
        new_points.append((1 - t) * points[i] + t * points[i + 1])
    return de_casteljau(new_points, t)

def generate_track(anchor_points, control_points, width = 50):
    track_spine = generate_track_spine(anchor_points, control_points)
    outer_walls = []
    inner_walls = []
    for segment in track_spine:
        start_point, end_point = Vector2(segment[0]), Vector2(segment[1])
        direction_vector = end_point - start_point
        normal_vector = direction_vector.rotate(90).normalize()
        outer_wall = (start_point + normal_vector * width,
                      end_point + normal_vector * width)
        inner_wall = (start_point - normal_vector * width,
                      end_point - normal_vector * width)
        if outer_walls:
            outer_walls.append((outer_walls[-1][1], outer_wall[0]))
        outer_walls.append(outer_wall)
        if inner_walls:
            inner_walls.append((inner_walls[-1][1], inner_wall[0]))
        inner_walls.append(inner_wall)
    return outer_walls, inner_walls

