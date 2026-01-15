import math
import pygame
import pygame_gui
from collections import defaultdict
from numpy import exp
from pygame import Vector2
import json
import numpy as np

"""
This module contains shared resources like constants, game-state variables (singletons), and utility 
functions/methods used throughout the project.
"""
#---------------------------------------------------------------------------------------------------------------------#

class GameCore:
    def __init__(self):
        # Program constants
        self.is_debugging = False
        self.is_paused = False
        self.load_model = False
        self.frame_rate = 50
        self.tick_speedup = 1
        self.screen_dimensions = (1400, 850)
        self.screen_fill = "White"
        self.current_model = None
        self.current_track = None
        self.meter_pixel_conversion = 8
        self.eps = 1e-9

        # Core program components
        self.main_screen = pygame.display.set_mode(self.screen_dimensions, pygame.RESIZABLE)
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


        # Main game rendering
        self.game_sprites.draw(self.main_screen)
        self.gui_manager.draw_ui(self.main_screen)


        # Debug graphics
        if self.is_debugging:
            if self.debug_elements:
                for element_type, elements in self.debug_elements.items():
                    for element in elements:
                        # if element_type == "hitboxes":
                        #     pygame.draw.polygon(self.main_screen, "red", element, 2)
                        # elif element_type == "AABB":
                        #     pygame.draw.rect(self.main_screen, "red", element.rect, 2)
                        if element_type == "rays":
                            try:
                                for ray in element:
                                    pygame.draw.line(self.main_screen, "red", ray[0], ray[1])
                            except:
                                print(element)
                        elif element_type == "track spine":
                            for i in range(len(element) - 1):
                                draw_line(self.main_screen, "Blue", element[i], element[i + 1])
                        # if element_type == "grid lines":
                        #     if element:
                        #         draw_grid(self.main_screen)
        pygame.display.flip()
        self.clock.tick(game_core.frame_rate)

    def cache_events(self, event):
        self.gui_manager.process_events(event)
        if self.is_paused:
            return
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

    def set_file(self, directory, filename):
        if directory == "Track":
            self.current_track = f"{directory}/{filename}.json"
        else:
            self.current_model = f"{directory}/{filename}.npy"

game_core = GameCore()

class RaceTimeManager:
    def __init__(self):
        self.start_time = None
        self.end_times = None
        self.lap_times = None

    def initialise_manager(self, num_of_cars, num_of_laps):
        self.start_time = pygame.time.get_ticks()
        self.end_times = [0] * num_of_cars
        self.lap_times = [[0] * num_of_cars] * num_of_laps

    def get_stats(self):
        end_times_ms = np.array(self.end_times)
        end_times_s = (end_times_ms - self.start_time) * 1000
        lap_times_ms = np.array(self.lap_times)
        lap_times_s = []
        for i, car_lap_times_for_lap in enumerate(lap_times_ms):
            times_s = car_lap_times_for_lap - lap_times_ms[i-1] if i > 0 else car_lap_times_for_lap - self.start_time
            times_s *= 1000
            lap_times_s.append(times_s)

        return end_times_s, lap_times_s

    def update(self, laps_completed, is_lap_completed):
        if is_lap_completed:
            self.lap_times[laps_completed - 1][i] = pygame.time.get_ticks()
        if laps_completed == self.num_of_laps:
            self.end_times[i] = pygame.time.get_ticks()


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
        with open("Track/"+filename + ".txt", "r") as file:
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

def load_bezier_track(filename):
    try:
        with open(f"Track/{filename}.json", "r") as f:
            data = json.load(f)

        anchor_points = [Vector2(point[0], point[1]) for point in data["anchors"]]
        control_points = [Vector2(point[0], point[1]) for point in data["controls"]]
        widths_dict = {float(k): v for k, v in data["widths"].items()}


        return anchor_points, control_points, widths_dict

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

def draw_line(surface, colour, p1, p2, width = 3):
    pygame.draw.line(surface, colour, p1, p2, width)

def plot_line(surface, colour, p1, p2):
    pygame.draw.circle(surface, colour, p1, 2)
    pygame.draw.circle(surface, colour, p2, 2)


def draw_alternating_line_segments(surface, points):
    is_black = True
    if points:
        for i in range(len(points)-1):
            if is_black:
                colour = "black"
            else:
                colour = "red"
            pygame.draw.line(surface, colour, points[i], points[i+1], 4)
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

    if 0 <= alpha <= 1 and 0 <= beta <= 1:
        x = x1 + alpha * (x2 - x1)
        y = y1 + alpha * (y2 - y1)
        return x, y
    else:
        return None


def point_segment_distance(point, segment):
    a, b = segment
    ap = point - a
    ab = a - b

    seg_len_sq = ab.length_squared()

    t = max(0, min(1, ap.dot(ab) / (seg_len_sq + 1e-10)))
    closest_point = a + ab * t
    return (point - closest_point).length()


def transformed_sigmoid(x):
    return (2.0 / (1.0 + exp(-x)))-1

def generate_bezier_track_spine(anchor_points, control_points):
    track_spine = []
    for i in range(len(anchor_points) - 1):
        p1, p2 = anchor_points[i], anchor_points[i + 1]
        b1, b2 = control_points[2 * i], control_points[2 * i + 1]
        track_spine.extend(generate_spine_points(p1, b1, b2, p2))

    cleaned_track_spine = []
    for point in track_spine:
        if not cleaned_track_spine or (point - cleaned_track_spine[-1]).length_squared() > 15:
            cleaned_track_spine.append(point)
    return cleaned_track_spine

def generate_spine_points(p1, b1,b2,p2, resolution = 100):
    spine_points = []
    for i in range(resolution + 1):
        t = i/resolution
        anchor_points = [p1, b1, b2, p2]
        spine_point = de_casteljau(anchor_points, t)
        spine_points.append(spine_point)
    return spine_points

def de_casteljau(points, t):
    if len(points) == 1:
        # Base case
        return points[0]

    # Recursive case
    new_points = []
    for i in range(len(points) - 1):
        new_points.append((1 - t) * points[i] + t * points[i + 1])
    return de_casteljau(new_points, t)

def generate_catmull_rom_track_spine(anchor_points, is_complete):
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

    track_spine = []
    if is_complete:
        for i in range(len(anchor_points)-1):
            p0 = anchor_points[(i - 1) % len(anchor_points)]
            p1 = anchor_points[i]
            p2 = anchor_points[(i + 1) % len(anchor_points)]
            p3 = anchor_points[(i + 2) % len(anchor_points)]
            track_spine.extend(catmull_rom(p0, p1, p2, p3))
    else:
        for i in range(len(anchor_points) -1):
            if i == 0:
                p0 = anchor_points[0] + (anchor_points[0] - anchor_points[1])
            else:
                p0 = anchor_points[i - 1]

            p1 = anchor_points[i]
            p2 = anchor_points[i + 1]

            if i == len(anchor_points) - 2:
                p3 = anchor_points[-1] + (anchor_points[-1] - anchor_points[-2])
            else:
                p3 = anchor_points[i + 2]

            track_spine.extend(catmull_rom(p0, p1, p2, p3))

    cleaned_track_spine = []
    for point in track_spine:
        if not cleaned_track_spine or (point - cleaned_track_spine[-1]).length_squared() > 15:
            cleaned_track_spine.append(point)
    return cleaned_track_spine


def generate_track_walls(track_spine, widths_dict, is_track_complete = False):
    outer_wall_points = []
    inner_wall_points = []
    width_point_locations = list(widths_dict.keys())
    width_point_locations.sort()
    width_indices = []
    for location in width_point_locations:
        width_indices.append(math.floor(location * (len(track_spine)-1)))


    for i in range(1, len(width_indices)):
        start_width_index, end_width_index = width_point_locations[i - 1], width_point_locations[i]
        start_track_index, end_track_index = width_indices[i - 1], width_indices[i]
        start_width = widths_dict[start_width_index]
        d_width = widths_dict[end_width_index] - widths_dict[start_width_index]

        for t in range(start_track_index, end_track_index):
            segment_width = start_width + (t - start_track_index) / (end_track_index - start_track_index) * d_width

            spine_point, next_point = track_spine[t], track_spine[t+1]

            forward = (next_point - spine_point).normalize()
            if t > 0:
                backward = (spine_point - track_spine[t - 1]).normalize()
            else:                                           # Getting vectors to next and prev points
                backward = forward

            tangent = (forward + backward)
            if tangent.length_squared() == 0:              # Getting the tangent vector
                tangent = forward

            # Getting normal to be able to generate 2 pairs of equidistant parallel points
            normal_vector = tangent.rotate(90).normalize()
            outer_wall_point = spine_point - normal_vector * segment_width
            inner_wall_point = spine_point + normal_vector * segment_width
            # Getting squared valid distance to avoid expensive sqrt operation


            def check_wall_point_validity(wall_point):
                valid_distance = (segment_width - 3) ** 2  # Included padding
                tollerence_limit = 80
                for j in range(t - tollerence_limit, t + tollerence_limit):

                    if is_track_complete:
                        spine_index = j % len(track_spine)
                    else:
                        spine_index = max(0, min(j, len(track_spine) - 1))

                    if(wall_point - track_spine[spine_index]).length_squared() < valid_distance:
                        if is_track_complete:
                            index_distance = abs(spine_index - t)
                            wrapped_index_distance = min(index_distance, len(track_spine) - index_distance)
                        else:
                            wrapped_index_distance = abs(spine_index - t)
                        if wrapped_index_distance <= 1:
                            continue
                        return False
                return True

            is_outer_wall_valid= check_wall_point_validity(outer_wall_point)
            is_inner_wall_valid = check_wall_point_validity(inner_wall_point)

            if is_outer_wall_valid:
                outer_wall_points.append(outer_wall_point)
            if is_inner_wall_valid:
                inner_wall_points.append(inner_wall_point)

    return outer_wall_points, inner_wall_points

