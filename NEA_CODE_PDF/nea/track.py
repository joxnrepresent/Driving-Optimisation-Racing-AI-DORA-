from nea import resources as r
from math import floor,sqrt
from pygame.sprite import Sprite
from pygame import Surface, Rect, SRCALPHA, Vector2
from collections import defaultdict
from nea.resources import game_core
import pygame
"""
Track class and Spacial Hash Grid data structure 

Implements race track sprite with optimised collision detection.
Supports hitbox collision for wall detection and ray casting for distance sensors.
Spatial hash grid enables optimised collision point checks.
Classes:
    Track: Race track sprite with collision detection and ray casting
    SpatialHashGrid: Spatial hash grid for efficient collision queries
"""

class Track(Sprite):
    """
    Track sprite with collision detection methods.

    Generates track spine from Bézier curve data. Generates track walls from the generated track spine and the width
    point data. Renders track with finish line and walls.

    Attributes:
        grid (SpatialHashGrid): Spatial hash for collision queries
        track_spine (list[Vector2]): Track centerline points
        outer_wall_points (list[Vector2]): Outer wall vertices
        inner_wall_points (list[Vector2]): Inner wall vertices
        wall_segments (list[tuple]): All wall segments for collision
        rect (Rect): Track bounding rectangle
        image (Surface): Track rendering surface
    """

    def __init__(self, anchors, controls, widths, position = ((0,0), game_core.screen_dimensions)):
        super().__init__()
        self.grid = SpatialHashGrid()


        self.track_spine = r.generate_bezier_track_spine(anchors, controls)
        if 1.0 not in widths:
            widths[1.0] = widths[0.0]
        outer_wall_points, inner_wall_points = r.generate_track_walls(self.track_spine, widths)
        if outer_wall_points[0] != outer_wall_points[-1]:
            outer_wall_points.append(outer_wall_points[0])
        if inner_wall_points[0] != inner_wall_points[-1]:
            inner_wall_points.append(inner_wall_points[0])
        self.outer_wall_points = outer_wall_points
        self.inner_wall_points = inner_wall_points
        self.wall_segments = []
        for i in range(len(outer_wall_points) - 1):
            segment = (outer_wall_points[i], outer_wall_points[i+1])
            self.wall_segments.append(segment)
        for i in range(len(inner_wall_points) - 1):
            segment = (inner_wall_points[i], inner_wall_points[i+1])
            self.wall_segments.append(segment)
        for i, segment in enumerate(self.wall_segments):
            self.grid.hash_segment(segment, i)

        self.rect = Rect(position)
        self.image = Surface((self.rect.width, self.rect.height), SRCALPHA)
        self._draw_track()

    def ray_cast(self, ray):
        """
        Get intersection point of ray with track wall, and normalised distance to wall for sensor data.

        Args:
            ray (Vector2): Start and end point for ray cast
        Returns:
            Vector2: Intersection point with wall (end of ray if no intersection point)
            float: Normalised distance to wall
        """
        ray_start = Vector2(ray[0])
        ray_end = Vector2(ray[1])
        hit_point = self.grid.return_collision_point(ray, self.wall_segments)
        if not hit_point:
            hit_point = ray_end
        normalised_distance = ((hit_point - ray_start).length_squared() / (ray_end-ray_start).length_squared())
        pygame.draw.line(self.image, (0, 255, 255), ray_start, hit_point)
        return hit_point, sqrt(normalised_distance)

    def hitbox_collision_detection(self, hitbox):
        """
        Detects collision between hitbox and wall segment.

        Args:
            hitbox (list(Vector2)): Corners of hitbox
        Returns:
            bool: True if collision detected, False otherwise
        """
        for i in range(len(hitbox)):
            hitbox_border_line = (hitbox[i], hitbox[(i + 1) % len(hitbox)] )
            collision = self.grid.return_collision_point(hitbox_border_line, self.wall_segments)
            if collision is not None:
                return True
        return False


    def _draw_track(self):
        """
        Draw track sprite with walls, road and finish line.
        """
        self.image.fill((0, 0, 0, 0))

        # Draw road in grey
        track_polygon = self.outer_wall_points + self.inner_wall_points
        pygame.draw.polygon(self.image, (80, 80, 80), track_polygon)
        self._draw_checkered_line()
        self._draw_alternating_wall()

        game_core.game_mode.debug_elements["track spine"] = [self.track_spine]

    def _draw_alternating_wall(self,width=10):
        """
        Draw walls with alternating red and white strips.

        Args:
            width (int): Wall line width
        """

        is_black = True
        current_segment_length = 0
        for segment in self.wall_segments:
            color = (200, 200, 200) if is_black else (200, 0, 0)
            pygame.draw.line(self.image, color, segment[0], segment[1], width)
            current_segment_length += Vector2(segment[0] - segment[1]).length()
            if current_segment_length >= 50:
                current_segment_length = 0
                is_black = not is_black

    def _draw_checkered_line(self, thickness = 20, checkered_rows = 10):
        """
        Draw checkered finish line across track start.

        Args:
            thickness (int): Line thickness perpendicular to track
            checkered_rows (int): Number of checker squares
        """
        finish_line = self.inner_wall_points[0] - self.outer_wall_points[0]
        length = finish_line.length()

        unit_vector = finish_line.normalize()
        normal_vector = Vector2(-unit_vector.y, unit_vector.x)

        check_length = length / checkered_rows

        for i in range(checkered_rows):
            start = self.outer_wall_points[0] + unit_vector * (i * check_length)
            end = self.outer_wall_points[0] + unit_vector * ((i + 1) * check_length)

            color = (255, 255, 255) if i % 2 == 0 else (0, 0, 0)

            # Get checkered square corners
            p1 = start + normal_vector * (thickness / 2)
            p2 = end + normal_vector * (thickness / 2)
            p3 = end - normal_vector * (thickness / 2)
            p4 = start - normal_vector * (thickness / 2)

            pygame.draw.polygon(self.image, color, [p1, p2, p3, p4])

    def get_starting_orientation(self):
        """
        Get starting heading direction of car on track

        Returns:
            float: Starting heading direction
        """
        start_line = Vector2(self.outer_wall_points[0] - self.inner_wall_points[0])
        unit_vector = start_line.normalize()
        normal_vector = Vector2(-unit_vector.y, unit_vector.x)
        return normal_vector.as_polar()[1] + 270

    def update(self, *args, **kwargs):
        self._draw_track()

class SpatialHashGrid:
    """
    Segments the set of wall segments into a hash set of cells, with each cell coordinate as the key, and a list of
    wall segments in that cell as the value. Also includes dda traversal method and helper methods to hash and query
    cells.
    """
    def __init__(self, cell_size = 10):
        self.cells = defaultdict(list)
        self.cell_size = cell_size
        game_core.debug_elements["grid lines"] = [True]

    def get_cell_index_of_point(self, point):
        x, y = point
        return int(floor(x / self.cell_size)), int(floor(y / self.cell_size))

    def _add_segment_to_cell(self, ix, iy, segment_index):
        self.cells[(ix, iy)].append(segment_index)

    def _check_cell_for_collision(self, ix, iy, ray, wall_segments):
        """
        Check single cell for any collision of track wall segment with ray.

        Args:
            ix, iy (int): Wall segment (x,y) index
            ray (Vector2): Ray start and end point
            wall_segments (list(Vector2)): List of track wall segments
        Returns:
            Vector2: coordinates of collision point (None if no collision)
        """
        if (ix, iy) in self.cells:
            for seg_index in self.cells[(ix, iy)]:
                seg = wall_segments[seg_index]
                hit_point = r.get_line_segments_intersection(ray, seg)
                if hit_point:
                    return hit_point
        return None

    def _get_all_collision_indices(self, ix, iy, check_segment, wall_points_list):
        """
        Get indices of all collisions of walls with a check segment (used for validity check in track maker)

        Args:
            ix, iy (int): Wall segment (x,y) index
            check_segment (Vector2): line segment to be checked
            wall_points_list (list(Vector2)): List of track wall segments
        """
        collisions = []
        if (ix, iy) in self.cells:
            for seg_index in self.cells[(ix, iy)]:
                seg = (wall_points_list[seg_index], wall_points_list[seg_index + 1])
                hit_point = r.get_line_segments_intersection(check_segment, seg)
                if hit_point:
                   collisions.append(seg_index)
        return collisions


    def _dda_grid_traverse(self, segment, on_visit):
        """
        Digital differential analyser (DDA) algorithm for traversing grid.

        Args:
            segment (Vector2): Line segment end points
            on_visit (lamda function): Function to be called when collision detected.

        Returns:
            result of lambda function
        """
        (x1, y1), (x2, y2) = segment

        # Change in x and change in y
        dx = x2 - x1
        dy = y2 - y1

        # Start cell indices
        start_cell_x, start_cell_y = self.get_cell_index_of_point((x1, y1))
        end_cell_x, end_cell_y = self.get_cell_index_of_point((x2, y2))

        """
        parameter t -> distance along the segment [0,1]
        """
        cell_step_x = r.sign(dx)
        t_delta_x = (self.cell_size / abs(dx)) if dx != 0 else float('inf')
        next_boundary_x = (start_cell_x + 1) * self.cell_size if cell_step_x == 1 else start_cell_x * self.cell_size
        t_max_next_boundary_x = (next_boundary_x - x1) / dx if dx != 0 else float("inf")

        cell_step_y = r.sign(dy)
        t_delta_y = (self.cell_size / abs(dy)) if dy != 0 else float('inf')
        next_boundary_y = (start_cell_y + 1) * self.cell_size if cell_step_y == 1 else start_cell_y * self.cell_size
        t_max_next_boundary_y = (next_boundary_y - y1) / dy if dy != 0 else float("inf")


        current_cell_x = start_cell_x
        current_cell_y = start_cell_y
        # Set maximum number of steps
        max_steps = (abs(end_cell_x - start_cell_x) + abs(end_cell_y - start_cell_y) + 10)
        for i in range(max_steps):

            # Call on visit function at each step
            result = on_visit(current_cell_x, current_cell_y)

            if result is not None:
                # Return if the required value has been found (Like collision point).
                return result

            if current_cell_x == end_cell_x and current_cell_y == end_cell_y:
                # Exit if end of segment is reached
                break

            # Step in the x or y direction depending upon distance to boundary
            if t_max_next_boundary_x < t_max_next_boundary_y:
                current_cell_x += cell_step_x
                t_max_next_boundary_x += t_delta_x
            else:
                current_cell_y += cell_step_y
                t_max_next_boundary_y += t_delta_y

    def hash_segment(self, wall_segment, segment_index):
        self._dda_grid_traverse(wall_segment, on_visit=lambda ix, iy: self._add_segment_to_cell(ix, iy, segment_index))

    def return_collision_point(self, ray, wall_segments):
        return self._dda_grid_traverse(ray, on_visit=lambda ix, iy:
        self._check_cell_for_collision(ix, iy, ray, wall_segments))

    def return_all_collisions(self, check_segment, points_list):
        return self._dda_grid_traverse(check_segment, on_visit=lambda ix, iy:
        self._get_all_collision_indices(ix, iy, check_segment, points_list))

    def clear_grid(self):
        self.cells = defaultdict(list)

    def get_wall_segment_indices(self):
        cell_elements = self.cells.values()
        indices = []
        for element in cell_elements:
            indices.extend(element)
        return indices
