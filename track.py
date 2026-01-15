import resources as r
from math import floor,sqrt
from pygame.sprite import Sprite
from pygame import Surface, Rect, SRCALPHA, Vector2
from collections import defaultdict
from resources import game_core

class Track(Sprite):

    def __init__(self, track_name, position = ((0,0), game_core.screen_dimensions)):
        super().__init__()
        self.grid = SpatialHashGrid()
        anchors, controls, widths = r.load_bezier_track(track_name)
        self.track_spine = r.generate_bezier_track_spine(anchors, controls)
        if 1.0 not in widths:
            widths[1.0] = widths[0.0]
        outer_wall_points, inner_wall_points = r.generate_track_walls(self.track_spine, widths)
        if outer_wall_points[0] != outer_wall_points[-1]:
            outer_wall_points.append(outer_wall_points[0])
        if inner_wall_points[0] != inner_wall_points[-1]:
            inner_wall_points.append(inner_wall_points[0])
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
        self.draw_track()

    # Cast ray and return shortened ray up till nearest point of intersection
    # Returns normalised distance which is the percentage of max length
    def ray_cast(self, ray):
        ray_start = Vector2(ray[0])
        ray_end = Vector2(ray[1])
        hit_point = self.grid.return_collision_point(ray, self.wall_segments)
        if not hit_point:
            hit_point = ray_end
        normalised_distance = ((hit_point - ray_start).length_squared() / (ray_end-ray_start).length_squared())
        return hit_point, sqrt(normalised_distance)

    # Checks each border of the hitbox for collision with track segments in the cells it passes through
    def hitbox_collision_detection(self, hitbox):
        for i in range(len(hitbox)):
            hitbox_border_line = (hitbox[i], hitbox[(i + 1) % len(hitbox)] )
            collision = self.grid.return_collision_point(hitbox_border_line, self.wall_segments)
            if collision is not None:
                return True
        return False

    # Draws all track segments as lines
    # Colour alternates between black and red
    def draw_track(self):
        self.image.fill((0, 0, 0, 0))
        wall_points =[]
        wall_indices = self.grid.get_wall_segment_indices()
        for index in wall_indices:
            wall_points.extend(self.wall_segments[index])
        r.draw_alternating_line_segments(self.image, wall_points)
        game_core.game_mode.debug_elements["track spine"] = [self.track_spine]


class SpatialHashGrid:
    """
    Segments the set of wall segments into a hash set of cells, with each cell coordinate as the key, and a list of
    wall segments in that cell as the value. Also includes dda travelsal method and helper methods to hash and query
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
        if (ix, iy) in self.cells:
            for seg_index in self.cells[(ix, iy)]:
                seg = wall_segments[seg_index]
                hit_point = r.get_line_segments_intersection(ray, seg)
                if hit_point:
                    return hit_point
        return None

    def get_all_collision_indices(self, ix, iy, check_segment, points_list):
        collisions = []
        if (ix, iy) in self.cells:
            for seg_index in self.cells[(ix, iy)]:
                seg = (points_list[seg_index], points_list[seg_index+1])
                hit_point = r.get_line_segments_intersection(check_segment, seg)
                if hit_point:
                   collisions.append(seg_index)
        return collisions


    def dda_grid_traverse(self, segment, on_visit):
        (x1, y1), (x2, y2) = segment
        dx = x2 - x1
        dy = y2 - y1
        ix, iy = self.get_cell_index_of_point((x1, y1))
        ex, ey = self.get_cell_index_of_point((x2, y2))

        step_x = r.sign(dx)
        t_delta_x = (self.cell_size / abs(dx)) if dx != 0 else float('inf')
        next_boundary_x = (ix + 1) * self.cell_size if step_x == 1 else ix * self.cell_size
        t_max_x = (next_boundary_x - x1) / dx if dx != 0 else float("inf")

        step_y = r.sign(dy)
        t_delta_y = (self.cell_size / abs(dy)) if dy != 0 else float('inf')
        next_boundary_y = (iy + 1) * self.cell_size if step_y == 1 else iy * self.cell_size
        t_max_y = (next_boundary_y - y1) / dy if dy != 0 else float("inf")

        max_steps = (abs(ex - ix) + abs(ey - iy) + 10)
        for i in range(max_steps):
            result = on_visit(ix, iy)
            if result is not None:
                return result

            if ix == ex and iy == ey:
                break

            if t_max_x < t_max_y:
                ix += step_x
                t_max_x += t_delta_x
            else:
                iy += step_y
                t_max_y += t_delta_y

    def hash_segment(self, wall_segment, segment_index):
        self.dda_grid_traverse(wall_segment, on_visit=lambda ix, iy: self._add_segment_to_cell(ix, iy, segment_index))

    def return_collision_point(self, ray, wall_segments):
        return self.dda_grid_traverse(ray, on_visit=lambda ix, iy:
        self._check_cell_for_collision(ix, iy, ray, wall_segments))

    def return_all_collisions(self, check_segment, points_list):
        return self.dda_grid_traverse(check_segment, on_visit=lambda ix, iy:
        self.get_all_collision_indices(ix, iy, check_segment, points_list))

    def clear_grid(self):
        self.cells = defaultdict(list)

    def get_wall_segment_indices(self):
        cell_elements = self.cells.values()
        indices = []
        for element in cell_elements:
            indices.extend(element)
        return indices



# grid = SpatialHashGrid()
#
# walls = [
#     ((0, 0), (200, 0)),
#     ((0, 0), (0, 200)),
#     ((0, 0), (200, 200)),
#     ((230, 20), (120, 175)),
# ]
#
# for i, seg in enumerate(walls):
#     grid.hash_segment(seg, i)
#
# print()
# print("Testing ray intersection")
# print()
#
# print("Test 1: dx > dy")
# ray1 = ((100, 50), (250, 100))
# hit1 = grid.return_collision_point(ray1, walls)
# print("Collision point:", hit1)
# print()
#
# print("Test 2: dy > dx")
# ray2 = ((100, 20), (130, 250))
# hit2 = grid.return_collision_point(ray2, walls)
# print("Collision point:", hit2)
# print()
#
# print("Test 3: no collision")
# ray3 = ((250, 250), (400, 400))
# hit3 = grid.return_collision_point(ray3, walls)
# print(" Collision point:", hit3)

