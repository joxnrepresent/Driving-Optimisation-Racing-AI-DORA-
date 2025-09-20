import resources as r
from math import floor, hypot
from pygame.sprite import Sprite
from pygame import Surface, Rect, SRCALPHA, Vector2
from collections import defaultdict

class Track(Sprite):
    """
    Creates a track sprite. The track is composed of individual wall segments which are represented as list of start
    and end points, and has a separate image that is loaded onto the screen. The grid is a hash map that maps the cells
    of the grid to the track segments in that cell.
    """
    def __init__(self, track_name, position = ((0,0), r.SCREEN_DIMENSIONS)):
        super().__init__()
        self.grid = SpatialHashGrid()
        track_data = r.load_track_from_file(track_name)
        track_spine = track_data.pop(0)
        self.track_spine = [Vector2(point) for point in track_spine]
        walls = track_data
        self.wall_segments = []
        segment_index = 0
        for wall in walls:
            for i in range(len(wall)):
                segment = (wall[i], wall[(i + 1) % len(wall)])
                self.wall_segments.append(segment)
                self.grid.hash_segment(segment, segment_index)
                segment_index += 1

        self.rect = Rect(position)
        self.image = Surface((self.rect.width, self.rect.height), SRCALPHA)
        self.draw_track()

    # Cast ray and return shortened ray up till nearest point of intersection
    # Returns normalised distance which is the percentage of max length
    def ray_cast(self, ray):
        hit_point = Vector2(self.grid.return_collision_point(ray, self.wall_segments))
        normalised_distance = ((hit_point - ray[0]).length()/r.CAR_MAX_RAY_CAST) ** 0.4
        return hit_point, normalised_distance

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
        is_black = True
        for cell_segments in self.grid.cells.values():
            wall_segments = [self.wall_segments[i] for i in cell_segments]
            r.draw_alternating_line_segments(self.image, wall_segments, is_black)
            is_black = not is_black
        r.DEBUG_ELEMENTS["track spine"] = [self.track_spine]


class SpatialHashGrid:
    """
    Segments the set of wall segments into a hash set of cells, with each cell coordinate as the key, and a list of
    wall segments in that cell as the value. Also includes dda travelsal method and helper methods to hash and query
    cells.
    """
    def __init__(self):
        self.cells = defaultdict(list)
        r.DEBUG_ELEMENTS["grid lines"].append(True)

    @staticmethod
    def get_cell_index_of_point(point):
        x, y = point
        return int(floor(x / r.SHG_CELL_SIZE)), int(floor(y / r.SHG_CELL_SIZE))

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

    def dda_grid_traverse(self, segment, on_visit):
        (x1, y1), (x2, y2) = segment
        dx = x2 - x1
        dy = y2 - y1
        ix, iy = self.get_cell_index_of_point((x1, y1))
        ex, ey = self.get_cell_index_of_point((x2, y2))

        step_x = r.sign(dx)
        t_delta_x = (r.SHG_CELL_SIZE / abs(dx)) if dx != 0 else float('inf')
        next_boundary_x = (ix + 1) * r.SHG_CELL_SIZE if step_x == 1 else ix * r.SHG_CELL_SIZE
        t_max_x = (next_boundary_x - x1) / dx if dx != 0 else float("inf")

        step_y = r.sign(dy)
        t_delta_y = (r.SHG_CELL_SIZE / abs(dy)) if dy != 0 else float('inf')
        next_boundary_y = (iy + 1) * r.SHG_CELL_SIZE if step_y == 1 else iy * r.SHG_CELL_SIZE
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

    def return_collision_point(self, line, wall_segments):
        return self.dda_grid_traverse(line, on_visit=lambda ix, iy:
                                                    self._check_cell_for_collision(ix, iy, line, wall_segments))

