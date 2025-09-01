import resources as r
from math import floor
from pygame.sprite import Sprite
from pygame import Surface, Rect, SRCALPHA
from collections import defaultdict

class Track(Sprite):
    def __init__(self, track_name, position = ((0,0), r.SCREEN_DIMENSIONS)):
        super().__init__()
        self.grid = SpatialHashGrid()
        self.walls = r.load_track_from_file(track_name)
        self.wall_segments = []
        self.is_black = True
        for wall in self.walls:
            for i in range(len(wall) - 1):
                segment = (wall[i], wall[i + 1])
                self.wall_segments.append(segment)
                self.hash_track_segment(segment, self.wall_segments.index(segment))
        self.rect = Rect(position)
        self.image = Surface((self.rect.width, self.rect.height), SRCALPHA)
        self.draw_track()

    def hash_track_segment(self, segment, segment_index):
        self.grid.dda_traverse(segment, segment_index= segment_index)

    def collision_detection(self, ray):
        return self.grid.dda_traverse(ray, wall_segments= self.wall_segments)

    def draw_track(self):
        self.image.fill((0, 0, 0, 0))
        is_black = True
        for cell_segments in self.grid.cells.values():
            wall_segments = [self.wall_segments[i] for i in cell_segments]
            r.draw_alternating_line_segments(self.image, wall_segments, is_black)
            is_black = not is_black

class SpatialHashGrid:

    def __init__(self):
        self.cells = defaultdict(list)
        r.DEBUG_ELEMENTS["grid lines"].append(True)

    @staticmethod
    def _get_cell_index_of_point(point):
        x, y = point
        return int(floor(x / r.SHG_CELL_SIZE)), int(floor(y / r.SHG_CELL_SIZE))

    def dda_traverse(self, segment, segment_index = None, wall_segments = None):
        (x1, y1), (x2, y2) = segment
        dx = x2 - x1
        dy = y2 - y1
        ix, iy = self._get_cell_index_of_point((x1, y1))
        ex, ey = self._get_cell_index_of_point((x2, y2))

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
            elif wall_segments is not None:
                hit_point = self.lookup_nearby_segments(ix, iy, wall_segments, segment)
                if hit_point:
                    return hit_point
            else:
                return AttributeError

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

    def lookup_nearby_segments(self, ix, iy, wall_segments, ray):
        if (ix, iy) in self.cells:
            for seg_index in self.cells[(ix, iy)]:
                seg = wall_segments[seg_index]
                hit_point = r.get_line_segments_intersection(ray, seg)
                if hit_point:
                    return hit_point
        return None
