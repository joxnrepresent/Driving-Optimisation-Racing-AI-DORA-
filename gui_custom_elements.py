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







#
# class UIBezierCanvas(UIElement):
#     def __init__(self, relative_rect, manager, container=None, enforce_g1=True):
#         super().__init__(
#             relative_rect=relative_rect,
#             manager=manager,
#             container=container,
#             starting_height=0,
#             layer_thickness=1,
#             object_id="#bezier_canvas"
#         )
#
#         # Data
#         self.anchors = []   # list[Vector2]  (anchor points in local coords)
#         self.segments = []  # list[dict] with keys: start, c1, c2, end, overridden_c1, overridden_c2
#
#         # Drawing surface sized to the element's relative_rect
#         self.image = pygame.Surface((self.relative_rect.width, self.relative_rect.height), pygame.SRCALPHA)
#
#         # Interaction state
#         self.dragging = None  # None or dict: {'type':'anchor'/'handle', ...}
#         self.anchor_hit_radius = 8
#         self.handle_hit_radius = 8
#         self.enforce_g1 = enforce_g1
#
#         # Appearance
#         self.bg_colour = pygame.Color(40, 40, 40)
#         self.border_colour = pygame.Color(200, 200, 200)
#         self.curve_colour = pygame.Color(220, 80, 40)
#         self.anchor_colour = pygame.Color(40, 200, 80)
#         self.handle_colour = pygame.Color(200, 120, 40)
#         self.helper_colour = pygame.Color(150, 150, 150)
#         self.anchor_radius = 5
#         self.handle_radius = 4
#         self.curve_width = 3
#
#         # initial image
#         self.rebuild()
#
#     # -------------------------
#     # Input handling
#     # -------------------------
#     def process_event(self, event):
#         # Convert global mouse pos to local canvas coords helper
#         def local_pos_from_event(evpos):
#             return Vector2(evpos) - Vector2(self.rect.topleft)
#
#         # Left click down -> either start drag or add anchor
#         if event.type == pygame.MOUSEBUTTONDOWN and event.button == 1:
#             if not self.rect.collidepoint(event.pos):
#                 return False
#
#             local = local_pos_from_event(event.pos)
#
#             # 1) Hit test anchors (closest)
#             for i, a in enumerate(self.anchors):
#                 if (a - local).length() <= self.anchor_hit_radius:
#                     # start dragging anchor i
#                     offset = a - local
#                     self.dragging = {'type': 'anchor', 'index': i, 'offset': offset}
#                     return True
#
#             # 2) Hit test handles (c1/c2) per segment
#             for si, seg in enumerate(self.segments):
#                 if (seg['c1'] - local).length() <= self.handle_hit_radius:
#                     offset = seg['c1'] - local
#                     seg['overridden_c1'] = True  # mark manual change
#                     self.dragging = {'type': 'handle', 'seg_index': si, 'which': 'c1', 'offset': offset}
#                     return True
#                 if (seg['c2'] - local).length() <= self.handle_hit_radius:
#                     offset = seg['c2'] - local
#                     seg['overridden_c2'] = True
#                     self.dragging = {'type': 'handle', 'seg_index': si, 'which': 'c2', 'offset': offset}
#                     return True
#
#             # 3) No hit -> add a new anchor (append)
#             self.anchors.append(Vector2(local))
#             self._generate_segments_from_anchors(preserve_overrides=True)
#             self.rebuild()
#             return True
#
#         # Left button up -> stop drag
#         if event.type == pygame.MOUSEBUTTONUP and event.button == 1:
#             if self.dragging:
#                 # when finishing dragging an anchor, recompute default handles
#                 if self.dragging['type'] == 'anchor':
#                     # anchor moved; regenerate default handles where not overridden
#                     self._generate_segments_from_anchors(preserve_overrides=True)
#                 # end drag
#                 self.dragging = None
#                 self.rebuild()
#                 return True
#
#         # Mouse motion -> if dragging, move the item
#         if event.type == pygame.MOUSEMOTION:
#             if self.dragging:
#                 local = Vector2(event.pos) - Vector2(self.rect.topleft)
#                 d = self.dragging
#                 if d['type'] == 'anchor':
#                     idx = d['index']
#                     new_pos = local + d['offset']
#                     # clamp to canvas bounds
#                     new_pos.x = max(0, min(self.relative_rect.width, new_pos.x))
#                     new_pos.y = max(0, min(self.relative_rect.height, new_pos.y))
#                     self.anchors[idx] = new_pos
#                     # update segment endpoints linked to this anchor
#                     self._update_segment_endpoints_from_anchors()
#                     # regenerate handles only for non-overridden segments
#                     self._generate_segments_from_anchors(preserve_overrides=True)
#                     self.rebuild()
#                     return True
#
#                 elif d['type'] == 'handle':
#                     si = d['seg_index']
#                     which = d['which']
#                     seg = self.segments[si]
#                     new_pos = local + d['offset']
#                     # clamp
#                     new_pos.x = max(0, min(self.relative_rect.width, new_pos.x))
#                     new_pos.y = max(0, min(self.relative_rect.height, new_pos.y))
#                     seg[which] = new_pos
#                     seg[f'overridden_{which}'] = True
#
#                     # If G1 enforcement on, update neighbouring segment's matching handle direction
#                     if self.enforce_g1:
#                         self._enforce_g1_after_handle_move(seg_index=si, which=which)
#                     self.rebuild()
#                     return True
#
#         # Right click -> undo last anchor
#         if event.type == pygame.MOUSEBUTTONDOWN and event.button == 3:
#             if self.rect.collidepoint(event.pos):
#                 if self.anchors:
#                     self.anchors.pop()
#                     self._generate_segments_from_anchors(preserve_overrides=True)
#                     self.rebuild()
#                 return True
#
#         # Keyboard: clear with 'c' or 'C'
#         if event.type == pygame.KEYDOWN:
#             if event.key == pygame.K_c:
#                 self.anchors.clear()
#                 self.segments.clear()
#                 self.rebuild()
#                 return True
#
#         return False
#
#     # -------------------------
#     # Segment generation & updates
#     # -------------------------
#     def _generate_segments_from_anchors(self, preserve_overrides=True):
#         """
#         Build self.segments from anchors using Catmull-Rom -> cubic Bézier conversion.
#         If preserve_overrides is True, keep any previously overridden control points for matching indices.
#         """
#         old = self.segments
#         n = len(self.anchors)
#         new_segs = []
#
#         for i in range(n - 1):
#             p1 = self.anchors[i]        # start
#             p2 = self.anchors[i + 1]    # end
#             p0 = self.anchors[i - 1] if i - 1 >= 0 else p1  # clamp
#             p3 = self.anchors[i + 2] if i + 2 < n else p2  # clamp
#
#             # default handles (Catmull-Rom -> Bezier)
#             default_c1 = p1 + (p2 - p0) / 6.0
#             default_c2 = p2 - (p3 - p1) / 6.0
#
#             seg = {
#                 'start': Vector2(p1),
#                 'c1': Vector2(default_c1),
#                 'c2': Vector2(default_c2),
#                 'end': Vector2(p2),
#                 'overridden_c1': False,
#                 'overridden_c2': False
#             }
#
#             # preserve old overrides/positions for the same segment index
#             if preserve_overrides and i < len(old):
#                 oldseg = old[i]
#                 if oldseg.get('overridden_c1', False):
#                     seg['c1'] = Vector2(oldseg['c1'])
#                     seg['overridden_c1'] = True
#                 if oldseg.get('overridden_c2', False):
#                     seg['c2'] = Vector2(oldseg['c2'])
#                     seg['overridden_c2'] = True
#
#             new_segs.append(seg)
#
#         self.segments = new_segs
#
#     def _update_segment_endpoints_from_anchors(self):
#         """When anchors move, update segment start/end positions to match anchors."""
#         for i, seg in enumerate(self.segments):
#             if i < len(self.anchors):
#                 seg['start'] = Vector2(self.anchors[i])
#             if i + 1 < len(self.anchors):
#                 seg['end'] = Vector2(self.anchors[i + 1])
#
#     def _enforce_g1_after_handle_move(self, seg_index, which):
#         """
#         Keep tangent direction continuous at joins:
#         - if which == 'c2', update next segment's c1 to be collinear (opposite direction).
#         - if which == 'c1', update previous segment's c2 similarly.
#         We try to preserve the neighbor's handle length.
#         """
#         seg = self.segments[seg_index]
#         if which == 'c2':
#             # Join at seg.end which is anchor between seg and next_seg
#             joint_anchor = seg['end']
#             v = seg['c2'] - joint_anchor
#             if v.length_squared() == 0:
#                 return
#             # update next seg c1
#             next_index = seg_index + 1
#             if next_index < len(self.segments):
#                 next_seg = self.segments[next_index]
#                 old_vec = next_seg['c1'] - joint_anchor
#                 old_len = old_vec.length()
#                 if old_len == 0:
#                     old_len = v.length()
#                 new_c1 = joint_anchor - v.normalize() * old_len
#                 next_seg['c1'] = new_c1
#                 next_seg['overridden_c1'] = True
#
#         elif which == 'c1':
#             # Join at seg.start; affect previous segment's c2
#             joint_anchor = seg['start']
#             v = seg['c1'] - joint_anchor
#             if v.length_squared() == 0:
#                 return
#             prev_index = seg_index - 1
#             if prev_index >= 0:
#                 prev_seg = self.segments[prev_index]
#                 old_vec = prev_seg['c2'] - joint_anchor
#                 old_len = old_vec.length()
#                 if old_len == 0:
#                     old_len = v.length()
#                 new_c2 = joint_anchor - v.normalize() * old_len
#                 prev_seg['c2'] = new_c2
#                 prev_seg['overridden_c2'] = True
#
#     # -------------------------
#     # Drawing / rebuild
#     # -------------------------
#     def rebuild(self):
#         """Redraw the committed image (cached)."""
#         # clear
#         self.image.fill((0, 0, 0, 0))
#         # background
#         pygame.draw.rect(self.image, self.bg_colour, pygame.Rect(0, 0, self.relative_rect.width, self.relative_rect.height))
#
#         # draw segments
#         for seg in self.segments:
#             self._draw_cubic_bezier_on_surf(self.image, seg['start'], seg['c1'], seg['c2'], seg['end'], self.curve_colour, self.curve_width)
#
#         # draw helper lines & handles
#         for seg in self.segments:
#             # helper lines from start->c1 and end->c2
#             pygame.draw.line(self.image, self.helper_colour, seg['start'], seg['c1'], 1)
#             pygame.draw.line(self.image, self.helper_colour, seg['end'], seg['c2'], 1)
#             # handles
#             pygame.draw.circle(self.image, self.handle_colour, (int(seg['c1'].x), int(seg['c1'].y)), self.handle_radius)
#             pygame.draw.circle(self.image, self.handle_colour, (int(seg['c2'].x), int(seg['c2'].y)), self.handle_radius)
#
#         # draw anchors on top
#         for a in self.anchors:
#             pygame.draw.circle(self.image, self.anchor_colour, (int(a.x), int(a.y)), self.anchor_radius)
#
#     def _draw_cubic_bezier_on_surf(self, surf, p0, c1, c2, p3, colour, width):
#         """Evaluate cubic with De Casteljau (lerp) and draw connected short lines."""
#         pts = []
#         steps = 48
#         for i in range(steps + 1):
#             t = i / steps
#             a = p0.lerp(c1, t)
#             b = c1.lerp(c2, t)
#             c = c2.lerp(p3, t)
#             d = a.lerp(b, t)
#             e = b.lerp(c, t)
#             pt = d.lerp(e, t)
#             pts.append((int(pt.x), int(pt.y)))
#         if len(pts) >= 2:
#             pygame.draw.lines(surf, colour, False, pts, width)
#
#     # -------------------------
#     # Main draw called each frame
#     # -------------------------
#     def draw(self, surface):
#         # blit cached image
#         surface.blit(self.image, self.rect.topleft)
#         # border
#         pygame.draw.rect(surface, self.border_colour, self.rect, width=1)
#
#         # live preview: if mouse inside and there is at least 1 anchor, preview next segment to mouse
#         mx, my = pygame.mouse.get_pos()
#         if self.rect.collidepoint((mx, my)) and self.anchors:
#             local_mouse = Vector2((mx - self.rect.x, my - self.rect.y))
#             last_anchor = self.anchors[-1]
#
#             keys = pygame.key.get_pressed()
#             space = keys[pygame.K_SPACE]
#
#             if space:
#                 # straight preview
#                 start = Vector2(self.rect.x + last_anchor.x, self.rect.y + last_anchor.y)
#                 end = Vector2(self.rect.x + local_mouse.x, self.rect.y + local_mouse.y)
#                 pygame.draw.line(surface, self.curve_colour, start, end, self.curve_width)
#             else:
#                 # compute preview cubic from last anchor -> mouse (use catmull style with clamped neighbours)
#                 p1 = last_anchor
#                 p2 = local_mouse
#                 p0 = self.anchors[-2] if len(self.anchors) >= 2 else p1
#                 p3 = p2
#                 c1 = p1 + (p2 - p0) / 6.0
#                 c2 = p2 - (p3 - p1) / 6.0
#                 # evaluate and draw
#                 preview_pts = []
#                 steps = 30
#                 for i in range(steps + 1):
#                     t = i / steps
#                     a = p1.lerp(c1, t)
#                     b = c1.lerp(c2, t)
#                     c = c2.lerp(p2, t)
#                     d = a.lerp(b, t)
#                     e = b.lerp(c, t)
#                     pt = d.lerp(e, t)
#                     preview_pts.append((int(self.rect.x + pt.x), int(self.rect.y + pt.y)))
#                 if len(preview_pts) >= 2:
#                     pygame.draw.lines(surface, self.curve_colour, False, preview_pts, self.curve_width)
#
#         # small on-canvas instructions
#         font = pygame.font.SysFont(None, 16)
#         info = "LClick: add • Drag anchors/handles • RClick: undo • C: clear • Hold SPACE: straight"
#         text_surf = font.render(info, True, (200, 200, 200))
#         surface.blit(text_surf, (self.rect.x + 6, self.rect.y + 6))
