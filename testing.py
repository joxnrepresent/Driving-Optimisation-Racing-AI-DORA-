EPS = 1e-12 # Very small number to avoid division by zero
def get_line_segments_intersection(seg1, seg2):
    (x1, y1), (x2, y2) = seg1
    (x3, y3), (x4, y4) = seg2

    # Early reject intersection if AABB's do not intersect
    if (max(x1, x2) < min(x3, x4) or max(x3, x4) < min(x1, x2) or
            max(y1, y2) < min(y3, y4) or max(y3, y4) < min(y1, y2)):
        return None

    # Intersection parameters
    determinant = (x2 - x1) * (y4 - y3) - (y2 - y1) * (x4 - x3)
    alpha_numerator = (x3 - x1) * (y4 - y3) - (y3 - y1) * (x4 - x3)
    beta_numerator = (x3 - x1) * (y2 - y1) - (y3 - y1) * (x2 - x1)

    if abs(determinant) < EPS: # Parallel lines

        if abs(alpha_numerator) < EPS and abs(beta_numerator) < EPS: # Checking collinear

            # dot_prod checks for direction
            dot_prod = (x2 - x1) * (x4 - x3) + (y2 - y1) * (y4 - y3)

            # Overlapping regions
            overlap_start_x = max(min(x1, x2), min(x3, x4))
            overlap_end_x = min(max(x1, x2), max(x3, x4))
            overlap_start_y = max(min(y1, y2), min(y3, y4))
            overlap_end_y = min(max(y1, y2), max(y3, y4))

            # Checking if they overlap
            if overlap_start_x <= overlap_end_x and overlap_start_y <= overlap_end_y:
                # Case 1: seg1 first endpoint lies on seg2
                if min(x3, x4) <= x1 <= max(x3, x4) and min(y3, y4) <= y1 <= max(y3, y4):
                    return x1, y1

                p1 = (overlap_start_x, overlap_start_y)
                p2 = (overlap_end_x, overlap_end_y)
                if dot_prod > 0:
                    # Case 2: Return Overlap start point
                    return p1
                else:
                    # Case 3: Return Overlap end point
                    return p2
        # Case 4: Parallel and not collinear lines
        return None

    else: # Non-Parallel lines
        alpha = alpha_numerator / determinant
        beta = beta_numerator / determinant

        if alpha >= 0 and 0 <= beta <= 1:

            x = x1 + alpha * (x2 - x1)
            y = y1 + alpha * (y2 - y1)
            return x, y
        else:
            return None

seg1 = ((5,3), (6,9))
seg2 = ((4,2), (9,5))
intersection_point = get_line_segments_intersection(seg1, seg2)
print("Line segment 1: " + str(seg1))
print("Line segment 2: " + str(seg2))
print("Point of intersection: " + str(intersection_point))