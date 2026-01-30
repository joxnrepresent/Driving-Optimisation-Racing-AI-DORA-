save_data = {'a': 1, 'b':2}
actor_layer_sizes = [ 1,2,34,5,5]

print(save_data.get('actor_activations', ['elu'] * len(actor_layer_sizes) + ['linear']))


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
