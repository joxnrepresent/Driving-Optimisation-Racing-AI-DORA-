from pygame import  Vector2
segment = ((10, 20), (20, 40))
normal = (Vector2(segment[0]) - Vector2(segment[1])).normalize()
print(normal)
