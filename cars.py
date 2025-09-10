import math
import resources as r
from pygame.transform import rotate, scale
from pygame.sprite import Sprite
from pygame import Vector2

class Car(Sprite):
    """
    This class is the Car sprite and extends the Sprite class. The movement of the car is controlled by user input.
    The throttle and steer values are used to control the car's movement. The car's acceleration, velocity and position
    are calculated using vector math and arbitrary constants. Movement is not modelling in terms of forces in this
    version.In later versions, forces will be used to model car movement which would allow lesser bugs, more accurate
    steering and implementing drifting physics, however for present testing, this level of abstraction is sufficient
    """
    def __init__(self, starting_position = (0,0)):
        super().__init__()
        self.is_crashed = False
        self.position = Vector2(starting_position)
        self.velocity = Vector2(0,0)
        self.acceleration = Vector2(0,0)
        self.direction = r.starting_orientation

        self.throttle = 0
        self.steer = 0.0
        """
        hitbox -> Stores coordinates of 4 corners of hitbox
        sensors -> Stores distances to nearest wall for each sensor
        """
        self.hitbox = []
        self.sensors = [0,0,0,0,0,0,0,0,0,0,0,0,0]

        """
        The '_original_car' variable is necessary for rotating the sprite properly since the car sprite is rotated
        to an angle with respect to the natural orientation of the sprite on the screen.
        """
        self._original_car = r.set_image("Mclaren")
        self._original_car = scale(self._original_car, r.car_proportions)
        self._original_car = rotate(self._original_car, 180)

        self._update_car_sprite_position()

    # Calls procedures which handle the movement of the car sprite and update its parameters
    def car_movement(self, steer_percentage, throttle_pedal_amount):
        if not self.is_crashed:
            self._apply_steer(steer_percentage)
            self._apply_throttle_and_braking(throttle_pedal_amount)
            self._update_car_vector_values()
            self._update_car_sprite_position()

    # Increments the steer value.
    # Steer percentage is the value on the steer slider (determines the limit to which the steer can be incremented).
    # The steer_factor is the value by which steer is incremented (gradual changes rather than abrupt updates).
    # Direction of car updated.
    def _apply_steer(self, steer_percentage):
        steer_limit = r.max_steer * abs(steer_percentage) - r.steer_factor
        self.steer += r.steer_factor * r.sign(steer_percentage)
        self.steer = r.clamp_value(self.steer, -steer_limit, steer_limit)
        self.direction += self.steer
        self.direction = r.wrap_value(self.direction, 0, 360)

    # Sets the throttle value as a percentage of the maximum driving force.
    def _apply_throttle_and_braking(self, throttle_pedal_amount):
        if throttle_pedal_amount > 0:
            self.throttle = r.driving_force * throttle_pedal_amount
        else:
            self.throttle = r.braking_force * throttle_pedal_amount

    # Updates the acceleration, velocity and position vectors based on the throttle and direction of the car.
    # Velocity gradually decays when throttle = 0 (roughly emulates friction).
    # Grip variable and lerp function are used to smoothly update car position each frame (arbitrary values).
    """This method would be different when modelling forces."""
    def _update_car_vector_values(self):
        direction_unit_vector = Vector2(0, 1).rotate(self.direction)
        self.acceleration = self.throttle * direction_unit_vector
        if self.throttle == 0:
            self.velocity *= 0.996
        else:
            self.velocity += self.acceleration * r.FRAME_TIME
        if self.velocity.length_squared() != 0:
            self.velocity.clamp_magnitude_ip(r.max_speed)
        grip = 0.5
        speed = self.velocity.magnitude()
        self.velocity = self.velocity.lerp(direction_unit_vector * speed, grip)
        self.position += self.velocity * r.FRAME_TIME

    # Updates the coordinates of the corners of the hitbox w.r.t the position and orientation of the car
    def _update_hitbox(self):
        center_x, center_y = self.rect.center
        width, length = r.car_proportions / 2
        width -= 5
        corners = [(-width, length), (width, length), (width, -length), (-width, -length)]

        for i in range(4):
            x, y = corners[i]
            angle = math.radians(self.direction)
            corners[i] = (center_x + x * math.cos(angle) - y * math.sin(angle),
                          center_y + x * math.sin(angle) + y * math.cos(angle))
        self.hitbox[:] = corners

    # Changes the state of the car to indicate it has crashed and stop its movement
    def collision_detection(self, track):
        self._update_hitbox()
        self.is_crashed = track.hitbox_collision_detection(self.hitbox)

    # Calls ray cast method of track to get point of collision and normalised distance (w.r.t max ray length)
    # Stores distances as sensor data
    # Adds coordinates of start and end point of each ray to debugger
    def ray_cast(self, track):
        center = Vector2(self.rect.center)
        rays = []

        forward_ray = Vector2(0, 1).rotate(self.direction) * r.CAR_MAX_RAY_CAST
        rays.append((center, center + forward_ray))
        for angle in r.RAY_CAST_ANGLES:
            rays.append((center, center + forward_ray.rotate(angle)))
            rays.append((center, center + forward_ray.rotate(-angle)))

        collided_rays = []
        self.sensors.clear()
        for ray in rays:
            hit_point, normalised_collision_distance = track.ray_cast(ray)
            collided_rays.append((center, hit_point))
            self.sensors.append(normalised_collision_distance)
        r.DEBUG_ELEMENTS["rays"] = collided_rays
        print(self.sensors)

    # Updates the rotation of the car sprite with respect to its direction and the original orientation
    # Updates position (rect) of the car sprite
    def _update_car_sprite_position(self):
        self.image = rotate(self._original_car, -self.direction)
        self.rect = self.image.get_rect(center=(int(self.position.x), int(self.position.y)))

class PlayerCar(Car):
    def __init__(self, starting_position = None):
        super().__init__(starting_position)
        self.steering_slider = elements.UIHorizontalSlider(
            relative_rect=pygame.Rect((r.SCREEN_DIMENSIONS[0] / 2, 700), (600, 30)),
            start_value=0,
            value_range=(-100, 100),
            manager=r.GUI_MANAGER
        )


