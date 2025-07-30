import pygame
import resources
from car import Car
from user_interface import Slider
from user_interface import GaugeMeter
from user_interface import HorizontalMeter

pygame.init()
CLOCK = pygame.time.Clock()
FRAME_TIME = 1/60
PRESSED_KEYS = set()
MAIN_SCREEN = pygame.display.set_mode((1000, 1000))
SCREEN_FILL = "White"

car_mass = 800
throttle = 0
max_steer = 1.6
driving_force = 8400
starting_orientation = 180
steer_factor = max_steer/10
throttle_factor = 0.05
brake_factor = 0.08

car = Car(car_mass, (500, 500), starting_orientation, max_steer, driving_force, steer_factor,  FRAME_TIME)
steering_slider = Slider(250, 650, 500, 20, "Black", "#47cd4a", -1,1)
throttle_meter = HorizontalMeter(350, 150, 500, 20, "grey", "orange", -1, 1)
speedometer = GaugeMeter(200, 170, 100, 400, "dark gray", "Red")

GUI_elements = []
GUI_elements.extend(steering_slider.elements)
GUI_elements.extend(speedometer.elements)
GUI_elements.extend(throttle_meter.elements)
game_elements = [(car.car, car.car_rect)]

def check_events():
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            exit()
        if event.type == pygame.KEYDOWN:
            PRESSED_KEYS.add(event.key)
        if event.type == pygame.KEYUP:
            PRESSED_KEYS.remove(event.key)
        if event.type in (pygame.MOUSEBUTTONDOWN, pygame.MOUSEBUTTONUP, pygame.MOUSEMOTION):
            handle_mouse_events(event)

def handle_mouse_events(event):
    handle_steering(event)

def handle_steering(event):
    steering_slider.event_handle(event)

def handle_throttle():
    global throttle
    if pygame.K_w in PRESSED_KEYS:
        throttle = resources.clamp_value(throttle+ throttle_factor, -1, 1)
    elif pygame.K_s in PRESSED_KEYS:
        throttle = resources.clamp_value(throttle - brake_factor, -1, 1)
    else:
        throttle *= 0.96
        if -0.01 < throttle < 0.01:
            throttle = 0
    throttle_meter.update_value(throttle)
    speedometer.update_value(car.velocity.magnitude())

def car_movement():
    handle_throttle()
    car.move_car_sprite(steering_slider.value, throttle)
    game_elements.pop()
    game_elements.append((car.car,car.car_rect))

def update_frame():
    MAIN_SCREEN.fill(SCREEN_FILL)
    resources.draw(MAIN_SCREEN, game_elements)
    resources.draw(MAIN_SCREEN, GUI_elements)
    pygame.display.flip()
    CLOCK.tick(1/FRAME_TIME)

def game_loop():
    while True:
        check_events()
        car_movement()
        update_frame()

game_loop()

