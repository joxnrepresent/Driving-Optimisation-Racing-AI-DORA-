# DORA: Driving Optimisation Racing AI

DORA is a 2D environment where you can train reinforcemenl learning models to race cars. Users can create custom tracks using track maker and train AI cars to drive them. Users can also race on tracks themselves and when an AI is sufficiently trained, you can race against it.

This project is my AQA A-level Computer Science NEA. The neural networks and reinforment learning training algorithms are written from scratch using only NumPy (No Tensorflow/Pytorch or other RL libraries). This incldes forward propagation, backpropagation, Adam optimiser, and the REINFORCE and actor-critic algorithms.

## What you can do

- **Make tracks.** Click to place points and the track curves between them. You can turn on Bézier handles to shape the corners yourself and change how wide the track is at different points. When you save a track it gets checked, so it won't let you save one with overlapping walls, a gap in it, or one that's too short.
- **Drive solo.** Pick a track and a number of laps, then drive. At the end you get your lap times.
- **Train an AI.** A batch of AI cars (12 by default) learn the track together, using either REINFORCE or Monte Carlo Actor-Critic. You can set the network size and pretty much every hyperparameter before you start, speed the simulation up to 10x, and save the model whenever you like.
- **Race the AI.** Load a model you trained and race it head to head.

## Getting it running

You'll need **Python 3.12 or newer** (the code uses `typing.override`, which was added in 3.12).

```bash
git clone https://github.com/joxnrepresent/Driving-Optimisation-Racing-AI-DORA-.git
cd Driving-Optimisation-Racing-AI-DORA-

python -m venv .venv
source .venv/bin/activate        # on Windows: .venv\Scripts\activate
pip install -r requirements.txt

python main.py
```

A few things to know:

- Run it from the repo root. Tracks, models, images and the UI theme are all loaded with relative paths, so it can't find them from anywhere else.
- The game opens full screen at your monitor's resolution. You quit with the **Exit** button on the main menu.
- `pygame_gui` needs **pygame-ce** (the community edition of pygame), not regular `pygame`. They both install as `import pygame` and clash if you have both, so if things look broken, run `pip uninstall pygame` and reinstall from `requirements.txt`.

## Controls

**Driving**

| Key | |
|---|---|
| W | accelerate |
| S | brake |
| A / D | steer left / right |

If you'd rather steer with a slider, there's an option for it in the pause menu.

**Track maker:** click to add a point, drag to move it, and press Backspace to delete it. Use **Anchor Mode** to edit the shape and **Width Mode** to change track width. Tracks save to `Tracks/` as JSON.

Every screen has a pause button (`||`) in the top left. The pause menu has the reset button, the debug overlay toggle (it shows hitboxes) and the way back to the menu.

## Training an AI

Go to **AI Training Mode**, pick a track, then choose REINFORCE or MCAC. You can then either load a model you've already saved to keep training it, or hit **Train New Model** to set one up from scratch. Last of all you pick how many laps each episode is.

An episode ends when every car has crashed, got stuck or finished, and the model updates at the end of each one. The panels on screen show how it's doing: average and best progress, lap times, and how many times it's been rolled back. **Adjust Speed** cycles the simulation through 1x, 2x, 5x and 10x, which helps a lot because training takes a while.

A couple of things happen in the background:

- **Rollback.** If the average over the last 10 episodes drops below 70% of the best average so far, the model gets reset to its best saved weights. Policy gradient training is unstable and this stops one bad run from wiping out hours of progress.
- **"Raceable" models.** A model only gets marked as ready to race once the cars average more than 75% of the race distance *and* their lap times are fast enough for that track. Only raceable models show up in **Race against AI**.

Models save to `Models/` as `.npy` files. They're loaded with `allow_pickle=True`.

## How it works

Each AI car sees **15 inputs**: 13 ray-cast distances to the walls, plus its current speed and steering angle. It outputs **2 values**, steering and throttle, each in the range -1 to 1. These are sampled from a Gaussian policy, so the network gives the mean and a learnable log-std controls how much the car explores.

The reward is made of 8 parts, all in `AICar.compute_reward` in `nea/cars.py`. In short, the car is rewarded for making progress and going fast, for steering into corners early and for staying away from walls. It's penalised for getting stuck or never steering. It also gets a small bonus each step for staying alive and a big one for finishing a lap.

Collision checks and ray casts go through a spatial hash grid (`nea/track.py`), so the game doesn't have to test every ray against every wall segment each frame.

### Code layout

```
main.py                    entry point and game loop
DORA_theme.json            pygame_gui theme
nea/
  resources.py             GameCore (screen, input, current mode), lap timing, geometry/spline helpers
  game_modes.py            main menu, track maker, solo, vs AI, training
  cars.py                  car physics, player car, AI car (sensors and rewards)
  track.py                 track rendering, collisions, spatial hash grid
  neural_network.py        NumPy MLP with Adam
  rl_model.py              REINFORCE and MCAC
  gui_custom_elements.py   custom widgets (track canvas, file browser, model setup screen, etc.)
Images/                    car sprites and menu background
Tracks/                    saved tracks (JSON)
Models/                    saved models (.npy)
tests/                     test scripts
```

## Tests

These are plain scripts rather than a pytest suite. They print PASS or FAIL as they go, and some of them plot graphs. Install the extra dependency (matplotlib), then run them from the repo root:

```bash
pip install -r requirements-dev.txt

python -m tests.white_box_tests       # track geometry, car physics, layers, RL returns/actions
python -m tests.nn_tests              # network shape, gradient flow, batching, activations, weight init
python -m tests.nn_function_fitting   # trains small networks on wavy functions and saves the plots
```
