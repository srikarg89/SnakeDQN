# Snake constants
SNAKE_INIT_LENGTH = 3

# Directions
NORTH = (0,-1)
SOUTH = (0,1)
EAST = (1,0)
WEST = (-1,0)

# Display constants
BACKGROUND_COLOR = (255, 255, 255)
SNAKE_COLOR = (0, 0, 255)
APPLE_COLOR = (255, 0, 0)
WALL_COLOR = (0, 0, 0)
CLOCK_SPEED = 12

# Board constants
BOARD_WIDTH = 20
BOARD_HEIGHT = 20
BLOCK_SIZE = 15

# AI hyperparameters
LEARNING_RATE = 1e-2
GAMMA = 0.99
HIDDEN = [400, 400]
MIN_EXPERIENCES = 750
MAX_EXPERIENCES = 7500
BATCH_SIZE = 64
MIN_EPSILON = 0.10
EPSILON_DECAY = 0.99
COPY_STEP = 25