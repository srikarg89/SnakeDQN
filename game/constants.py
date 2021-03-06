# Snake constants
SNAKE_INIT_LENGTH = 3
APPLE_TIMER = 60

# Directions
NORTH = (0,-1)
NORTHEAST = (1, -1)
EAST = (1,0)
SOUTHEAST = (1, 1)
SOUTH = (0,1)
SOUTHWEST = (-1, 1)
WEST = (-1,0)
NORTHWEST = (-1, -1)

# Display constants
BACKGROUND_COLOR = (255, 255, 255)
SNAKE_COLOR = (0, 0, 255)
APPLE_COLOR = (255, 0, 0)
WALL_COLOR = (0, 0, 0)
CLOCK_SPEED = 20

# Board constants
BOARD_WIDTH = 15
BOARD_HEIGHT = 15
BLOCK_SIZE = 15

# AI Rewards
LIFE_REWARD = 0
EAT_REWARD = 1
DEATH_REWARD = -1

# AI hyperparameters
LEARNING_RATE = 1e-3
GAMMA = 0.98
#HIDDEN = [200, 200]
HIDDEN = [50]
#HIDDEN = []
BATCH_SIZE = 32
MIN_EXPERIENCES = BATCH_SIZE
MAX_EXPERIENCES = 7500
MIN_EPSILON = 0.02
EPSILON_DECAY = 0.9999
TAU = 0.125
COPY_STEP = 25
SURVIVAL_REWARD = 0.1
