import numpy as np
from game.helpers import add, sub, rotate_clockwise, rotate_counter_clockwise, angle

class BoardSpace:

    STATE_SIZE = 1200
    ACTION_SIZE = 4

    @staticmethod
    def get_state(snake, env):
        arr1, arr2, arr3 = [], [], []
        for i in range(snake.board_height):
            for j in range(snake.board_width):
                a = 1 if (i,j) == env.apple else 0
                b = 1 if (i,j) in  snake.body else 0
                h = 1 if (i,j) == snake.head else 0
                arr1.append(h)
                arr2.append(b)
                arr3.append(a)
        return np.array(arr1 + arr2 + arr3)


    @staticmethod
    def get_action(snake, state):
        # Calculate stuff
        idx = snake.train_brain.get_action(state, snake.epsilon)
        # Save variables to add to experience replay later
        return idx

    @staticmethod
    def interpret(snake, idx):
        return snake.directions[idx]


class FPVSpace:

    STATE_SIZE = 25
    ACTION_SIZE = 3

    @staticmethod
    def get_state(snake, env):
        head = snake.head
        arr = []
        block_groups = [snake.body, [env.apple], env.wall]
        for xd, group in enumerate(block_groups):
            dir = snake.direction
            temp = []
            for i in range(8):
                val = 0.0
                for b in group:
                    d = sub(b, head)
                    if d[0] == 0 and d[1] == 0:
                        val = 1
                        break
#                    if xd == 0 and i == 3:
#                        print(d)
                    if angle(dir) != angle(d):
                        continue
                    val = abs(1.0 / d[0] if d[0] != 0 else 1.0 / d[1])
                    break
                dir = rotate_clockwise(dir)
                temp.append(val)
            arr.extend(temp)
        arr.append(env.timer / env.timer_threshold)
#        print("ARRAY")
#        print(arr)
        return np.array(arr)


    @staticmethod
    def get_action(snake, state):
        # Calculate stuff
        idx = snake.brain.get_action(state)
        # Save variables to add to experience replay later
        return idx
    
    @staticmethod
    def interpret(snake, idx):
        if idx == 0:
            return rotate_clockwise(rotate_clockwise(snake.direction))
        if idx == 1:
            return snake.direction
        if idx == 2:
            return rotate_counter_clockwise(rotate_counter_clockwise(snake.direction))
