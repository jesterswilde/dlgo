import numpy as np
import pandas as pd
import queue
from enum import IntEnum
import random
from time import sleep
import os


class Points(IntEnum):
    EMPTY = 0
    WALL = -1
    SNAKE = 1
    HEAD = 2
    FOOD = 3


class Facing(IntEnum):
    UP = 0
    RIGHT = 1
    DOWN = 2
    LEFT = 3


class Turn(IntEnum):
    FORWARD = 0
    RIGHT = 1
    LEFT = 2


FILENAME = './snake_agent_slim.pkl'
# EYE_SIGHT = 2
SAVE_EVERY = 50
PRINT_EVERY = 1
EPOCHS = 1000000
MAX_HISTORY = 10
SHOULD_SLEEP = True
RAND_MOVE = 0
SLEEP_AMOUNT = 0.1


class Agent:
    learning_amount = 0.01
    discount = 0.90

    def __init__(self):
        if os.path.isfile(FILENAME):
            self.q_table = pd.read_pickle(FILENAME)
        else:
            self.q_table = pd.DataFrame(columns=[0, 1, 2])
        # self.q_table = pd.DataFrame(columns=[0, 1, 2, 3])
        self.last_state = None
        self.last_action = None

    def check_value(self, state):
        if state not in self.q_table.index:
            self.q_table = self.q_table.append(pd.Series([
                random.uniform(0, 1),
                random.uniform(0, 1),
                random.uniform(0, 1),
            ], name=state))

    def pick_random(self, state):
        action = random.randint(0, 2)
        self.last_state = state
        self.last_action = action
        return action

    def get_score_at(self, state, action):
        return self.q_table.loc[state, action]

    def pick_action(self, state):
        self.check_value(state)
        action = self.q_table.loc[state, :].idxmax()
        # print(self.q_table.loc[state])
        # action = self.get_max(self.q_table.loc[state, :], facing)
        self.last_state = state
        self.last_action = action
        return action

    def print_last(self):
        print(self.q_table.loc[self.last_state, self.last_action])
        print(self.q_table.loc[self.last_state])

    def break_loop(self):
        new_reward = self.q_table.loc[self.last_state, self.last_action] - 0.1
        self.q_table.loc[self.last_state, self.last_action] = new_reward

    def learn(self, reward, new_state):
        self.check_value(new_state)
        if self.last_state is not None:
            new_q = self.q_table.loc[new_state].max()
            old_q = self.q_table.loc[self.last_state, self.last_action]
            self.q_table.loc[self.last_state, self.last_action] = old_q * (
                1 - self.learning_amount) + self.learning_amount * (reward + self.discount * new_q)


class Game:
    def __init__(self, board_size=20):
        self.board_size = board_size
        self.board = Board(board_size)
        self.agent = Agent()

    def reset(self):
        self.board = Board(self.board_size)

    def play_game(self):
        # while self.board.game_is_going:
        #     self.board.take_action(Facing.RIGHT)
        #     self.board.print_board()
        #     sleep(0.3)

        record = []
        for game in range(EPOCHS):
            count = 0
            while self.board.game_is_going:
                printing = (game % PRINT_EVERY) == 0
                count += 1
                board = self.encode_board()
                if random.uniform(0, 1) < RAND_MOVE:
                    move = self.agent.pick_random(board)
                else:
                    move = self.agent.pick_action(board)
                reward = self.board.take_action(move)
                state = self.encode_board()
                # cur_reward = self.agent.get_score_at(board, move)
                # if reward == -1 and cur_reward < -0.95:
                #     reward = cur_reward - 0.2
                self.agent.learn(reward=reward, new_state=self.encode_board())
                if printing:
                    # self.agent.print_last()
                    self.board.print_board()
                    print("Game #: {}".format(game))
                    if SHOULD_SLEEP:
                        sleep(SLEEP_AMOUNT)
            record.append(self.board.fruit_eaten)
            if len(record) == 25:
                total = sum(record)
                print("Average: {} Max:{} Min:{} for game {} - {}".format(total /
                                                                          len(record), max(record), min(record), game - 25, game))
                record = []
            self.reset()
            if (game % SAVE_EVERY) == 0 and game != 0:
                self.agent.q_table.to_pickle(FILENAME)
        # print(self.agent.q_table)

    def encode_board_small(self):
        return self.encode_board(size=1)

    def get_val_in_board(self, pos):
        row, col = pos
        if row < 0 or row >= self.board_size or col < 0 or col >= self.board_size:
            return Points.WALL
        return self.board.board[row][col]

    def encode_board(self, size=0):
        head_row, head_col = self.board.head
        # self.board
        encoded_board = []
        forward_dir, right_dir, left_dir = self.board.get_dir_from_facing(
            self.board.facing)
        forward_pos = (head_row + forward_dir[0], head_col + forward_dir[1])
        right_pos = (head_row + right_dir[0], head_col + right_dir[1])
        left_pos = (head_row + left_dir[0], head_col + left_dir[1])
        fr_pos = (forward_pos[0] + right_dir[0], forward_pos[1] + right_dir[1])
        fl_pos = (forward_pos[0] + left_dir[0], forward_pos[1] + left_dir[1])

        forward_val = self.get_val_in_board(forward_pos)
        right_val = self.get_val_in_board(right_pos)
        left_val = self.get_val_in_board(left_pos)
        # fr_val = self.get_val_in_board(fr_pos)
        # fl_val = self.get_val_in_board(fl_pos)

        forward_val = 1 if forward_val == Points.EMPTY else 0
        right_val = 1 if right_val == Points.EMPTY else 0
        left_val = 1 if left_val == Points.EMPTY else 0
        # fr_val = 1 if fr_val == Points.EMPTY else 0
        # fl_val = 1 if fl_val == Points.EMPTY else 0
        food_row, food_col = self.board.food_pos
        if head_row < food_row:
            right_diff = -1
        elif head_row > food_row:
            right_diff = 1
        else:
            right_diff = 0
        if head_col < food_col:
            forward_diff = -1
        elif head_col > food_col:
            forward_diff = 1
        else:
            forward_diff = 0

        # if self.board.facing == Facing.RIGHT:
        #     right_diff, forward_diff = forward_diff * -1, right_diff
        # elif self.board.facing == Facing.DOWN:
        #     right_diff, forward_diff = right_diff * -1, forward_diff * -1
        # elif self.board.facing == Facing.LEFT:
        #     right_diff, forward_diff = forward_diff, right_diff * -1
        # print(str(([forward_val, left_val, right_val], forward_diff, right_diff)))
        return str(([forward_val, left_val, right_val], forward_diff, right_diff, int(self.board.facing)))


class Board:
    def __init__(self, size=20):
        self.board_size = size
        self.board = []
        self.fruit_eaten = 0
        for i in range(size):
            self.board.append([Points.EMPTY]*size)
        self.snake = queue.Queue()
        self.game_is_going = True
        row, col = size//2, size//2
        self.snake.put((row, col))
        self.board[row][col] = Points.HEAD
        self.head = (row, col)
        self.place_food()
        # self.board[5][7] = Points.FOOD
        self.facing = Facing.RIGHT

    def get_dir_from_facing(self, facing):
        if facing == Facing.UP:
            return (1, 0), (0, 1), (0, -1)
        if facing == Facing.RIGHT:
            return (0, 1), (-1, 0), (1, 0)
        if facing == Facing.DOWN:
            return (-1, 0), (0, -1), (0, 1)
        return (0, -1), (1, 0), (-1, 0)

    def place_food(self):
        valid_moves = []
        for row in range(self.board_size):
            for col in range(self.board_size):
                if self.board[row][col] is Points.EMPTY:
                    valid_moves.append((row, col))
        food_row, food_col = random.choice(valid_moves)
        self.board[food_row][food_col] = Points.FOOD
        self.food_pos = (food_row, food_col)

    def take_action(self, action: Turn):
        old_facing = self.facing
        if action == Turn.RIGHT:
            if self.facing == Facing.UP:
                self.facing = Facing.RIGHT
            elif self.facing == Facing.RIGHT:
                self.facing = Facing.DOWN
            elif self.facing == Facing.DOWN:
                self.facing = Facing.LEFT
            else:
                self.facing = Facing.UP
        elif action == Turn.LEFT:
            if self.facing == Facing.UP:
                self.facing = Facing.LEFT
            elif self.facing == Facing.LEFT:
                self.facing = Facing.DOWN
            elif self.facing == Facing.DOWN:
                self.facing = Facing.RIGHT
            else:
                self.facing = Facing.UP
        cur_head_row, cur_head_col = self.head
        forward, _, _ = self.get_dir_from_facing(self.facing)
        next_head = (cur_head_row + forward[0], cur_head_col + forward[1])
        if self.hit_wall(next_head) or self.hit_self(next_head):
            self.game_is_going = False
            return -1
        ate_food = self.ate_fruit(next_head)
        self.board[cur_head_row][cur_head_col] = Points.SNAKE
        self.board[next_head[0]][next_head[1]] = Points.HEAD
        self.snake.put(next_head)
        self.head = next_head
        if ate_food:
            self.place_food()
            self.fruit_eaten += 1
            return 1
        else:
            old_row, old_col = self.snake.get()
            self.board[old_row][old_col] = Points.EMPTY
            return 0

    def ate_fruit(self, pos):
        row, col = pos
        return self.board[row][col] == Points.FOOD

    def hit_self(self, pos):
        row, col = pos
        if self.board[row][col] == Points.SNAKE:
            return True
        return False

    def hit_wall(self, pos):
        row, col = pos
        if row < 0 or col < 0:
            return True
        if row >= self.board_size or col >= self.board_size:
            return True
        return False

    def print_board(self):
        result = ""
        for row in self.board:
            result += self.print_row(row)
            result += "\n"
        print(result)

    def print_row(self, row):
        result = "|"
        for r in row:
            if r == Points.EMPTY:
                result += " "
            elif r == Points.SNAKE:
                result += "X"
            elif r == Points.HEAD:
                if self.facing == Facing.UP:
                    result += "v"
                elif self.facing == Facing.RIGHT:
                    result += ">"
                elif self.facing == Facing.DOWN:
                    result += "^"
                elif self.facing == Facing.LEFT:
                    result += "<"
            elif r == Points.FOOD:
                result += "O"
        result += "|"
        return result


game = Game()
game.play_game()
# df = pd.read_pickle('./snake_agent.pkl')
# print(df)
