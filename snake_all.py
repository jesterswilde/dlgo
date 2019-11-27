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


random.seed(1)
rand_state = random.getstate()
FILENAME = './snake_agent_all.pkl'
EYE_SIGHT = 2
SAVE_EVERY = 500
PRINT_EVERY = 1
EPOCHS = 1000000
MAX_HISTORY = 10
SHOULD_SLEEP = True
RAND_MOVE = 0.05
SLEEP_AMOUNT = 0.05
HISTORY_LEN = 100


class Agent:
    learning_amount = 0.5
    discount = 0.95

    def __init__(self):
        if os.path.isfile(FILENAME):
            self.q_table = pd.read_pickle(FILENAME)
        else:
            self.q_table = pd.DataFrame(columns=[0, 1, 2, 3])
        # self.q_table = pd.DataFrame(columns=[0, 1, 2, 3])
        self.last_state = None
        self.last_action = None

    def check_value(self, state):
        if state not in self.q_table.index:
            self.q_table = self.q_table.append(pd.Series([
                random.uniform(0, 1),
                random.uniform(0, 1),
                random.uniform(0, 1),
                random.uniform(0, 1)
            ], name=state))

    def pick_random(self, state):
        action = random.randint(0, 3)
        self.last_state = state
        self.last_action = action
        return action

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
    def __init__(self, board_size=8):
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
                if self.board.is_valid_action(move):
                    reward = self.board.take_action(move)
                    state = self.encode_board()
                    self.agent.learn(
                        reward=reward, new_state=self.encode_board())
                else:
                    self.agent.break_loop()
                if printing:
                    # self.agent.print_last()
                    self.board.print_board()
                    print("Game #: {}".format(game))
                    if SHOULD_SLEEP:
                        sleep(SLEEP_AMOUNT)
            record.append(self.board.fruit_eaten)
            if len(record) == HISTORY_LEN:
                total = sum(record)
                print("Average: {} Max:{} Min:{} for game {} - {}".format(total /
                                                                          len(record), max(record), min(record), game - HISTORY_LEN, game))
                record = []
            random.setstate(rand_state)
            self.reset()
            if (game % SAVE_EVERY) == 0 and game != 0:
                self.agent.q_table.to_pickle(FILENAME)
        # print(self.agent.q_table)

    def encode_board_small(self):
        return self.encode_board(size=1)

    def encode_board(self, size=0):
        # head_row, head_col = self.board.head
        # # self.board
        # encoded_board = []
        # for row in range(head_row-EYE_SIGHT+size, head_row+EYE_SIGHT+1-size):
        #     for col in range(head_col-EYE_SIGHT+size, head_col+EYE_SIGHT+1-size):
        #         if row < 0 or row >= self.board_size:
        #             value = int(Points.WALL)
        #         elif col < 0 or col >= self.board_size:
        #             value = int(Points.WALL)
        #         else:
        #             value = int(self.board.board[row][col])
        #         encoded_board.append(value)
        # food_row, food_col = self.board.food_pos
        # if head_row < food_row:
        #     x_diff = -1
        # elif head_row > food_row:
        #     x_diff = 1
        # else:
        #     x_diff = 0
        # if head_col < food_col:
        #     y_diff = -1
        # elif head_col > food_col:
        #     y_diff = 1
        # else:
        #     y_diff = 0
        result = []
        for row in range(self.board_size):
            for col in range(self.board_size):
                result.append(int(self.board.board[row][col]))
        return str(result)


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

    def place_food(self):
        valid_moves = []
        for row in range(self.board_size):
            for col in range(self.board_size):
                if self.board[row][col] is Points.EMPTY:
                    valid_moves.append((row, col))
        food_row, food_col = random.choice(valid_moves)
        self.board[food_row][food_col] = Points.FOOD
        self.food_pos = (food_row, food_col)

    def is_valid_action(self, move: Facing):
        if self.facing == Facing.UP:
            return move != Facing.DOWN
        if self.facing == Facing.RIGHT:
            return move != Facing.LEFT
        if self.facing == Facing.DOWN:
            return move != Facing.UP
        if self.facing == Facing.LEFT:
            return move != Facing.RIGHT
        return True

    def take_action(self, action: Facing):
        row, col = self.head
        if action == Facing.RIGHT:
            next_head = (row, col + 1)
        elif action == Facing.UP:
            next_head = (row + 1, col)
        elif action == Facing.DOWN:
            next_head = (row - 1, col)
        else:
            next_head = (row, col - 1)
        if self.hit_wall(next_head) or self.hit_self(next_head):
            self.game_is_going = False
            return -1
        ate_food = self.ate_fruit(next_head)
        self.board[row][col] = Points.SNAKE
        self.board[next_head[0]][next_head[1]] = Points.HEAD
        self.snake.put(next_head)
        self.head = next_head
        self.facing = action
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
                result += "G"
            elif r == Points.FOOD:
                result += "O"
        result += "|"
        return result


game = Game()
game.play_game()
# df = pd.read_pickle('./snake_agent.pkl')
# print(df)
