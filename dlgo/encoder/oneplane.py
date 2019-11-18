import numpy as np
from typing import Tuple

from dlgo.encoder.base import Encoder
from dlgo.goboard import GameState
from dlgo.gotypes import Point

__all__ = [
    "OnePlaneEncoder",
    "create"
]


class OnePlaneEncoder(Encoder):
    board_width: int
    board_height: int

    def __init__(self, board_size: Tuple[int]):
        self.board_width, self.board_height = board_size
        self.num_planes = 1

    def name(self):
        return 'oneplane'

    def encode(self, game_state: GameState):
        board_matrix = np.zeros(self.shape())
        next_player = game_state.next_player
        for row in range(self.board_width):
            for col in range(self.board_height):
                point = Point(row=row+1, col=col+1)
                go_string = game_state.board.get_go_string(point)
                if go_string is None:
                    continue
                if go_string.color == next_player:
                    board_matrix[0, row, col] = 1
                else:
                    board_matrix[0, row, col] = -1
        return board_matrix

    def encode_point(self, point: Point):
        return self.board_width * (point.row - 1) + (point.col - 1)

    def decode_point_index(self, index):
        row = index // self.board_width
        col = index % self.board_width
        return Point(row=row+1, col=col+1)

    def num_points(self):
        return self.board_height * self.board_width

    def shape(self):
        return self.num_planes, self.board_height, self.board_width


def create(board_size):
    return OnePlaneEncoder(board_size)
