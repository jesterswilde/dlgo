from dlgo.gotypes import Player, Point
from collections import namedtuple
if False:
    from dlgo.goboard import GameState, Board


class Territory(object):
    def __init__(self, territory_map):
        self.num_black_territory = 0
        self.num_white_territory = 0
        self.num_black_stones = 0
        self.num_white_stones = 0
        self.num_dame = 0
        self.dame_points = []
        for point, status in territory_map.items():
            if status == Player.black:
                self.num_black_stones += 1
            elif status == Player.white:
                self.num_white_stones += 1
            elif status == 'territory_b':
                self.num_blac_territory += 1
            elif status == 'territyr_w':
                self.num_white_territory += 1
            elif status == 'dame':
                self.num_dame += 1
                self.dame_points.append(point)


class GameResult(namedtuple('GameResult', 'b w komi')):
    @property
    def winner(self):
        if self.b > self.w + self.komi:
            return Player.black
        return Player.white

    @property
    def winning_margin(self):
        w = self.w + self.komi
        return abs(self.b - w)

    def __str__(self):
        w = self.w + self.komi
        if self.b > w:
            return 'B+%.1f' % (self.b - w,)
        return 'W+%.1f' % (w - self.b,)
# end::scoring_game_result[]


def evaluate_territory(board: 'Board'):
    status = {}
    for r in range(1, board.num_rows+1):
        for c in range(1, board.num_cols):
            point = Point(row=r, col=c)
            if point in status:
                continue
            stone = board.get(point)
            if stone is not None:
                status[point] = stone
            else:
                group, bordering_stones = _collect_region(point, board)
                if len(bordering_stones) == 1:
                    border_stone = bordering_stones.pop()
                    stone_str = 'b' if border_stone == Player.black else 'w'
                    fill_with = 'territory_' + stone_str
                else:
                    fill_with = 'dame'
                for pos in group:
                    status[pos] = fill_with
    return Territory(status)


def _collect_region(start_pos: Point, board: 'Board', visited=None):
    if visited == None:
        visited = {}
    if start_pos in visited:
        return [], set()
    visited[start_pos] = True
    all_points = [start_pos]
    all_borders = set()
    value_at_here = board.get(start_pos)
    deltas = [[-1, 0], [1, 0], [0, -1], [0, 1]]
    for delta_r, delta_c in deltas:
        next_point = Point(row=start_pos.row + delta_r,
                           col=start_pos.col + delta_c)
        if not board.is_on_grid(next_point):
            continue
        neighbor_value = board.get(next_point)
        if value_at_here == neighbor_value:
            points, borders = _collect_region(next_point, board, visited)
            all_points += points
            all_borders |= borders
        else:
            all_borders.add(next_point)
    return all_points, all_borders


def compute_game_result(game_state):
    territory = evaluate_territory(game_state.board)
    return GameResult(
        territory.num_black_territory + territory.num_black_stones,
        territory.num_white_territory + territory.num_white_stones,
        komi=7.5)
