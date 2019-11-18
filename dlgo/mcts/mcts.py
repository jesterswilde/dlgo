from dlgo.goboard import GameState, Move
from dlgo.gotypes import Player
from dlgo.agent.base import Agent
from dlgo.agent.naive import RandomBot
from typing import List
import math
import random

__all__ = [
    'MCTSAgent'
]


class MCTSAgent(Agent):
    def __init__(self, num_rounds, temperature):
        super().__init__()
        self.temperature = temperature
        self.num_rounds = num_rounds

    def select_move(self, game_state: GameState):
        root = MCTSNode(game_state)

        for i in range(self.num_rounds):
            node = root
            while (not node.can_add_child()) and (not node.is_terminal()):
                node = self.select_child(node)

            if node.can_add_child():
                node = node.add_random_child()

            winner = self.simulate_random_game(node.game_state)

            while node is not None:
                node.record_win(winner)
                node = node.parent

        best_move = None
        best_pct = -1.0
        for child in root.children:
            child_pct = child.winning_frac(game_state.next_player)
            if child_pct > best_pct:
                best_pct = child_pct
                best_move = child.move

        return best_move

    def select_child(self, node):
        total_rollouts = sum(child.num_rollouts for child in node.children)

        best_score = -1
        best_child = None
        for child in node.children:
            score = uct_score(
                total_rollouts,
                child.num_rollouts,
                child.winning_frac(node.game_state.next_player),
                self.temperature
            )
            if score > best_score:
                best_score = score
                best_child = child

        return best_child

    @staticmethod
    def simulate_random_game(game: GameState):
        bots = {
            Player.black: RandomBot(),
            Player.white: RandomBot()
        }

        while(not game.is_over()):
            bot_move = bots[game.next_player].select_move(game)
            game = game.apply_move(bot_move)
        return game.winner()


def uct_score(parent_rollout, child_rollout, win_pct, temperature) -> float:
    exploration = math.sqrt(math.log(parent_rollout) / child_rollout)
    return win_pct + temperature * exploration


class MCTSNode:
    def __init__(self, game_state: GameState, parent: 'MCTSNode' = None, move: Move = None):
        self.parent = parent
        self.game_state = game_state
        self.move = move
        self.win_counts = {
            Player.white: 0,
            Player.black: 0
        }
        self.num_rollouts = 0
        self.children: List[MCTSNode] = []
        self.univisted_moves = game_state.legal_moves()

    def add_random_child(self):
        index = random.randint(0, len(self.univisted_moves) - 1)
        next_move = self.univisted_moves.pop(index)
        next_game_state = self.game_state.apply_move(next_move)
        new_node = MCTSNode(next_game_state, self, next_move)
        self.children.append(new_node)
        return new_node

    def record_win(self, winner):
        self.win_counts[winner] += 1
        self.num_rollouts += 1

    def can_add_child(self):
        return len(self.univisted_moves) > 0

    def is_terminal(self):
        return self.game_state.is_over()

    def winning_frac(self, player):
        return float(self.win_counts[player]) / float(self.num_rollouts)
