import argparse
import numpy as np

from dlgo.encoder import get_encoder_by_name, Encoder
from dlgo import goboard
from dlgo import mcts
from dlgo.utils import print_board, print_move


def generate_game(board_size, mcts_rounds, max_rounds, temperature):
    boards, moves = [], []
    encoder: Encoder = get_encoder_by_name('oneplane', board_size)
    bot = mcts.MCTSAgent(mcts_rounds, temperature)

    num_moves = 0
    game = goboard.GameState.new_game(board_size)

    while not game.is_over():
        print_board(game.board)
        move = bot.select_move(game)

        if move.is_play:
            boards.append(encoder.encode(game))
            encoded_move = np.zeros(encoder.num_points())
            encoded_move[encoder.encode_point(move.point)] = 1
            moves.append(encoded_move)
        print_move(game.next_player, move)
        num_moves += 1

        if num_moves > max_rounds:
            break

        game = game.apply_move(move)
    return np.array(boards), np.array(moves)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--board-size', '-b', type=int, default=9)
    parser.add_argument('--rounds', '-r', type=int, default=1000)
    parser.add_argument('--temperature', '-t', type=float, default=0.8)
    parser.add_argument('--max-moves', '-m', type=int, default=60,
                        help='Max moves per game.')
    parser.add_argument('--num-games', '-n', type=int, default=10)
    parser.add_argument('--board-out')
    parser.add_argument('--move-out')

    args = parser.parse_args()
    all_boards = []
    all_moves = []

    for i in range(args.num_games):
        boards, moves = generate_game(
            args.board_size, args.rounds, args.rounds, args.temperature)
        all_boards.append(boards)
        all_moves.append(moves)

    out_boards = np.concatenate(all_boards)
    out_moves = np.concatenate(all_moves)

    np.save(args.board_out, out_boards)
    np.save(args.move_Out, out_moves)


if __name__ == '__main__':
    main()
