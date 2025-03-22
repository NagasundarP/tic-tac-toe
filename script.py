import numpy as np
import random
import pickle
import pygame


def init_pygame():
    pygame.init()
    screen = pygame.display.set_mode((300, 300))
    pygame.display.set_caption("Tic Tac Toe RL")
    return screen

WHITE = (255, 255, 255)
BLACK = (0, 0, 0)
RED = (255, 0, 0)
BLUE = (0, 0, 255)

def draw_board(screen, board):
    screen.fill(WHITE)
    for i in range(1, 3):
        pygame.draw.line(screen, BLACK, (0, i * 100), (300, i * 100), 3)
        pygame.draw.line(screen, BLACK, (i * 100, 0), (i * 100, 300), 3)
    for r in range(3):
        for c in range(3):
            if board[r, c] == 1:
                pygame.draw.circle(screen, BLUE, (c * 100 + 50, r * 100 + 50), 40, 3)
            elif board[r, c] == -1:
                pygame.draw.line(screen, RED, (c * 100 + 20, r * 100 + 20), (c * 100 + 80, r * 100 + 80), 3)
                pygame.draw.line(screen, RED, (c * 100 + 80, r * 100 + 20), (c * 100 + 20, r * 100 + 80), 3)
    pygame.display.flip()

class TicTacToe:
    def __init__(self):
        self.board = np.zeros((3, 3), dtype=int)  

    def reset(self):
        self.board.fill(0)
        return self.get_state()

    def get_state(self):
        return tuple(self.board.flatten())

    def available_moves(self):
        return [(r, c) for r in range(3) for c in range(3) if self.board[r, c] == 0]

    def make_move(self, row, col, player):
        self.board[row, col] = player

    def check_winner(self):
        for i in range(3):
            if abs(sum(self.board[i, :])) == 3:
                return np.sign(sum(self.board[i, :]))
            if abs(sum(self.board[:, i])) == 3:
                return np.sign(sum(self.board[:, i]))
        if abs(sum([self.board[i, i] for i in range(3)])) == 3:
            return np.sign(sum([self.board[i, i] for i in range(3)]))
        if abs(sum([self.board[i, 2 - i] for i in range(3)])) == 3:
            return np.sign(sum([self.board[i, 2 - i]]))
        if len(self.available_moves()) == 0:
            return 0  # Draw
        return None

class QLearningAgent:
    def __init__(self, alpha=0.1, gamma=0.9, epsilon=0.1):
        self.q_table = {}
        self.alpha = alpha
        self.gamma = gamma
        self.epsilon = epsilon

    def get_q_value(self, state, action):
        return self.q_table.get((state, action), 0.0)

    def best_action(self, state, available_moves):
        if random.random() < self.epsilon:
            return random.choice(available_moves)
        q_values = {move: self.get_q_value(state, move) for move in available_moves}
        return max(q_values, key=q_values.get)

    def update_q_value(self, state, action, reward, next_state):
        max_future_q = max([self.get_q_value(next_state, move) for move in TicTacToe().available_moves()], default=0)
        self.q_table[(state, action)] = (1 - self.alpha) * self.get_q_value(state, action) + self.alpha * (reward + self.gamma * max_future_q)

class MinimaxAgent:
    def minimax(self, game, depth, is_max):
        winner = game.check_winner()
        if winner == 1:
            return 10 - depth
        elif winner == -1:
            return depth - 10
        elif winner == 0:
            return 0
        
        if is_max:
            best_score = -float('inf')
            for move in game.available_moves():
                game.make_move(*move, 1)
                score = self.minimax(game, depth + 1, False)
                game.make_move(*move, 0)
                best_score = max(best_score, score)
            return best_score
        else:
            best_score = float('inf')
            for move in game.available_moves():
                game.make_move(*move, -1)
                score = self.minimax(game, depth + 1, True)
                game.make_move(*move, 0)
                best_score = min(best_score, score)
            return best_score

    def best_move(self, game):
        best_score = -float('inf')
        best_move = None
        for move in game.available_moves():
            game.make_move(*move, 1)
            score = self.minimax(game, 0, False)
            game.make_move(*move, 0)
            if score > best_score:
                best_score = score
                best_move = move
        return best_move

def play_game():
    screen = init_pygame()
    game = TicTacToe()
    agent = QLearningAgent()
    opponent = MinimaxAgent()
    running = True
    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            elif event.type == pygame.MOUSEBUTTONDOWN:
                x, y = event.pos
                row, col = y // 100, x // 100
                if (row, col) in game.available_moves():
                    game.make_move(row, col, 1)
                    draw_board(screen, game.board)
                    if game.check_winner():
                        running = False
                    move = opponent.best_move(game)
                    if move:
                        game.make_move(*move, -1)
                    draw_board(screen, game.board)
                    if game.check_winner():
                        running = False
    pygame.quit()

if __name__ == "__main__":
    play_game()
