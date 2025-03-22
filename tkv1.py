import numpy as np
import random
import tkinter as tk
from tkinter import messagebox

class TicTacToe:
    def __init__(self):
        self.board = np.zeros((3, 3), dtype=int)

    def reset(self):
        self.board.fill(0)

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

class TicTacToeApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Tic Tac Toe RL")
        self.game = TicTacToe()
        self.opponent = MinimaxAgent()
        self.buttons = [[None for _ in range(3)] for _ in range(3)]
        self.create_board()
    
    def create_board(self):
        for r in range(3):
            for c in range(3):
                self.buttons[r][c] = tk.Button(self.root, text="", font=("Arial", 24), height=2, width=5,
                                               command=lambda row=r, col=c: self.on_click(row, col))
                self.buttons[r][c].grid(row=r, column=c)
    
    def on_click(self, row, col):
        if self.game.board[row, col] == 0:
            self.game.make_move(row, col, 1)
            self.update_board()
            if self.check_game_over():
                return
            move = self.opponent.best_move(self.game)
            if move:
                self.game.make_move(*move, -1)
            self.update_board()
            self.check_game_over()
    
    def update_board(self):
        for r in range(3):
            for c in range(3):
                if self.game.board[r, c] == 1:
                    self.buttons[r][c].config(text="X", fg="blue")
                elif self.game.board[r][c] == -1:
                    self.buttons[r][c].config(text="O", fg="red")
    
    def check_game_over(self):
        winner = self.game.check_winner()
        if winner is not None:
            if winner == 1:
                messagebox.showinfo("Game Over", "You Win!")
            elif winner == -1:
                messagebox.showinfo("Game Over", "AI Wins!")
            else:
                messagebox.showinfo("Game Over", "It's a Draw!")
            self.reset_game()
            return True
        return False
    
    def reset_game(self):
        self.game.reset()
        for r in range(3):
            for c in range(3):
                self.buttons[r][c].config(text="")

if __name__ == "__main__":
    root = tk.Tk()
    app = TicTacToeApp(root)
    root.mainloop()