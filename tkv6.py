import numpy as np
import random
import tkinter as tk
from tkinter import messagebox

class TicTacToe:
    def __init__(self, root):
        self.root = root
        self.root.title("Tic Tac Toe RL + Minimax")
        self.board = np.zeros((3, 3), dtype=int)
        self.buttons = [[tk.Button(root, text="", font=("Arial", 20), height=2, width=5, state=tk.DISABLED) for c in range(3)] for r in range(3)]
        for r in range(3):
            for c in range(3):
                self.buttons[r][c].grid(row=r, column=c)
        self.agent1 = HybridAgent()
        self.agent2 = HybridAgent()
        self.current_player = random.choice([1, -1])
        self.make_random_first_move()
        self.root.after(500, self.auto_play)
    
    def make_random_first_move(self):
        if self.current_player == 1:
            first_move = random.choice(self.available_moves())
            self.board[first_move[0], first_move[1]] = self.current_player
            self.buttons[first_move[0]][first_move[1]].config(text="X")
            self.current_player *= -1
    
    def auto_play(self):
        if not self.check_winner():
            agent = self.agent1 if self.current_player == 1 else self.agent2
            move = agent.best_action(tuple(self.board.flatten()), self.available_moves(), self.current_player)
            if move:
                self.board[move[0], move[1]] = self.current_player
                self.buttons[move[0]][move[1]].config(text="X" if self.current_player == 1 else "O")
            self.current_player *= -1
            self.root.after(500, self.auto_play)
    
    def available_moves(self):
        return [(r, c) for r in range(3) for c in range(3) if self.board[r, c] == 0]
    
    def check_winner(self):
        for i in range(3):
            if abs(sum(self.board[i, :])) == 3:
                self.end_game(f"Player {'X' if self.board[i, 0] == 1 else 'O'} Wins!")
                return True
            if abs(sum(self.board[:, i])) == 3:
                self.end_game(f"Player {'X' if self.board[0, i] == 1 else 'O'} Wins!")
                return True
        if abs(sum([self.board[i, i] for i in range(3)])) == 3:
            self.end_game(f"Player {'X' if self.board[0, 0] == 1 else 'O'} Wins!")
            return True
        if abs(sum([self.board[i, 2 - i] for i in range(3)])) == 3:
            self.end_game(f"Player {'X' if self.board[0, 2] == 1 else 'O'} Wins!")
            return True
        if not self.available_moves():
            self.end_game("Draw!")
            return True
        return False
    
    def end_game(self, message):
        messagebox.showinfo("Game Over", message)
        self.root.quit()

class HybridAgent:
    def __init__(self, alpha=0.1, gamma=0.9, epsilon=0.1):
        self.q_table = {}
        self.alpha = alpha
        self.gamma = gamma
        self.epsilon = epsilon
    
    def get_q_value(self, state, action):
        return self.q_table.get((state, action), 0.0)

    def best_action(self, state, available_moves, player):
        if random.random() < self.epsilon:
            return random.choice(available_moves) if available_moves else None
        
        q_values = {move: self.get_q_value(state, move) for move in available_moves}
        best_q_move = max(q_values, key=q_values.get, default=None)
        
        if best_q_move is None or q_values[best_q_move] == 0:
            return self.minimax_move(np.array(state).reshape((3, 3)), player)
        return best_q_move
    
    def minimax_move(self, board, player):
        best_score = -float('inf') if player == 1 else float('inf')
        best_move = None
        
        for move in [(r, c) for r in range(3) for c in range(3) if board[r, c] == 0]:
            board[move[0], move[1]] = player
            score = self.minimax(board, 0, False if player == 1 else True)
            board[move[0], move[1]] = 0
            
            if (player == 1 and score > best_score) or (player == -1 and score < best_score):
                best_score = score
                best_move = move
        
        return best_move
    
    def minimax(self, board, depth, is_max):
        winner = self.evaluate(board)
        if winner is not None:
            return winner
        
        scores = []
        for move in [(r, c) for r in range(3) for c in range(3) if board[r, c] == 0]:
            board[move[0], move[1]] = 1 if is_max else -1
            scores.append(self.minimax(board, depth + 1, not is_max))
            board[move[0], move[1]] = 0
        
        return max(scores) if is_max else min(scores)
    
    def evaluate(self, board):
        for i in range(3):
            if abs(sum(board[i, :])) == 3 or abs(sum(board[:, i])) == 3:
                return 1 if sum(board[i, :]) == 3 or sum(board[:, i]) == 3 else -1
        if abs(sum([board[i, i] for i in range(3)])) == 3 or abs(sum([board[i, 2 - i] for i in range(3)])) == 3:
            return 1 if sum([board[i, i] for i in range(3)]) == 3 or sum([board[i, 2 - i] for i in range(3)]) == 3 else -1
        if not any(0 in row for row in board):
            return 0
        return None

if __name__ == "__main__":
    root = tk.Tk()
    game = TicTacToe(root)
    root.mainloop()