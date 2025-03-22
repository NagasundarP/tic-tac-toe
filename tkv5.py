import numpy as np
import random
import tkinter as tk
from tkinter import messagebox

class TicTacToe:
    def __init__(self, root):
        self.root = root
        self.root.title("Tic Tac Toe RL")
        self.board = np.zeros((3, 3), dtype=int)
        self.buttons = [[tk.Button(root, text="", font=("Arial", 20), height=2, width=5, state=tk.DISABLED) for c in range(3)] for r in range(3)]
        for r in range(3):
            for c in range(3):
                self.buttons[r][c].grid(row=r, column=c)
        self.agent1 = QLearningAgent()
        self.agent2 = QLearningAgent()
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

class QLearningAgent:
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
        
        # Check for a winning move
        for move in available_moves:
            temp_board = np.array(state).reshape((3, 3))
            temp_board[move[0], move[1]] = player
            if self.is_winning_move(temp_board, player):
                return move
        
        # Check for a blocking move
        opponent = -player
        for move in available_moves:
            temp_board = np.array(state).reshape((3, 3))
            temp_board[move[0], move[1]] = opponent
            if self.is_winning_move(temp_board, opponent):
                return move
        
        # Prioritize center if available
        if (1, 1) in available_moves:
            return (1, 1)
        
        # Try to create a fork move
        for move in available_moves:
            temp_board = np.array(state).reshape((3, 3))
            temp_board[move[0], move[1]] = player
            if self.count_potential_wins(temp_board, player) > 1:
                return move
        
        # Otherwise, pick the best Q-value move
        q_values = {move: self.get_q_value(state, move) for move in available_moves}
        return max(q_values, key=q_values.get, default=None)
    
    def is_winning_move(self, board, player):
        for i in range(3):
            if abs(sum(board[i, :])) == 3 or abs(sum(board[:, i])) == 3:
                return True
        if abs(sum([board[i, i] for i in range(3)])) == 3 or abs(sum([board[i, 2 - i] for i in range(3)])) == 3:
            return True
        return False
    
    def count_potential_wins(self, board, player):
        win_count = 0
        for i in range(3):
            if sum(board[i, :]) == 2 * player and 0 in board[i, :]:
                win_count += 1
            if sum(board[:, i]) == 2 * player and 0 in board[:, i]:
                win_count += 1
        if sum([board[i, i] for i in range(3)]) == 2 * player and 0 in [board[i, i] for i in range(3)]:
            win_count += 1
        if sum([board[i, 2 - i] for i in range(3)]) == 2 * player and 0 in [board[i, 2 - i] for i in range(3)]:
            win_count += 1
        return win_count

if __name__ == "__main__":
    root = tk.Tk()
    game = TicTacToe(root)
    root.mainloop()