from game.players import BasePokerPlayer
import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions import Categorical
from game.engine.hand_evaluator import HandEvaluator
from game.engine.card import Card
import math
import numpy as np
import random
class MCTSNode:
    def __init__(self, state, parent=None, action=None):
        self.state = state
        self.parent = parent
        self.action = action
        self.children = []
        self.visits = 0
        self.value = 0

    def is_fully_expanded(self):
        # Check if all possible actions from this node have been expanded
        possible_actions = self.state.get_legal_actions()
        return len(self.children) == len(possible_actions)

    def best_child(self, c_param=1.4):
        # Select the child with the highest UCB1 value
        choices_weights = [
            (child.value / child.visits) + c_param * math.sqrt((2 * math.log(self.visits) / child.visits))
            for child in self.children
        ]
        return self.children[np.argmax(choices_weights)]

    def expand(self):
        # Expand a new child node from the current node
        possible_actions = self.state.get_legal_actions()
        for action in possible_actions:
            if action not in [child.action for child in self.children]:
                new_state = self.state.take_action(action)
                child_node = MCTSNode(new_state, parent=self, action=action)
                self.children.append(child_node)
                return child_node

    def rollout(self):
        # Simulate a random play until the game ends and return the reward
        current_state = self.state
        while not current_state.is_terminal():
            action = random.choice(current_state.get_legal_actions())
            current_state = current_state.take_action(action)
        return current_state.get_reward()

    def backpropagate(self, reward):
        # Backpropagate the reward through the ancestors of the node
        self.visits += 1
        self.value += reward
        if self.parent:
            self.parent.backpropagate(reward)

class MCTS:
    def __init__(self, root_state):
        self.root = MCTSNode(root_state)

    def search(self, n_simulations):
        for _ in range(n_simulations):
            node = self.select(self.root)
            reward = node.rollout()
            node.backpropagate(reward)
        return self.root.best_child(c_param=0.0).action

    def select(self, node):
        # Select a node to expand
        while not node.state.is_terminal():
            if not node.is_fully_expanded():
                return node.expand()
            else:
                node = node.best_child()
        return node

class PokerState:
    def __init__(self, game_state):
        self.game_state = game_state

    def get_legal_actions(self):
        # Return a list of legal actions from this state
        return self.game_state.legal_actions()

    def take_action(self, action):
        # Return a new state after taking the given action
        new_game_state = self.game_state.apply_action(action)
        return PokerState(new_game_state)

    def is_terminal(self):
        # Return True if the game is over
        return self.game_state.is_terminal()

    def get_reward(self):
        # Return the reward for the current state
        return self.game_state.get_reward()

class MCTSPokerPlayer(BasePokerPlayer):
    def __init__(self, n_simulations=1000):
        super().__init__()
        self.n_simulations = n_simulations

    def declare_action(self, valid_actions, hole_card, round_state):
        poker_state = PokerState(round_state)
        mcts = MCTS(poker_state)
        best_action = mcts.search(self.n_simulations)
        return best_action.action, best_action.amount

    def receive_game_start_message(self, game_info):
        pass

    def receive_round_start_message(self, round_count, hole_card, seats):
        pass

    def receive_street_start_message(self, street, round_state):
        pass

    def receive_game_update_message(self, new_action, round_state):
        pass

    def receive_round_result_message(self, winners, hand_info, round_state):
        pass


def setup_ai():
    return MCTSPokerPlayer()