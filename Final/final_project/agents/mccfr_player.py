import numpy as np
from collections import defaultdict
from game.players import BasePokerPlayer
from game.engine.hand_evaluator import HandEvaluator
from game.engine.card import Card
from game.engine.table import Table
from game.engine.player import Player
from game.engine.poker_constants import PokerConstants as Const
from game.engine.round_manager import RoundManager

class PokerState:
    def __init__(self, round_state, hole_card):
        self.round_state = round_state
        self.hole_card = [Card.from_str(card) for card in hole_card]
        self.community_cards = [Card.from_str(card) for card in round_state.get('community_card', [])]
        self.hand_evaluator = HandEvaluator()
        self.history = []

    def get_legal_actions(self):
        # Return a list of valid actions for the current state
        return self.round_state['legal_actions']

    def take_action(self, action, bet_amount=0):
        # Simulate taking an action and return the new state
        print('aa')
        new_round_state, _ = RoundManager.apply_action(self.round_state, action, bet_amount)
        print(new_round_state)
        print("aaa")
        new_state = PokerState(new_round_state, self.hole_card)
        new_state.history = self.history + [action]
        return new_state

    def is_terminal(self):
        # Check if the game has ended
        return self.round_state['street'] == Const.Street.SHOWDOWN

    def get_reward(self):
        # Calculate the reward for the current state
        return self.round_state['pot']['main']['amount']

    def get_information_set_key(self):
        # Generate a unique key for the current information set
        hole_card_str = ''.join([str(card) for card in self.hole_card])
        community_card_str = ''.join([str(card) for card in self.community_cards])
        history_str = '_'.join(self.history)
        return f"{hole_card_str}_{community_card_str}_{history_str}"

    @staticmethod
    def apply_action(round_state, action, bet_amount):
        new_round_state, messages = RoundManager.apply_action(round_state, action, bet_amount)
        return new_round_state, messages


class MCCFR:
    def __init__(self):
        self.regret_sum = defaultdict(lambda: np.zeros(3))  # Assuming 3 actions: fold, call, raise
        self.strategy_sum = defaultdict(lambda: np.zeros(3))

    def get_legal_actions(self):
        # Return the list of legal actions in the game
        return ['fold', 'call', 'raise']

    def get_strategy(self, information_set, realization_weight):
        regret_sum = self.regret_sum[information_set]
        strategy = np.maximum(regret_sum, 0)
        normalizing_sum = np.sum(strategy)
        if normalizing_sum > 0:
            strategy /= normalizing_sum
        else:
            strategy = np.ones(len(strategy)) / len(strategy)
        self.strategy_sum[information_set] += realization_weight * strategy
        return strategy

    def get_average_strategy(self, information_set):
        strategy_sum = self.strategy_sum[information_set]
        normalizing_sum = np.sum(strategy_sum)
        if normalizing_sum > 0:
            return strategy_sum / normalizing_sum
        else:
            return np.ones(len(strategy_sum)) / len(strategy_sum)

    def cfr(self, state: PokerState, realization_weight):
        if state.is_terminal():
            return state.get_reward()

        information_set = state.get_information_set_key()
        legal_actions = self.get_legal_actions()
        strategy = self.get_strategy(information_set, realization_weight)
        action_utilities = np.zeros(len(legal_actions))

        node_util = 0
        for i, action in enumerate(legal_actions):
            print("here",i,action)
            bet_amount = self.determine_bet_amount(action, state)
            if action == 'fold':
                action_utilities[i] += 0
            else:    
                next_state = state.take_action(action, bet_amount)

                action_utilities[i] = self.cfr(next_state, realization_weight * strategy[i])
                node_util += strategy[i] * action_utilities[i]

        for i, action in enumerate(legal_actions):
            regret = action_utilities[i] - node_util
            self.regret_sum[information_set][i] += realization_weight * regret

        return node_util

    def determine_bet_amount(self, action, state):
        # Determine the bet amount based on the action and current state
        current_player = state.round_state['seats'][state.round_state['next_player']]
        if action == 'raise':
            # Get the minimum and maximum raise amounts
            action_info = next(filter(lambda x: x['action'] == 'raise', state.get_legal_actions()))
            min_raise = action_info['amount']['min']
            max_raise = action_info['amount']['max']
            raise_amount = np.random.randint(min_raise, max_raise + 1)
            return raise_amount
        elif action == 'call':
            # Call amount is the current bet
            action_info = next(filter(lambda x: x['action'] == 'call', state.get_legal_actions()))
            return action_info['amount']
        else:
            return 0

    def run_iteration(self, initial_state):
        self.cfr(initial_state, 1.0)




class MCCFRPokerPlayer(BasePokerPlayer):
    def __init__(self, n_iterations=1000):
        super().__init__()
        self.mccfr = MCCFR()
        self.n_iterations = n_iterations

    def declare_action(self, valid_actions, hole_card, round_state):
        poker_state = PokerState(round_state, hole_card)
        for i in range(self.n_iterations):
            self.mccfr.run_iteration(poker_state)
        
        information_set = poker_state.get_information_set_key()
        strategy = self.mccfr.get_average_strategy(information_set)
        action_index = np.random.choice(len(valid_actions), p=strategy)
        selected_action = valid_actions[action_index]
        bet_amount = self.mccfr.determine_bet_amount(selected_action, poker_state)
        return selected_action, bet_amount



    def receive_game_start_message(self, game_info):
        print("receive_game_start_message called")

    def receive_round_start_message(self, round_count, hole_card, seats):
        print("receive_round_start_message called")

    def receive_street_start_message(self, street, round_state):
        print("receive_street_start_message called")

    def receive_game_update_message(self, new_action, round_state):
        print("receive_game_update_message called")

    def receive_round_result_message(self, winners, hand_info, round_state):
        # No need to update the policy here for MCCFR
        print("receive_round_result_message called")

def setup_ai():
    return MCCFRPokerPlayer()