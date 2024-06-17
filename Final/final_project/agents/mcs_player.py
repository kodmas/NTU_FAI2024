import random
from game.engine.hand_evaluator import HandEvaluator
from game.players import BasePokerPlayer
from agents.utils import gen_cards, estimate_hole_card_win_rate, _fill_community_card, \
    _pick_unused_card

NB_SIMULATION = 3000
def _montecarlo_simulation(nb_player, hole_card, community_card):
    community_card = _fill_community_card(community_card, used_card=hole_card+community_card)
    unused_cards = _pick_unused_card((nb_player-1)*2, hole_card + community_card)
    opponents_hole = [unused_cards[2*i:2*i+2] for i in range(nb_player-1)]
    opponents_score = [HandEvaluator.eval_hand(hole, community_card) for hole in opponents_hole]
    my_score = HandEvaluator.eval_hand(hole_card, community_card)
    return 1 if my_score >= max(opponents_score) else 0

def estimate_hole_card_win_rate(nb_simulation, nb_player, hole_card, community_card=None):
    if not community_card: community_card = []
    win_count = sum([_montecarlo_simulation(nb_player, hole_card, community_card) for _ in range(nb_simulation)])
    return 1.0 * win_count / nb_simulation

class MonteCarloPlayer(BasePokerPlayer):
    def __init__(self):
        super().__init__()
        self.wins = 0
        self.losses = 0
        self.bluffing = True
        self.turnCount = 0
        self.prevRound = 0
        self.opponent_actions = []
        self.opponent_model = {'aggressive': 0, 'passive': 0, 'neutral': 0}
        self.rounds_observed = 0
        self.opponent_type = 'neutral'
        self.threshold = [.75, .65 , .55, .25, .4]
        self.opponent_allin = -1
    # valid_actions[0] = fold, valid_actions[1] = call, and valid_actions[2] = raise
    #  we define the logic to make an action through this method. (so this method would be the core of your AI)
    def declare_action(self, valid_actions, hole_card, round_state):
        #determines the current round state so that we may better our AI's action accordingly
        if len(self.opponent_actions) != 0:
            opponent_action = self.opponent_actions[-1]['action']
            if opponent_action == 'raise':
                if round_state['seats'][0]['state'] == 'allin':
                    self.threshold[1] = self.threshold[2] = 0.72

        else:
            opponent_action = None
        #Need to take into account the community cards when calculating our winrate
        community_card = round_state['community_card']
        win_rate = estimate_hole_card_win_rate(
            nb_simulation=NB_SIMULATION,
            nb_player=self.nb_player,
            hole_card=gen_cards(hole_card),
            community_card=gen_cards(community_card)
        )

        #Print out the cards in the agent's hand for debugging purposes
        # print(hole_card)

        #Reset our initial variables before the start of each round
        if 'round_count' not in round_state.keys() and self.turn == 0:
            self.bluffing = False

        if 'round_count' in round_state.keys():
            if round_state['round_count'] != self.prevRound:
                self.prevRound = round_state['round_count']
                self.bluffing = False
                self.turnCount = 0

        pot_amount = round_state['pot']['main']['amount']
        stack = [player['stack'] for player in round_state['seats'] if player['uuid'] == self.uuid][0]
        raise_amount_options = [item for item in valid_actions if item['action'] == 'raise'][0]['amount']
        # opponent_action = opponent_action_dict['action']
        max = raise_amount_options['max']
        min = raise_amount_options['min']

        #Determine whether or not calling is a valid action during the current round state
        can_call = len([item for item in valid_actions if item['action'] == 'call']) > 0
        if can_call:
            # If so, compute the amount that needs to be called
            call_amount = [item for item in valid_actions if item['action'] == 'call'][0]['amount']
        else:
            call_amount = 0

        amount = None
        if opponent_action is not None:
            self.update_opponent_model(opponent_action)

        if opponent_action == "raise":
            if win_rate > self.threshold[0]:
                action = 'raise'
                amount = max
            elif win_rate > self.threshold[1]:
                action = 'raise'
                amount = int(((2000-stack)/2000)*(max-min)+min)
            elif win_rate > self.threshold[2]:
                action = 'call'
            else:
                if self.bluffing:
                    action = 'call'
                else:
                    num = random.uniform(0, 1)
                    if num > win_rate / 2:
                        action = 'fold'
                    elif can_call:
                        action = 'call'
        else:
            # raise less if opponent calls
            if win_rate > self.threshold[0]:
                action = 'raise'
                amount = int(((2000-stack)/2000)*(max-min)+min)
            elif win_rate > self.threshold[1]:
                action = 'raise'
                amount = int(.75*((2000-stack)/(2000))*(max - min) + min)
            elif win_rate > self.threshold[2]:
                action = 'call'
            else:
                num = random.uniform(0, 1)
                if self.bluffing:
                    if num > .5:
                        action = 'raise'
                        amount = int((stack/1000)/8*(max-min))
                        amount += min
                    else:
                        action = "call"
                else:
                    if can_call and call_amount == 0:
                        action = "call"
                        # print("Match Call 0")
                    elif num > win_rate:
                        action = 'fold'
                    elif num > win_rate/2 and can_call:
                        action = 'call'
                    else:
                        action = 'raise'
                        amount = int((stack/1000)/8*(max-min))
                        amount += min
                        self.bluffing = True

        if amount is None:
            items = [item for item in valid_actions if item['action'] == action]
            amount = items[0]['amount']

        if amount < 0 or self.turnCount == 0:
            action = 'call'
            items = [item for item in valid_actions if item['action'] == action]
            amount = items[0]['amount']

            if win_rate < self.threshold[3]:
                action = 'fold'

            if opponent_action == 'raise' and win_rate < self.threshold[4]:
                action = 'fold'

        if action == "raise" and amount > max:
            amount = max
        if action == "raise" and amount < min:
            amount = min

        self.turnCount += 1
        return action, amount
    
    def update_opponent_model(self, opponent_action):
        if opponent_action == 'raise':
            self.opponent_model['aggressive'] += 1
        elif opponent_action == 'call':
            self.opponent_model['neutral'] += 1
        elif opponent_action == 'fold':
            self.opponent_model['passive'] += 1

        self.rounds_observed += 1
        if self.rounds_observed == 8:
            self.classify_opponent()

    def classify_opponent(self):
        total_actions = sum(self.opponent_model.values())
        if total_actions == 0:
            return  # No actions observed

        for key in self.opponent_model:
            self.opponent_model[key] = self.opponent_model[key] / total_actions

        
        if self.opponent_model['aggressive'] >= 0.5:
            self.opponent_type = 'aggressive'
            self.threshold[0] += 0.1
            self.threshold[1] += 0.05
            self.threshold[2] -= 0.2 
            self.threshold[3] -= 0.1
        elif self.opponent_model['passive'] >= 0.4:
            self.opponent_type = 'passive'
            self.threshold[0] -= 0.1
            self.threshold[2] -= 0.1

        
        print(self.opponent_model)
        print("Opponent Model:", self.opponent_type)

    def get_last_opponent_action(self, round_state):
        try:
            return round_state['action_histories'][round_state['street']][-1]
        except:
            if round_state['street'] == 'turn':
                return round_state['action_histories']['flop'][-1]
            else:
                return round_state['action_histories']['preflop'][-1]
            
    def receive_game_start_message(self, game_info):
        self.nb_player = game_info['player_num']

    def receive_round_start_message(self, round_count, hole_card, seats):
        self.opponent_actions = []

    def receive_street_start_message(self, street, round_state):
        pass

    def receive_game_update_message(self, action, round_state):
        self.opponent_actions.append(action)

    def receive_round_result_message(self, winners, hand_info, round_state):
        result = self.uuid in [item['uuid'] for item in winners]
        self.wins += int(result)
        self.losses += int(not result)
        pass

def setup_ai():
    return MonteCarloPlayer()