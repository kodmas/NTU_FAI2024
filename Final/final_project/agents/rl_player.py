from game.players import BasePokerPlayer
import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions import Categorical

class PPOPlayer(BasePokerPlayer):
    def __init__(self):
        super().__init__()
        self.network = PolicyValueNetwork()
        self.optimizer = optim.Adam(self.network.parameters(), lr=0.0003)
        self.memory = []

    def declare_action(self, valid_actions, hole_card, round_state):
        state = self.encode_state(hole_card, round_state)
        action_probs, value = self.network(state)
        dist = Categorical(action_probs)
        action_index = dist.sample()

        action = valid_actions[action_index.item()]["action"]
        amount = valid_actions[action_index.item()]["amount"]
        if action == "raise":
            min_raise, max_raise = amount['min'], amount['max']
            amount = max(min_raise, min(max_raise, min_raise + (max_raise - min_raise) // 2))

        # Append to memory with log_prob and value
        log_prob = torch.log(action_probs.squeeze(0)[action_index])
        self.memory.append((state, log_prob, value))

        return action, amount


    def encode_state(self, hole_card, round_state):
        encoded_state = []
        # Encoding hole cards
        for card in hole_card:
            encoded_state.extend(self.encode_card(card))
        # Encoding community cards
        community_cards = round_state.get('community_card', [])
        for card in community_cards:
            encoded_state.extend(self.encode_card(card))
        # Normalizing and adding other state features
        pot_size = round_state['pot']['main']['amount'] / 1000  # Example normalization
        encoded_state.append(pot_size)

        return torch.tensor(encoded_state, dtype=torch.float32).unsqueeze(0)


    def encode_card(self, card):
        # Basic card encoding
        rank = '23456789TJQKA'.index(card[0])
        suit = 'cdhs'.index(card[1])
        return [rank / 13, suit / 4]  # Normalized

    def update_policy(self, reward):
        # Implement the PPO update rule here
        # Check if there is anything in memory to update from
        if not self.memory:
            print("No actions taken, skipping update.")
            return

        states, actions, old_values, log_probs = zip(*self.memory)
        states = torch.cat(states)
        actions = torch.stack(actions)
        old_values = torch.cat(old_values)
        
        # Dummy example: compute returns and advantages
        returns = torch.tensor(reward)
        advantages = returns - old_values

        # Calculate action probabilities and state values from the current policy
        new_probs, new_values = self.network(states)
        new_probs = new_probs.gather(1, actions.unsqueeze(1)).squeeze(1)
        old_probs = old_probs.gather(1, actions.unsqueeze(1)).squeeze(1)

        # PPO Loss calculation here
        ratio = new_probs / old_probs
        surr1 = ratio * advantages
        surr2 = torch.clamp(ratio, 1 - 0.2, 1 + 0.2) * advantages
        policy_loss = -torch.min(surr1, surr2).mean()
        value_loss = 0.5 * (returns - new_values).pow(2).mean()
        entropy = -(new_probs * torch.log(new_probs + 1e-5)).sum(1).mean()
        loss = policy_loss + value_loss - 0.01 * entropy  # entropy term to encourage exploration
        # loss = policy_loss + value_loss

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        self.memory = []

    def receive_game_start_message(self, game_info):
        pass

    def receive_round_start_message(self, round_count, hole_card, seats):
        pass

    def receive_street_start_message(self, street, round_state):
        pass

    def receive_game_update_message(self, new_action, round_state):
        pass
    
    def receive_round_result_message(self, winners, hand_info, round_state):
        # Determine reward based on whether the agent won or lost
        reward = 1 if self.uuid in [winner['uuid'] for winner in winners] else -1
        self.update_policy(reward)

class PolicyValueNetwork(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(20, 128)
        self.fc2 = nn.Linear(128, 128)
        self.action_head = nn.Linear(128, 3)
        self.value_head = nn.Linear(128, 1)
        # Initialize action head to promote equal probability of actions
        self.action_head.weight.data.fill_(0.0)
        self.action_head.bias.data.fill_(0.0)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        action_probs = torch.softmax(self.action_head(x), dim=-1)
        state_values = self.value_head(x)
        return action_probs, state_values

def setup_ai():
    return PPOPlayer()
