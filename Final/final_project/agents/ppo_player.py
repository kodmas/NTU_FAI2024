from game.players import BasePokerPlayer
import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions import Categorical
from game.engine.hand_evaluator import HandEvaluator
from game.engine.card import Card

class PPOPlayer(BasePokerPlayer):
    def __init__(self, model_path=None):
        super().__init__()
        self.network = PolicyValueNetwork()
        self.optimizer = optim.Adam(self.network.parameters(), lr=0.001)
        self.memory = []
        self.rewards = []
        self.gamma = 0.98  # Discount factor for future rewards
        self.update_frequency = 50  # Lower update frequency for more frequent updates
        self.hand_evaluator = HandEvaluator()

        if model_path:
            self.load_model(model_path)
        # print("PPOPlayer initialized")

    def declare_action(self, valid_actions, hole_card, round_state):
        print("declare_action called")
        try:
            state = self.encode_state(hole_card, round_state)
            print("State encoded:", state)
            action_probs, value, raise_amount = self.network(state)
            print("Network output - action_probs:", action_probs, "value:", value, "raise_amount:", raise_amount)
            dist = Categorical(action_probs)

            action_index = dist.sample()  # Sample during training for exploration
            # print("a",action_index)
            # else:
            # action_index = torch.tensor([torch.argmax(action_probs)]) # Use argmax during evaluation
            # print("b",action_index)
            print(f"Action probabilities: {action_probs}, Selected action index: {action_index.item()}")

            action = valid_actions[action_index.item()]["action"]
            amount = valid_actions[action_index.item()]["amount"]
            if action == "raise":
                min_raise, max_raise = amount['min'], amount['max']
                amount = abs(raise_amount.item() * (max_raise - min_raise) + min_raise)
                amount = max(min_raise, min(max_raise, amount))

            # Append to memory with state, action, log_prob, and value
            log_prob = torch.log(action_probs.squeeze(0)[action_index])
            self.memory.append((state, action_index, log_prob, value, raise_amount))

            print(f"Action taken: {action}, Amount: {amount}")
        except Exception as e:
            print(f"Exception in declare_action: {e}")
        finally:
            import sys
            sys.stdout.flush()
        return action, amount


    def encode_state(self, hole_card, round_state):
        # print("encode_state called")
        encoded_state = []

        # Encoding hole cards
        formatted_hole_card = [Card.from_str(card) for card in hole_card]
        for card in hole_card:
            encoded_state.extend(self.encode_card(card))

        # Encoding community cards with placeholders for missing cards
        community_cards = round_state.get('community_card', [])
        formatted_community_cards = [Card.from_str(card) for card in community_cards]
        for card in community_cards:
            encoded_state.extend(self.encode_card(card))
        for _ in range(5 - len(community_cards)):  # Placeholder for missing community cards
            encoded_state.extend([0, 0])

        # Hand strength evaluation
        hand_rank_info = self.hand_evaluator.gen_hand_rank_info(formatted_hole_card, formatted_community_cards)
        hand_strength = self.hand_strength_to_value(hand_rank_info['hand']['strength'])
        # print("ff",hand_rank_info['hand']['high'],hand_rank_info['hand']['low'])
        hand_high = hand_rank_info['hand']['high'] / 13  # Normalize by the highest rank
        hand_low = hand_rank_info['hand']['low'] / 13   # Normalize by the highest rank
        encoded_state.append(hand_strength)
        encoded_state.append(hand_high)
        encoded_state.append(hand_low)

        # Encoding pot size (normalized)
        pot_size = round_state['pot']['main']['amount'] / 1000  # Example normalization
        encoded_state.append(pot_size)

        # Optionally encode player stack sizes or other game features
        player_stacks = [seat['stack'] for seat in round_state['seats']]
        encoded_state.extend(player_stacks)

        # print(f"Encoded state: {encoded_state}")
        return torch.tensor(encoded_state, dtype=torch.float32).unsqueeze(0)
    
    
    def hand_strength_to_value(self, strength):
        # Convert hand strength from string to numerical value
        mapping = {
            "HIGHCARD": 0,
            "ONEPAIR": 1,
            "TWOPAIR": 2,
            "THREECARD": 3,
            "STRAIGHT": 4,
            "FLASH": 5,
            "FULLHOUSE": 6,
            "FOURCARD": 7,
            "STRAIGHTFLASH": 8
        }
        return mapping.get(strength, 0)
    
    def encode_card(self, card):
        # print("encode_card called")
        # Basic card encoding
        rank = '23456789TJQKA'.index(card[1])
        suit = 'CDHS'.index(card[0])
        return [rank / 13, suit / 4]  # Normalized

    def update_policy(self):
        # print("update_policy called")
        # Check if there is enough data to perform an update
        if len(self.memory) < self.update_frequency:
            # print("Not enough data to update policy")
            return

        states, actions, log_probs, values, raise_amounts = zip(*self.memory)
        states = torch.cat(states)
        actions = torch.tensor(actions)
        log_probs = torch.stack(log_probs)
        values = torch.cat(values)
        raise_amounts = torch.cat(raise_amounts)

        # Compute returns and advantages
        returns = []
        R = 0
        for reward in reversed(self.rewards):
            R = reward + self.gamma * R
            returns.insert(0, R)
        returns = torch.tensor(returns, dtype=torch.float32).unsqueeze(1)  # Shape: [batch_size, 1]

        advantages = returns - values.unsqueeze(1)  # Shape: [batch_size, 1]

        # Calculate action probabilities and state values from the current policy
        new_probs, new_values, new_raise_amounts = self.network(states)
        # print(f"new_probs shape: {new_probs.shape}")
        # print(f"new_values shape: {new_values.shape}")
        # print("---------")
        new_probs = new_probs.gather(1, actions.unsqueeze(1)).squeeze(1)  # Shape: [batch_size]
        old_probs = torch.exp(log_probs).squeeze(1)  # Shape: [batch_size]

        # Ensure dimensions are correct
        # print(f"new_probs shape: {new_probs.shape}")
        # print(f"old_probs shape: {old_probs.shape}")
        # print(f"advantages shape: {advantages.shape}")

        # PPO Loss calculation
        ratio = new_probs / old_probs
        surr1 = ratio * advantages.squeeze(1)  # Shape: [batch_size]
        surr2 = torch.clamp(ratio, 1 - 0.2, 1 + 0.2) * advantages.squeeze(1)  # Shape: [batch_size]
        policy_loss = -torch.min(surr1, surr2).mean()
        value_loss = 0.5 * (returns.squeeze(1) - new_values).pow(2).mean()
        entropy = -(new_probs * torch.log(new_probs + 1e-5)).mean()
        raise_loss = (raise_amounts - new_raise_amounts).pow(2).mean()  # Loss for the raise amount prediction
        loss = 0.5 * policy_loss + value_loss - 0.05 * entropy + 2 * raise_loss  # Adjust weights as necessary

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        self.memory = []  # Clear memory after update
        self.rewards = []  # Clear rewards after update
        print("Policy updated")

    def save_model(self, file_path='ppo_model.pth'):
        torch.save(self.network.state_dict(), file_path)
        print(f"Model saved to {file_path}")

    def load_model(self, file_path='ppo_model.pth'):
        self.network.load_state_dict(torch.load(file_path))
        print(f"Model loaded from {file_path}")

    def receive_game_start_message(self, game_info):
        print("receive_game_start_message called")

    def receive_round_start_message(self, round_count, hole_card, seats):
        print("receive_round_start_message called")

    def receive_street_start_message(self, street, round_state):
        print("receive_street_start_message called")

    def receive_game_update_message(self, new_action, round_state):
        print("receive_game_update_message called")

    def receive_round_result_message(self, winners, hand_info, round_state):
        print("receive_round_result_message called")
        stack = winners[0]['stack']
        reward = 1 + stack/1000 if self.uuid in [winner['uuid'] for winner in winners] else -1 - + stack/1000
        self.rewards.append(reward)
        self.update_policy()

class PolicyValueNetwork(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(20, 64)
        self.fc2 = nn.Linear(64, 128)
        self.fc3 = nn.Linear(128,128)
        self.action_head = nn.Linear(128, 3)
        self.value_head = nn.Linear(128, 1)
        self.raise_head = nn.Linear(128, 1)  # Output for the raise amount
        # Initialize action head to promote equal probability of actions
        self.action_head.weight.data.fill_(0.0)
        self.action_head.bias.data.fill_(0.0)
        # print("PolicyValueNetwork initialized")

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = torch.relu(self.fc3(x))
        action_probs = torch.softmax(self.action_head(x), dim=-1)
        state_values = self.value_head(x)
        raise_amount = torch.sigmoid(self.raise_head(x))  # Sigmoid to keep the raise amount between 0 and 1
        return action_probs, state_values, raise_amount

def setup_ai(model_path=None):
    return PPOPlayer(model_path)
