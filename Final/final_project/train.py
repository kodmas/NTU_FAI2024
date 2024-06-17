import json
from game.game import setup_config, start_poker
from agents.call_player import setup_ai as call_ai
from agents.random_player import setup_ai as random_ai
from agents.console_player import setup_ai as console_ai
from agents.ppo_player import setup_ai as ppo_ai
from baseline1 import setup_ai as baseline1_ai
import random
from tqdm import tqdm
def train_agent(episodes=100, load_model_path=None):
    win = 0
    agent = ppo_ai(load_model_path)
    for episode in tqdm(range(episodes)):
        config = setup_config(max_round=20, initial_stack=1000, small_blind_amount=5)
        config.register_player(name="p1", algorithm=baseline1_ai())
        config.register_player(name="p2", algorithm=agent)

        game_result = start_poker(config)

        print(f"Episode {episode}:")
        print(json.dumps(game_result, indent=4))
        if(game_result["players"][1]["stack"] > 1000):
            win+=1
            print("------------")
            print("This round p2 wins")
            print("------------")

        # Save the model periodically
        if (episode + 1) % 5 == 0:
            agent.save_model(f'checkpoints/ppo_model_checkpoint_{episode + 1}.pth')

    print(f"win_rate = {win}/{episodes}")
# Train for 1000 episodes, optionally loading a pretrained model
train_agent(load_model_path="checkpoints/ppo_model_checkpoint_100.pth")
# train_agent()
