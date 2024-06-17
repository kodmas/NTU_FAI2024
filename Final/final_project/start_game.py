import json
from game.game import setup_config, start_poker
from agents.call_player import setup_ai as call_ai
from agents.random_player import setup_ai as random_ai
from agents.console_player import setup_ai as console_ai
from agents.rl_player import setup_ai as rl_ai
from agents.mc_player import setup_ai as mc_ai
from agents.mccfr_player import setup_ai as mccfr_ai
from agents.mcs_player import setup_ai as mcs_ai
from baseline4 import setup_ai as baseline4_ai
from tqdm import tqdm
episodes = 5
win = 0

for episode in tqdm(range(episodes)):
    config = setup_config(max_round=20, initial_stack=1000, small_blind_amount=5)
    config.register_player(name="p1", algorithm=baseline4_ai())
    config.register_player(name="p2", algorithm=mcs_ai())

    game_result = start_poker(config)

    print(f"Episode {episode}:")
    print(json.dumps(game_result, indent=4))
    if(game_result["players"][1]["stack"] > 1000):
        win+=1
        print("------------")
        print("This round p2 wins")
        print("------------")


print(f"win_rate = {win}/{episodes}")
