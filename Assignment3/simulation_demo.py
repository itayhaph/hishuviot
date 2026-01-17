try:
    from tqdm import tqdm
except ImportError:  # pragma: no cover - optional dependency
    tqdm = None

from agents import MeanQLearner, FixedRateQLearner, PreferenceLearner
from environment import get_goal_keeper

GOAL_KEEPERS = ["adverserial"]
AGENTS_TO_TEST = [
    (MeanQLearner, {"epsilon": 0.045}),
    (FixedRateQLearner, {"eta": 0.075, "epsilon": 0.075}),
    (PreferenceLearner, {"eta": 0.85})
]

def run_match(agent, goal_keeper, kicks_per_match=50):
    total_reward = 0
    for _ in range(kicks_per_match):
        action = agent.choose_action()
        _, reward = goal_keeper.step(action)
        agent.update(action, reward)
        total_reward += reward
    return total_reward

def main(matches=12000, kicks_per_match=50):
    for goal_keeper_type in GOAL_KEEPERS:
        print(f"\n--- Testing vs. {goal_keeper_type} Goal Keeper ---")
        
        # Iterate over each agent type
        for agent_class, params in AGENTS_TO_TEST:
            results = []
            agent_name = agent_class.__name__
            
            # Run simulation matches
            for _ in tqdm(range(matches), desc=f"Simulating {agent_name}"):
                # Create a fresh environment for each match
                goal_keeper = get_goal_keeper(goal_keeper_type)
                # Initialize the agent with its specific optimal parameters
                agent = agent_class(**params)
                
                result = run_match(agent, goal_keeper, kicks_per_match=kicks_per_match)
                results.append(result)
            
            avg_reward = sum(results) / matches
            print(f"Agent: {agent_name} | Average Reward: {avg_reward:.3f}")

if __name__ == "__main__":
    main()
