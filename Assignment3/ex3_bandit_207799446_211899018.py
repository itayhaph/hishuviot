import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from environment import get_goal_keeper
from agents import MeanQLearner, FixedRateQLearner, PreferenceLearner

def run_match(agent, goal_keeper, kicks_per_match=50):
    """
    Executes a single match of 50 penalty kicks.
    Returns the total reward accumulated by the agent.
    """
    total_reward = 0
    for _ in range(kicks_per_match):
        action = agent.choose_action()
        _, reward = goal_keeper.step(action)
        agent.update(action, reward)
        total_reward += reward
    return total_reward

def tune_agent(agent_class, param_grid, matches_count=1000):
    """
    Generalized function to perform grid search over hyperparameters.
    Handles agents with either one or two hyperparameters.
    """
    param_names = list(param_grid.keys())
    
    if len(param_names) == 2:
        # Tuning for agents with two parameters (e.g., FixedRateQLearner)
        p1_name, p2_name = param_names
        p1_vals, p2_vals = param_grid[p1_name], param_grid[p2_name]
        results = np.zeros((len(p1_vals), len(p2_vals)))
        
        for i, v1 in enumerate(p1_vals):
            for j, v2 in enumerate(p2_vals):
                params = {p1_name: v1, p2_name: v2}
                rewards = [run_match(agent_class(**params), get_goal_keeper('biased')) 
                           for _ in range(matches_count)]
                results[i, j] = np.mean(rewards)
        return results, param_names
    else:
        # Tuning for agents with one parameter (e.g., MeanQLearner)
        p_name = param_names[0]
        p_vals = param_grid[p_name]
        results = np.zeros(len(p_vals))
        
        for i, v in enumerate(p_vals):
            params = {p_name: v}
            rewards = [run_match(agent_class(**params), get_goal_keeper('biased')) 
                       for _ in range(matches_count)]
            results[i] = np.mean(rewards)
        return results, param_names

# Define initial search space for all three agents
agents_to_test = [
    (MeanQLearner, {'epsilon': [0.01, 0.1, 0.2, 0.4]}),
    (FixedRateQLearner, {'eta': [0.01, 0.1, 0.5], 'epsilon': [0.01, 0.1, 0.2]}),
    (PreferenceLearner, {'eta': [0.01, 0.1, 0.5, 0.8]})
]

# Create a figure with 3 subplots for simultaneous display
fig, axes = plt.subplots(1, 3, figsize=(18, 5))
final_results_summary = []

for idx, (agent_class, initial_grid) in enumerate(agents_to_test):
    agent_name = agent_class.__name__
    print(f"Starting Tuning for {agent_name}...")
    
    # Stage 1: Fast preliminary search (100 matches)
    pre_results, p_names = tune_agent(agent_class, initial_grid, matches_count=300)
    
    # Stage 2: Define refined search range around best preliminary values
    refined_grid = {}
    if pre_results.ndim == 2:
        best_idx = np.unravel_index(np.argmax(pre_results), pre_results.shape)
        for i, p_name in enumerate(p_names):
            val = initial_grid[p_name][best_idx[i]]
            refined_grid[p_name] = np.linspace(max(0.001, val - 0.05), min(1.0, val + 0.05), 5)
    else:
        best_val = initial_grid[p_names[0]][np.argmax(pre_results)]
        refined_grid[p_names[0]] = np.linspace(max(0.001, best_val - 0.05), min(1.0, best_val + 0.05), 5)

    # Full Search: Denser grid with 1000 matches per combination
    full_results, _ = tune_agent(agent_class, refined_grid, matches_count=1000)
    ax = axes[idx]
    
    # Plotting logic for subplots
    if full_results.ndim == 2:
        # Render Heatmap for 2-parameter agents
        im = ax.imshow(full_results, cmap='GnBu', interpolation='nearest')
        fig.colorbar(im, ax=ax, label='Avg Reward')
        ax.set_xticks(range(len(refined_grid[p_names[1]])))
        ax.set_xticklabels([f"{x:.3f}" for x in refined_grid[p_names[1]]])
        ax.set_yticks(range(len(refined_grid[p_names[0]])))
        ax.set_yticklabels([f"{x:.3f}" for x in refined_grid[p_names[0]]])
        ax.set_xlabel(p_names[1])
        ax.set_ylabel(p_names[0])
        
        # Mark optimal point with star
        best_f_idx = np.unravel_index(np.argmax(full_results), full_results.shape)
        ax.scatter(best_f_idx[1], best_f_idx[0], color='red', marker='*', s=200, label='Optimal')
        best_params = {p_names[0]: refined_grid[p_names[0]][best_f_idx[0]], 
                       p_names[1]: refined_grid[p_names[1]][best_f_idx[1]]}
    else:
        # Render Line plot for 1-parameter agents
        ax.plot(refined_grid[p_names[0]], full_results, marker='o')
        ax.set_xlabel(p_names[0])
        ax.set_ylabel('Avg Reward')
        best_val_f = refined_grid[p_names[0]][np.argmax(full_results)]
        ax.scatter(best_val_f, np.max(full_results), color='red', marker='*', s=200, label='Optimal')
        best_params = {p_names[0]: best_val_f}

    param_str = ", ".join([f"{k}: {v:.3f}" for k, v in best_params.items()])
    ax.set_title(f'{agent_name}\n{param_str}')
    ax.legend(loc='lower center', bbox_to_anchor=(0.5, -0.25))
    
    # Final Stage: Generate unbiased estimate with fresh environment instances
    unbiased_test = [run_match(agent_class(**best_params), get_goal_keeper('biased')) 
                      for _ in range(1000)]
    final_results_summary.append({
        'Agent': agent_name, 
        'Params': best_params, 
        'Unbiased': np.mean(unbiased_test)
    })

plt.tight_layout()
print("\n" + "="*70)
print(f"{'Agent Class':<20} | {'Optimal Parameters':<30} | {'Unbiased Reward':<15}")
print("-" * 75)
for row in final_results_summary:
    params_str = ", ".join([f"{k}={v:.3f}" for k, v in row['Params'].items()])
    print(f"{row['Agent']:<20} | {params_str:<30} | {row['Unbiased']:<15.3f}")

results_df = pd.DataFrame(final_results_summary)

# Display the table in a neat format
print("\nSection 3.5: Final Summary Table")
print(results_df.to_string(index=False))
# Display all subplots simultaneously in a single window
plt.show()