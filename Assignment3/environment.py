from abc import abstractmethod
import numpy as np

class GoalKeeper:
    """
    A goal keeper environment class.
    
    In each turn, the goal keeper predecides
    whether to dive left or right.

    If the kicker shoots in the same direction as the goal keeper,
    the goal keeper saves the goal and the kicker
    receives a reward of -1.

    Otherwise, the kicker scores a goal and receives a reward of +1.
    """

    def __init__(self):
        self.kick_history = []
        self.dive_history = []
        self.total_reward = 0
        
    def step(self, kicker_action):
        """
        Take a step in the environment.

        Parameters:
        kicker_action (str): The action taken by the kicker ('left' or 'right').

        returns:
        goal_keeper_decision (str): The action taken by the goal keeper ('left' or 'right').
        reward (int): The reward received by the kicker (+1 for scoring, -1 for being saved).
        """
        
        assert kicker_action in ['left', 'right'], "kicker_action must be 'left' or 'right'"
        goal_keeper_decision = self.predecide_goal_keeper_action()

        if kicker_action == goal_keeper_decision:
            reward = -1  # Goal keeper saves the goal
        else:
            reward = 1   # Kicker scores a goal
        
        self.total_reward += reward
        self.kick_history.append(kicker_action)
        self.dive_history.append(goal_keeper_decision)
        return goal_keeper_decision, reward
    
    @abstractmethod
    def predecide_goal_keeper_action(self):
        raise NotImplementedError("This method should be overridden by subclasses.")

class Biased_Goal_Keeper(GoalKeeper):
    """
    A biased goal keeper environment class.

    The goal keepers has a probability theta of diving left.
    """
    def __init__(self, theta=0.5, seed=None):
        super().__init__()
        self.theta = theta
        self.rng = np.random.default_rng(seed)
    
    def predecide_goal_keeper_action(self):
        return 'left' if self.rng.random() < self.theta else 'right'
    
class AdversarialGoalKeeper(GoalKeeper):
    def __init__(self, seed=None):
        super().__init__()
        self.rng = np.random.default_rng(seed)
        self.kicker_q = {'left': 0.0, 'right': 0.0}
        self.kicker_counts = {'left': 0, 'right': 0}
        self.action = None

    def predecide_goal_keeper_action(self):
        if self.kicker_q['left'] > self.kicker_q['right']:
            return 'left'
        elif self.kicker_q['right'] > self.kicker_q['left']:
            return 'right'
        else:
            return self.rng.choice(['left', 'right'])

    def update_internal_model(self, action, reward):
        self.kicker_counts[action] += 1
        n = self.kicker_counts[action]
        self.kicker_q[action] += (reward - self.kicker_q[action]) / n

def get_goal_keeper(goal_keeper_type, seed=None):
    """"
    Factory method to get a goal keeper instance based on the index.
    
    args:
    goal_keeper_type (str): The type of goal keeper ('biased', 'unbiased', 'adversarial').

    """

    rng = np.random.default_rng(seed)

    assert isinstance(goal_keeper_type, str), "goal_keeper_type must be a string"
    
    if goal_keeper_type == 'biased':
        return Biased_Goal_Keeper(theta=rng.random()*0.3 + 0.35, seed=seed)  # biased between [0.35, 0.65]
    else:
        return AdversarialGoalKeeper(seed=seed)
    
