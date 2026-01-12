import numpy as np

class MeanQLearner:
    """Q-learner that tracks the sample-mean value for each action."""

    def __init__(self, actions=("left", "right"), epsilon=0.1, seed=None):
        self.actions = list(actions)
        self.epsilon = float(epsilon)
        self.rng = np.random.default_rng(seed)
        self.q = {action: 0.0 for action in self.actions}
        self.counts = {action: 0 for action in self.actions}

    def choose_action(self):
        if self.rng.random() < self.epsilon:
            return self.rng.choice(self.actions)
        max_q = max(self.q.values())
        best_actions = [a for a, v in self.q.items() if v == max_q]
        return self.rng.choice(best_actions)

    def update(self, action, reward):
        self.counts[action] += 1
        n = self.counts[action]
        self.q[action] += (reward - self.q[action]) / n

class FixedRateQLearner:
    """Q-learner that updates action values with a fixed learning rate."""

    def __init__(self, actions=("left", "right"), eta=0.1, epsilon=0.1, seed=None):
        self.actions = list(actions)
        self.epsilon = float(epsilon)
        self.eta = float(eta)
        self.rng = np.random.default_rng(seed)
        self.q = {action: 0.0 for action in self.actions}
    
    def choose_action(self):
        if self.rng.random() < self.epsilon:
            return self.rng.choice(self.actions)
        max_q = max(self.q.values())
        best_actions = [a for a, v in self.q.items() if v == max_q]
        return self.rng.choice(best_actions)
    
    def update(self, action, reward):
        self.q[action] += (reward - self.q[action]) * self.eta

class PreferenceLearner:
    """Preference-based learner using a softmax policy and reward baseline."""

    def __init__(self, actions=("left", "right"), eta=0.5, seed=None):
        self.actions = list(actions)
        self.eta = float(eta)
        self.rng = np.random.default_rng(seed)
        self.preferences = {action: 0.0 for action in self.actions}
        self.average_reward = 0.0
        self.t = 0
    
    def _get_probabilities(self):
        prefs = np.array([self.preferences[a] for a in self.actions])
        exp_prefs = np.exp(prefs)
        probs = exp_prefs / np.sum(exp_prefs)
        return probs
    
    def choose_action(self):
        probs = self._get_probabilities()
        return self.rng.choice(self.actions, p=probs)
    
    def update(self, action, reward):
        self.t += 1
        self.average_reward += (reward - self.average_reward) / self.t
        probs = self._get_probabilities()

        for i, a in enumerate(self.actions):
            if a == action:
                self.preferences[a] += self.eta * (reward - self.average_reward) * (1 - probs[i])
            else:
                self.preferences[a] -= self.eta * (reward - self.average_reward) * probs[i]