import random
import numpy as np
from collections import defaultdict

class RLTrafficController:
    """
    Q-learning agent with state = (x, y, traffic_bin, weather_bin)
    Actions: 0=up,1=down,2=left,3=right
    """
    def __init__(self, learning_rate=0.1, discount=0.95, exploration_rate=1.0,
                 exploration_decay=0.995, seed=None):
        if seed is not None:
            random.seed(seed)
            np.random.seed(seed)
        self.q_table = defaultdict(lambda: [0.0, 0.0, 0.0, 0.0])
        self.lr = learning_rate
        self.gamma = discount
        self.epsilon = exploration_rate
        self.epsilon_decay = exploration_decay
        self.training_history = []  # list of episode steps

    def _discretize(self, value, bins=3):
        """Convert continuous traffic/weather to bin index (0,1,2)."""
        if value < 0.33:
            return 0
        elif value < 0.66:
            return 1
        else:
            return 2

    def _get_state_key(self, pos, traffic, weather):
        """Create state string including position and environmental factors."""
        t_bin = self._discretize(traffic)
        w_bin = self._discretize(weather)
        return f"{pos[0]},{pos[1]},{t_bin},{w_bin}"

    def _get_action(self, state_key, greedy=False):
        """Epsilon-greedy action selection."""
        if not greedy and random.random() < self.epsilon:
            return random.randint(0, 3)
        q_vals = self.q_table[state_key]
        return np.argmax(q_vals)

    def _update(self, state_key, action, reward, next_state_key):
        """Q-learning update."""
        current_q = self.q_table[state_key][action]
        max_next_q = max(self.q_table[next_state_key])
        td_target = reward + self.gamma * max_next_q
        self.q_table[state_key][action] += self.lr * (td_target - current_q)

    def train(self, env, start=(0,0), goal=(9,9), episodes=500, max_steps=100):
        """
        Train on the given environment (which provides step() method).
        Returns list of steps per episode.
        """
        self.training_history = []
        for ep in range(episodes):
            pos = start
            steps = 0
            total_reward = 0
            while pos != goal and steps < max_steps:
                # Get current environmental factors from env
                traffic = env.get_traffic("stochastic")   # use stochastic during training
                weather = env.get_weather("stochastic")
                state_key = self._get_state_key(pos, traffic, weather)

                action = self._get_action(state_key)

                # Execute action (move)
                if action == 0:    # up
                    new_pos = (pos[0]-1, pos[1])
                elif action == 1:  # down
                    new_pos = (pos[0]+1, pos[1])
                elif action == 2:  # left
                    new_pos = (pos[0], pos[1]-1)
                else:               # right
                    new_pos = (pos[0], pos[1]+1)

                # Check bounds and obstacles
                rows, cols = env.grid_size, env.grid_size
                if (0 <= new_pos[0] < rows and 0 <= new_pos[1] < cols and
                        env.grid[new_pos[0]][new_pos[1]] == 0):
                    pos = new_pos
                    reward = -1  # step cost (aligned with cost function's time component)
                else:
                    # Invalid move: stay put, higher penalty
                    reward = -2

                # Get next state's factors
                next_traffic = env.get_traffic("stochastic")
                next_weather = env.get_weather("stochastic")
                next_state_key = self._get_state_key(pos, next_traffic, next_weather)

                self._update(state_key, action, reward, next_state_key)
                steps += 1
                total_reward += reward

                if pos == goal:
                    # Bonus reward upon reaching goal (aligned with cost negative)
                    reward = 100  # not used in update here, but could be
                    break

            self.training_history.append(steps)
            self.epsilon *= self.epsilon_decay

        return self.training_history

    def greedy_path_length(self, env, start, goal, condition="medium", max_steps=100):
        """
        Use learned greedy policy to navigate from start to goal under given condition.
        Returns steps taken (or max_steps if fails) and whether goal reached.
        """
        pos = start
        steps = 0
        reached = False
        while pos != goal and steps < max_steps:
            traffic = env.get_traffic(condition)
            weather = env.get_weather(condition)
            state_key = self._get_state_key(pos, traffic, weather)
            action = self._get_action(state_key, greedy=True)

            if action == 0:
                new_pos = (pos[0]-1, pos[1])
            elif action == 1:
                new_pos = (pos[0]+1, pos[1])
            elif action == 2:
                new_pos = (pos[0], pos[1]-1)
            else:
                new_pos = (pos[0], pos[1]+1)

            rows, cols = env.grid_size, env.grid_size
            if (0 <= new_pos[0] < rows and 0 <= new_pos[1] < cols and
                    env.grid[new_pos[0]][new_pos[1]] == 0):
                pos = new_pos

            steps += 1

        reached = (pos == goal)
        return steps if reached else max_steps, reached