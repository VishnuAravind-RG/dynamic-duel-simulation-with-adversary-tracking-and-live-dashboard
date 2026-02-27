import numpy as np
import random

class RLTrafficController:
    def __init__(self, state_size=4, action_size=3):
        self.q_table = {}
        self.learning_rate = 0.1
        self.discount = 0.95
        self.exploration_rate = 1.0
        self.exploration_decay = 0.995
        self.action_size = action_size
        
    def get_state(self, traffic_prob, distance, time_of_day, weather):
        """Discretize continuous values into state tuple"""
        # traffic_prob: 0=low, 1=medium, 2=high
        t_prob = 0 if traffic_prob < 0.3 else 1 if traffic_prob < 0.7 else 2
        
        # distance: 0=near, 1=medium, 2=far
        dist = 0 if distance < 0.3 else 1 if distance < 0.7 else 2
        
        # time_of_day: 0=morning, 1=afternoon, 2=evening/night
        time_slot = 0 if 6 <= time_of_day < 12 else 1 if 12 <= time_of_day < 18 else 2
        
        # weather: 0=clear, 1=rain/fog
        weather_state = 0 if weather in ["Clear", "Sunny"] else 1
        
        return (t_prob, dist, time_slot, weather_state)
    
    def get_action(self, state):
        """Epsilon-greedy action selection"""
        if random.random() < self.exploration_rate:
            return random.randint(0, self.action_size - 1)  # Explore
        
        # Exploit - get best action from Q-table
        if state in self.q_table:
            return np.argmax(self.q_table[state])
        return 0  # Default action if state unseen
    
    def update_q(self, state, action, reward, next_state):
        """Q-learning update rule"""
        if state not in self.q_table:
            self.q_table[state] = [0] * self.action_size
        
        if next_state not in self.q_table:
            self.q_table[next_state] = [0] * self.action_size
        
        # Q-learning formula
        current_q = self.q_table[state][action]
        next_max_q = max(self.q_table[next_state])
        
        new_q = current_q + self.learning_rate * (reward + self.discount * next_max_q - current_q)
        self.q_table[state][action] = new_q
        
        # Decay exploration rate
        self.exploration_rate *= self.exploration_decay
    
    def get_best_action_for_current(self, traffic_prob, distance, time_of_day, weather):
        """Convenience method to get best action for current conditions"""
        state = self.get_state(traffic_prob, distance, time_of_day, weather)
        
        if state in self.q_table:
            return np.argmax(self.q_table[state])
        return 2  # Default to cautious action if unseen
    
    def get_q_stats(self):
        """Return statistics about the Q-table"""
        if not self.q_table:
            return {
                "size": 0,
                "avg_q": 0,
                "max_q": 0,
                "min_q": 0
            }
        
        all_q_values = [q for values in self.q_table.values() for q in values]
        return {
            "size": len(self.q_table),
            "avg_q": np.mean(all_q_values),
            "max_q": np.max(all_q_values),
            "min_q": np.min(all_q_values)
        }