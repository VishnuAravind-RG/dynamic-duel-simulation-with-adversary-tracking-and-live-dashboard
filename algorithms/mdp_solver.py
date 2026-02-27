import numpy as np

class MDPPolicyIterator:
    def __init__(self, grid, goal, discount=0.9, noise=0.2, step_cost=-1):
        self.grid = grid
        self.rows = len(grid)
        self.cols = len(grid[0])
        self.goal = goal
        self.discount = discount
        self.noise = noise
        self.step_cost = step_cost

        self.values = np.zeros((self.rows, self.cols))
        self.policy = np.full((self.rows, self.cols), None)

        self.actions = [(-1,0), (1,0), (0,-1), (0,1)]

    def is_valid(self, r, c):
        return 0 <= r < self.rows and 0 <= c < self.cols and self.grid[r][c] == 0

    def reward(self, r, c):
        if (r, c) == self.goal:
            return 100
        if self.grid[r][c] == 1:
            return -50
        return self.step_cost

    def value_iteration(self, tolerance=1e-4, max_iterations=1000):
        for _ in range(max_iterations):
            delta = 0
            new_values = np.copy(self.values)

            for r in range(self.rows):
                for c in range(self.cols):

                    if not self.is_valid(r, c):
                        continue

                    if (r, c) == self.goal:
                        new_values[r, c] = self.reward(r, c)
                        continue

                    action_values = []

                    for dr, dc in self.actions:
                        expected_value = 0

                        # Intended move
                        intended = (r + dr, c + dc)
                        if self.is_valid(*intended):
                            expected_value += (1 - self.noise) * self.values[intended]
                        else:
                            expected_value += (1 - self.noise) * self.values[r, c]

                        # Slips
                        slip_prob = self.noise / (len(self.actions) - 1)
                        for sr, sc in self.actions:
                            if (sr, sc) != (dr, dc):
                                slipped = (r + sr, c + sc)
                                if self.is_valid(*slipped):
                                    expected_value += slip_prob * self.values[slipped]
                                else:
                                    expected_value += slip_prob * self.values[r, c]

                        total = self.reward(r, c) + self.discount * expected_value
                        action_values.append(total)

                    best_value = max(action_values)
                    new_values[r, c] = best_value
                    delta = max(delta, abs(best_value - self.values[r, c]))

            self.values = new_values

            if delta < tolerance:
                break

        self.extract_policy()

    def extract_policy(self):
        for r in range(self.rows):
            for c in range(self.cols):

                if not self.is_valid(r, c) or (r, c) == self.goal:
                    continue

                action_values = []

                for dr, dc in self.actions:
                    intended = (r + dr, c + dc)
                    if self.is_valid(*intended):
                        val = self.values[intended]
                    else:
                        val = self.values[r, c]

                    action_values.append(val)

                best_idx = np.argmax(action_values)
                self.policy[r, c] = self.actions[best_idx]

    def get_policy(self):
        return self.policy