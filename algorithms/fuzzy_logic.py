class FuzzyUrgencyController:

    def triangular(self, x, a, b, c):
        """
        Triangular membership function.
        """
        if x <= a or x >= c:
            return 0
        elif x == b:
            return 1
        elif x < b:
            return (x - a) / (b - a)
        else:
            return (c - x) / (c - b)

    def fuzzify_distance(self, dist, max_dist):
        ratio = dist / max_dist

        near = self.triangular(ratio, 0.0, 0.0, 0.5)
        medium = self.triangular(ratio, 0.2, 0.5, 0.8)
        far = self.triangular(ratio, 0.5, 1.0, 1.0)

        return near, medium, far

    def fuzzify_traffic(self, traffic_prob):
        low = self.triangular(traffic_prob, 0.0, 0.0, 0.5)
        medium = self.triangular(traffic_prob, 0.2, 0.5, 0.8)
        high = self.triangular(traffic_prob, 0.5, 1.0, 1.0)

        return low, medium, high

    def compute_speed_multiplier(self, dist, max_dist, traffic_prob):
        d_near, d_medium, d_far = self.fuzzify_distance(dist, max_dist)
        t_low, t_medium, t_high = self.fuzzify_traffic(traffic_prob)

        # Rule Base
        rule_fast = max(
            min(d_far, t_low),
            min(d_medium, t_low)
        )

        rule_normal = max(
            min(d_medium, t_medium),
            min(d_far, t_medium)
        )

        rule_slow = max(
            min(d_near, t_high),
            min(d_near, t_medium)
        )

        # Output values
        speed_fast = 1.5
        speed_normal = 1.0
        speed_slow = 0.5

        numerator = (
            rule_fast * speed_fast +
            rule_normal * speed_normal +
            rule_slow * speed_slow
        )

        denominator = rule_fast + rule_normal + rule_slow + 1e-6

        speed = numerator / denominator

        return max(0.5, min(1.5, speed))