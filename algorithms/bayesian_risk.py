class BayesianRiskNet:

    def __init__(self):

        self.p_delay = {
            (True, True): 0.95,
            (True, False): 0.70,
            (False, True): 0.60,
            (False, False): 0.10
        }

        self.p_traffic_given_rain = {
            True: 0.85,
            False: 0.30
        }

    def compute_traffic_probability(self, is_raining):
        return self.p_traffic_given_rain[is_raining]

    def infer_delay_probability(self, is_raining, is_traffic):
        return self.p_delay[(is_traffic, is_raining)]

    def get_weather_state(self, iteration):
        return iteration % 5 == 0