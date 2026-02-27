import numpy as np

class HMMTrafficPredictor:
    def __init__(self, states, observations, start_prob, trans_prob, emit_prob):
        self.states = states
        self.observations = observations

        self.start_prob = np.array(start_prob)
        self.trans_prob = np.array(trans_prob)
        self.emit_prob = np.array(emit_prob)

        # Precompute observation index lookup (O(1))
        self.obs_index = {obs: i for i, obs in enumerate(observations)}

    def forward_algorithm(self, obs_seq):
        """
        Log-space forward algorithm to prevent underflow.
        Returns normalized probability distribution over states.
        """
        T = len(obs_seq)
        N = len(self.states)

        log_start = np.log(self.start_prob + 1e-12)
        log_trans = np.log(self.trans_prob + 1e-12)
        log_emit = np.log(self.emit_prob + 1e-12)

        alpha = np.zeros((T, N))

        # Initialization
        obs_idx = self.obs_index[obs_seq[0]]
        alpha[0] = log_start + log_emit[:, obs_idx]

        # Recursion
        for t in range(1, T):
            obs_idx = self.obs_index[obs_seq[t]]
            for j in range(N):
                alpha[t, j] = np.logaddexp.reduce(
                    alpha[t-1] + log_trans[:, j]
                ) + log_emit[j, obs_idx]

        # Convert back from log space
        final_log_probs = alpha[T-1]
        max_log = np.max(final_log_probs)
        probs = np.exp(final_log_probs - max_log)
        probs /= np.sum(probs)

        return probs