class MinimaxAdversary:
    def __init__(self, max_depth=3):
        self.max_depth = max_depth

    def minimax(self, node, depth, alpha, beta, maximizing_player, get_children, evaluate):
        """
        node: current state
        get_children(node) -> list of next states
        evaluate(node) -> heuristic value
        """

        if depth == self.max_depth or not get_children(node):
            return evaluate(node)

        if maximizing_player:
            max_eval = float('-inf')

            for child in get_children(node):
                eval = self.minimax(child, depth + 1, alpha, beta,
                                    False, get_children, evaluate)
                max_eval = max(max_eval, eval)
                alpha = max(alpha, eval)
                if beta <= alpha:
                    break

            return max_eval

        else:
            min_eval = float('inf')

            for child in get_children(node):
                eval = self.minimax(child, depth + 1, alpha, beta,
                                    True, get_children, evaluate)
                min_eval = min(min_eval, eval)
                beta = min(beta, eval)
                if beta <= alpha:
                    break

            return min_eval