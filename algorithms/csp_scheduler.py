class CSPScheduler:
    def __init__(self, tasks, domains, constraints):
        """
        tasks: list of variables
        domains: dict {task: [possible_values]}
        constraints: list of constraint functions
        """
        self.tasks = tasks
        self.domains = domains
        self.constraints = constraints

    def is_consistent(self, task, value, assignment):
        """
        Check all constraints against current partial assignment.
        """
        for constraint in self.constraints:
            if not constraint(task, value, assignment):
                return False
        return True

    def select_unassigned_variable(self, assignment):
        """
        MRV Heuristic:
        Choose variable with minimum remaining legal values.
        """
        unassigned = [t for t in self.tasks if t not in assignment]

        # Count valid domain values for each unassigned variable
        mrv_task = min(
            unassigned,
            key=lambda t: sum(
                1 for v in self.domains[t]
                if self.is_consistent(t, v, assignment)
            )
        )
        return mrv_task

    def forward_check(self, task, value, assignment):
        """
        Remove inconsistent values from neighboring domains.
        Returns a backup of pruned values for restoration.
        """
        pruned = {}

        for other in self.tasks:
            if other not in assignment and other != task:
                pruned[other] = []
                for v in self.domains[other]:
                    if not self.is_consistent(other, v, {**assignment, task: value}):
                        pruned[other].append(v)

                # Remove inconsistent values
                for v in pruned[other]:
                    self.domains[other].remove(v)

                # Failure condition
                if not self.domains[other]:
                    return None

        return pruned

    def restore_domains(self, pruned):
        """
        Restore domains after backtracking.
        """
        for task, values in pruned.items():
            self.domains[task].extend(values)

    def backtrack(self, assignment={}):
        if len(assignment) == len(self.tasks):
            return assignment

        task = self.select_unassigned_variable(assignment)

        for value in list(self.domains[task]):
            if self.is_consistent(task, value, assignment):
                assignment[task] = value

                pruned = self.forward_check(task, value, assignment)

                if pruned is not None:
                    result = self.backtrack(assignment)
                    if result:
                        return result

                # Restore after failure
                if pruned:
                    self.restore_domains(pruned)

                del assignment[task]

        return None