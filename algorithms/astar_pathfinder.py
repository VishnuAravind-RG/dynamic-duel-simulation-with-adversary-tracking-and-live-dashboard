import heapq

class AStarPathfinder:
    def __init__(self, grid, terrain_costs=None):
        """
        grid: 2D matrix (0 = free, 1 = obstacle)
        terrain_costs: optional dict {(r,c): cost}
        """
        self.grid = grid
        self.rows = len(grid)
        self.cols = len(grid[0])
        self.terrain_costs = terrain_costs if terrain_costs else {}

    def heuristic(self, a, b):
        """
        Manhattan distance heuristic.
        Safe and admissible for 4-direction grid.
        """
        return abs(a[0] - b[0]) + abs(a[1] - b[1])

    def get_cost(self, node):
        """
        Allows terrain-based movement cost.
        Default cost = 1
        """
        return self.terrain_costs.get(node, 1)

    def reconstruct_path(self, came_from, current):
        path = []
        while current in came_from:
            path.append(current)
            current = came_from[current]
        path.append(current)
        return path[::-1]

    def find_path(self, start, goal):
        open_set = []
        heap_counter = 0  # tie-breaker for heap stability
        heapq.heappush(open_set, (0, heap_counter, start))

        came_from = {}
        g_score = {start: 0}
        f_score = {start: self.heuristic(start, goal)}
        closed_set = set()

        while open_set:
            _, _, current = heapq.heappop(open_set)

            if current in closed_set:
                continue  # Prevent reprocessing

            if current == goal:
                return self.reconstruct_path(came_from, current)

            closed_set.add(current)

            for dx, dy in [(-1,0), (1,0), (0,-1), (0,1)]:
                neighbor = (current[0] + dx, current[1] + dy)

                # Boundary check
                if not (0 <= neighbor[0] < self.rows and 0 <= neighbor[1] < self.cols):
                    continue

                # Obstacle check
                if self.grid[neighbor[0]][neighbor[1]] == 1:
                    continue

                tentative_g = g_score[current] + self.get_cost(neighbor)

                if neighbor not in g_score or tentative_g < g_score[neighbor]:
                    came_from[neighbor] = current
                    g_score[neighbor] = tentative_g
                    f_score[neighbor] = tentative_g + self.heuristic(neighbor, goal)

                    heap_counter += 1
                    heapq.heappush(open_set, (f_score[neighbor], heap_counter, neighbor))

        return []  # No path found