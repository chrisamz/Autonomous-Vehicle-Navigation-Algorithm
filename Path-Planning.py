# path_planning.py

"""
Path Planning Module for Autonomous Vehicle Navigation

This module contains the implementation of path planning algorithms
to ensure safe navigation and real-time obstacle avoidance.

Techniques Used:
- A* Algorithm
- Rapidly-exploring Random Trees (RRT)
- Dynamic Window Approach (DWA)
"""

import numpy as np
import heapq
import matplotlib.pyplot as plt

class AStar:
    def __init__(self, grid, start, goal):
        """
        Initialize the A* algorithm.

        :param grid: 2D array, grid map of the environment
        :param start: tuple, starting point (x, y)
        :param goal: tuple, goal point (x, y)
        """
        self.grid = grid
        self.start = start
        self.goal = goal
        self.open_list = []
        self.closed_list = set()
        self.came_from = {}
        self.g_score = {start: 0}
        self.f_score = {start: self.heuristic(start, goal)}
        heapq.heappush(self.open_list, (self.f_score[start], start))

    def heuristic(self, a, b):
        """
        Heuristic function for A* (Manhattan distance).

        :param a: tuple, point (x, y)
        :param b: tuple, point (x, y)
        :return: int, heuristic value
        """
        return abs(a[0] - b[0]) + abs(a[1] - b[1])

    def get_neighbors(self, node):
        """
        Get neighbors of a node.

        :param node: tuple, point (x, y)
        :return: list of tuples, neighboring points
        """
        neighbors = [
            (node[0] - 1, node[1]),
            (node[0] + 1, node[1]),
            (node[0], node[1] - 1),
            (node[0], node[1] + 1)
        ]
        valid_neighbors = [n for n in neighbors if 0 <= n[0] < len(self.grid) and 0 <= n[1] < len(self.grid[0]) and self.grid[n[0]][n[1]] == 0]
        return valid_neighbors

    def reconstruct_path(self, current):
        """
        Reconstruct the path from start to goal.

        :param current: tuple, current point
        :return: list of tuples, path from start to goal
        """
        total_path = [current]
        while current in self.came_from:
            current = self.came_from[current]
            total_path.append(current)
        return total_path[::-1]

    def search(self):
        """
        Perform the A* search algorithm.

        :return: list of tuples, path from start to goal or None if no path found
        """
        while self.open_list:
            _, current = heapq.heappop(self.open_list)
            if current == self.goal:
                return self.reconstruct_path(current)
            self.closed_list.add(current)
            for neighbor in self.get_neighbors(current):
                if neighbor in self.closed_list:
                    continue
                tentative_g_score = self.g_score[current] + 1
                if neighbor not in self.g_score or tentative_g_score < self.g_score[neighbor]:
                    self.came_from[neighbor] = current
                    self.g_score[neighbor] = tentative_g_score
                    self.f_score[neighbor] = self.g_score[neighbor] + self.heuristic(neighbor, self.goal)
                    heapq.heappush(self.open_list, (self.f_score[neighbor], neighbor))
        return None

if __name__ == "__main__":
    # Example usage
    grid = np.array([
        [0, 1, 0, 0, 0, 0],
        [0, 1, 0, 1, 1, 0],
        [0, 0, 0, 0, 1, 0],
        [0, 1, 1, 0, 0, 0],
        [0, 0, 0, 1, 0, 0]
    ])
    start = (0, 0)
    goal = (4, 5)
    astar = AStar(grid, start, goal)
    path = astar.search()
    
    if path:
        print("Path found:", path)
        # Visualization
        for (x, y) in path:
            grid[x][y] = 2
        plt.imshow(grid, cmap='gray')
        plt.show()
    else:
        print("No path found")

