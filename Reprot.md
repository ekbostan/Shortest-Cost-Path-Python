# Skamania II GPS: Off-Trail Pathfinding
*Author: Your Name Here*

## Problem Statement

The Skamania II GPS needs to find the most hiker-friendly trail for an arbitrary terrain represented as a 3D surface discretized into uniformly spaced tiles. The hiker can move with chessboard motion across the XY plane of this surface, with the cost per step defined by the change in height for each step using one of two cost functions:

- Cost function 1: $cost(h_0, h_1) = e^{h_1-h_0} = e^{\Delta h}$
- Cost function 2: $cost(h_0, h_1) = \frac{h_0}{h_1+1}$

Here, $h_0$ is the height of the current tile and $h_1$ is the height of the tile to be moved to. The cost surface can be visualized with the following surfaces:

- Cost function 1: $cost(h_0, h_1) = e^{h_1-h_0} = e^{\Delta h}$
- Cost function 2: $cost(h_0, h_1) = \frac{h_0}{h_1+1}$

The total path cost is defined as the sum of all of the individual step costs.

We plan to implement the A* algorithm for this task, but our team lacks the expertise to code this and design admissible heuristics to ensure that the optimal path is found.

## Proposed Solution

To solve this problem, we propose using the A* algorithm with an admissible heuristic to ensure the optimal path is found. The A* algorithm is a widely used search algorithm in artificial intelligence and is well-suited for pathfinding problems.

To apply the A* algorithm, we first represent the terrain as a graph, where each tile is a node in the graph, and each edge represents a valid move from one tile to another. We can then assign weights to each edge based on the cost function used, and use the A* algorithm to find the path with the lowest total cost.

To design an admissible heuristic, we propose using 2-D distance combined with the height difference between the current node and the goal node as an estimate of the remaining cost. This heuristic is admissible because it never overestimates the true cost of reaching the goal node for the cost function 1. For cost function 2, we propose using the formula:

