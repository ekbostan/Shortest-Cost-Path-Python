# Skamania II GPS: Off-Trail Pathfinding
*Author: Erol Kaan Bostan

This code repository contains implementations of different pathfinding algorithms in Python for the Skamania II GPS off-trail pathfinding problem. The problem requires finding the most hiker-friendly trail for an arbitrary terrain represented as a 3D surface. The hiker can move with chessboard motion across the XY plane of this surface, with the cost per step defined by the change in height for each step using one of two cost functions. The proposed solution involves using the A* algorithm implemented as a subclass called AStarExp, which uses both the actual cost of the path and an estimate of the remaining distance to the goal node to decide which node to explore next. The repository also includes two other subclasses of the AIModule class, namely StupidAI and Dijkstra's algorithm. Test cases and a map are provided for reference.

