\documentclass{article}

% Packages for math and graphics
\usepackage{amsmath, amssymb}
% Package for color
\usepackage{xcolor}

% Define colors for cost surfaces
\definecolor{cost1}{RGB}{239, 150, 150}
\definecolor{cost2}{RGB}{150, 239, 150}

\begin{document}

\title{Skamania II GPS: Off-Trail Pathfinding}
\author{Your Name Here}
\maketitle

\section{Problem Statement}

The Skamania II GPS needs to find the most hiker-friendly trail for an arbitrary terrain represented as a 3D surface discretized into uniformly spaced tiles. The hiker can move with chessboard motion across the XY plane of this surface, with the cost per step defined by the change in height for each step using one of two cost functions:

\begin{itemize}
\item Cost function 1: $cost(h_0, h_1) = e^{h_1-h_0} = e^{\Delta h}$
\item Cost function 2: $cost(h_0, h_1) = \frac{h_0}{h_1+1}$
\end{itemize}

Here, $h_0$ is the height of the current tile and $h_1$ is the height of the tile to be moved to. The cost surface can be visualized with the following surfaces:

\begin{itemize}
\item Cost function 1: $cost(h_0, h_1) = e^{h_1-h_0} = e^{\Delta h}$
\item Cost function 2: $cost(h_0, h_1) = \frac{h_0}{h_1+1}$
\end{itemize}

The total path cost is defined as the sum of all of the individual step costs.

We plan to implement the A* algorithm for this task, but our team lacks the expertise to code this and design admissible heuristics to ensure that the optimal path is found.

\section{Proposed Solution}

To solve this problem, we propose using the A* algorithm with an admissible heuristic to ensure the optimal path is found. The A* algorithm is a widely used search algorithm in artificial intelligence and is well-suited for pathfinding problems.

To apply the A* algorithm, we first represent the terrain as a graph, where each tile is a node in the graph, and each edge represents a valid move from one tile to another. We can then assign weights to each edge based on the cost function used, and use the A* algorithm to find the path with the lowest total cost.

To design an admissible heuristic, we propose using 2-D distance combined with the height difference between the current node and the goal node as an estimate of the remaining cost. This heuristic is admissible because it never overestimates the true cost of reaching the goal node for the cost function 1. For cost function 2 , we propose using To design an admissible heuristic, we propose using the formula: h=0.4⋅distance(MAX(horiginal​),hvertical​)

where $h_{\text{original}}$ is the height of the current node, $h_{\text{vertical}}$ is the height of the goal node, and $\text{MAX}(h_{\text{original}}, h_{\text{vertical}})$ is the maximum height between the two nodes. The \text{distance} function is the Euclidean distance between the two nodes. This heuristic is admissible because it never overestimates the true cost of reaching the goal node, and is consistent because it satisfies the triangle inequality. We can use this heuristic in combination with the A* algorithm to find the optimal path on the terrain.
\end{document}