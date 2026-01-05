# _____________________________________________________________________ROBOT NAVIGATION ~ RIMJHIM JAIN_____________________________________________________________
# _________________________________________________________________________START OF PROGRAM____________________________________________________________________

# Importing necessary libraries for our Maze Robot project.

import numpy as np # This line imports the NumPy library. NumPy is fundamental for numerical operations in Python, especially for working with arrays (like our maze grid) efficiently. It allows us to perform mathematical operations on entire arrays quickly.
import heapq # This line imports the heapq module. This module provides an implementation of the heap queue algorithm, also known as the priority queue. It's vital for the A* algorithm as it allows us to efficiently retrieve the "next best" node to explore based on its estimated total cost.
import matplotlib.pyplot as plt # This line imports the pyplot module from Matplotlib, which is Python's most popular plotting library. We will use it extensively to create visual representations of our maze and the robot's calculated path.
import matplotlib.colors as mcolors # This line imports the colors module from Matplotlib. This module is specifically used to define and work with custom color maps, which is how we will set the colors of our maze blocks.

# _______________________________________________________________________Maze and A* Algorithm___________________________________________________________________

# This section defines the structure of our maze and implements the intelligent A* pathfinding algorithm
# that the robot will use to navigate through it.

# Define the maze grid using a NumPy array.
# In this 10x10 grid:
#   '0' represents an open cell – a walkable path where the robot can move.
#   '1' represents an obstacle – a wall or blocked cell that the robot cannot pass through.
# The current arrangement is designed with various turns and twists to provide a challenging pathfinding scenario.

maze = np.array([
    [0, 0, 0, 0, 1, 0, 0, 0, 0, 0],
    [1, 1, 0, 0, 1, 0, 1, 1, 1, 0],
    [0, 0, 0, 0, 0, 0, 1, 0, 0, 0],
    [0, 1, 1, 1, 1, 0, 1, 0, 1, 0],
    [0, 0, 0, 0, 0, 0, 0, 0, 1, 0],
    [0, 1, 0, 1, 1, 1, 1, 0, 1, 0],
    [0, 1, 0, 0, 0, 0, 0, 0, 1, 0],
    [0, 1, 1, 1, 1, 1, 1, 1, 1, 0],
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    [0, 1, 1, 1, 1, 1, 1, 1, 1, 0]
])

# Define the starting and ending coordinates for the robot's journey within the maze.
# Coordinates are given as (row, column).

start = (0, 0) # The robot begins its journey at the top-left corner of the maze.
end = (9, 9)   # The robot's ultimate goal is to reach the bottom-right corner of the maze.

def heuristic(a, b):
    """
    This is our 'heuristic' function, often denoted as h(n) in pathfinding algorithms.
    It provides an *estimate* of the cost (or distance) from any current cell 'a'
    to the designated goal cell 'b'. For our grid-based maze, where movement is restricted
    to horizontal and vertical steps (no diagonals), the Manhattan distance is an ideal heuristic.
    It calculates the sum of the absolute differences in their row and column coordinates,
    effectively telling us the minimum number of grid steps required to get from 'a' to 'b'
    if there were no obstacles.
    """
    return abs(a[0] - b[0]) + abs(a[1] - b[1])

def astar(maze, start, end):
    """
    This function implements the A* (A-star) search algorithm. A* is a highly efficient
    and widely used pathfinding algorithm that finds the shortest path between a starting
    node and a target node in a graph (or, in our case, a grid-based maze).
    It works by intelligently balancing two factors:
    1. The actual cost already incurred from the start node to the current node (g_score).
    2. An estimated cost from the current node to the end node (h_score, derived from our heuristic).
    The algorithm prioritizes nodes with the lowest combined estimated cost (f_score = g_score + h_score).
    """
    # Define the four possible cardinal movements a robot can make within the maze:
    # (delta_row, delta_column) for moving Right, Down, Left, and Up respectively.
    
    movements = [(0, 1), (1, 0), (0, -1), (-1, 0)]

    def is_valid_move(x, y):
        """
        This is a helper function designed to check if a potential move to a new cell (x, y)
        is permissible according to the maze rules. A move is considered valid if:
        1. The new row (x) and column (y) coordinates are within the defined boundaries of the maze grid.
        2. The cell at these (x, y) coordinates is not an obstacle (i.e., its value in the 'maze' array is '0').
        """
        return 0 <= x < maze.shape[0] and \
               0 <= y < maze.shape[1] and maze[x, y] == 0

    # `open_set`: This is a priority queue (implemented using `heapq`). It holds all the nodes
    # that have been discovered but not yet fully evaluated. Nodes are stored as tuples:
    # (f_score, g_score, position), and `heapq` ensures that the node with the lowest `f_score`
    # (most promising) is always at the top, ready for the next evaluation.
    
    open_set = []
    
    # We initialize the `open_set` by pushing the `start` node.
    # Its actual cost from start (g_score) is 0. Its estimated total cost (f_score)
    # is just its heuristic distance to the `end` node.
    
    heapq.heappush(open_set, (0 + heuristic(start, end), 0, start))

    # `came_from`: This dictionary is crucial for reconstructing the optimal path.
    # For each node that the algorithm processes, `came_from[node]` stores the node
    # that immediately preceded it on the most efficient path found so far.
    
    came_from = {}

    # `g_score`: This dictionary stores the *actual* cost (number of steps taken)
    # from the `start` node to each discovered node. We initialize the `start` node's
    # `g_score` to 0, as it costs nothing to reach itself.
    
    g_score = {start: 0}

    # `f_score`: This dictionary stores the estimated total cost for each discovered node.
    # It's the sum of the `g_score` (actual cost from start) and the `heuristic` (estimated cost to end).
    # For the `start` node, it's initially just its heuristic value to the `end`.
    
    f_score = {start: heuristic(start, end)}

    # This is the main loop of the A* algorithm. It continues as long as there are
    # nodes left to explore in the `open_set`.
    
    while open_set:
        
        # Retrieve the node with the lowest `f_score` from the `open_set`.
        # This is the "current" node – the most promising one to expand next.
        # `_f_val` (f_score) is extracted but not always used, `current_g` is the g_score,
        # and `current` holds the (row, column) position of the node.
        
        _f_val, current_g, current = heapq.heappop(open_set)

        # Check if the `current` node is the `end` goal. If it is, we've successfully
        # found the shortest path, and we can now reconstruct it.
        
        if current == end:
            path = [] # Initialize an empty list to build the path.
            
            # Reconstruct the path by backtracking from the `end` node to the `start` node
            # using the `came_from` dictionary.
            while current in came_from:
                path.append(current)          # Add the current node to the path.
                current = came_from[current]  # Move to its predecessor.
            path.append(start)                # Add the `start` node, which is the final step in reconstruction.
            return path[::-1]                 # Reverse the path list to get it in correct order (start to end).

        # If the `end` hasn't been reached, explore all possible `movements` (neighbors)
        # from the `current` node.
        
        for move in movements:
            
            # Calculate the coordinates of the `neighbor` cell.
            
            neighbor = (current[0] + move[0], current[1] + move[1])

            # Check if the `neighbor` is a valid and walkable cell within the maze.
            # The `*neighbor` unpacks the (row, col) tuple into two separate arguments for `is_valid_move`.
            
            if is_valid_move(*neighbor):
                
                # Calculate the `tentative_g_score`: the cost to reach this `neighbor`
                # if we were to move from the `current` node (each step costs 1).
                
                tentative_g_score = current_g + 1

                # Compare the `tentative_g_score` with the previously known `g_score` for this `neighbor`.
                # If this is a new neighbor, or if we've found a shorter path to it:
                
                if neighbor not in g_score or tentative_g_score < g_score[neighbor]:
                    came_from[neighbor] = current          # Update `came_from` to reflect this shorter path.
                    g_score[neighbor] = tentative_g_score  # Update the actual cost to reach `neighbor`.
                    
                    # Update the `f_score` for the `neighbor` with its new `g_score` and heuristic.
                    
                    f_score[neighbor] = g_score[neighbor] + heuristic(neighbor, end)
                    
                    # Add (or re-add with updated priority) the `neighbor` to the `open_set`
                    # so it can be evaluated later.
                    
                    heapq.heappush(open_set, (f_score[neighbor], g_score[neighbor], neighbor))

    # If the `while` loop finishes and the `end` node was never reached (meaning `open_set` became empty),
    # it indicates that no path exists from `start` to `end` in this maze.
    
    return None # Return `None` to signify that no path was found.

# _______________________________________________________________Visualization Code (using Matplotlib)__________________________________________________________

# This section of the code is responsible for drawing the maze, the robot's start and end points,
# and the calculated path with directional indicators using Matplotlib.

def visualize_maze_and_path(maze, start, end, path):
    """
    This function creates a static visual representation of the maze.
    It displays the maze grid, highlights the start and end points,
    draws the entire found path, and includes directional arrows along that path.
    """
    
    # Define the custom colormap for the maze blocks.
    # The first color in the list corresponds to '0' (walkable path).
    # The second color in the list corresponds to '1' (obstacle/wall).
    
    cmap = mcolors.ListedColormap(['aquamarine', 'mediumpurple']) # Example: Aquamarine for paths, Medium Purple for obstacles.
   
    # Create the main figure and a single drawing area (subplot) for our plot.
    # `figsize=(8, 8)` sets the width and height of the plot in inches, making it suitable for our 10x10 maze.
    
    fig, ax = plt.subplots(figsize=(8, 8))

    # Display the maze grid as an image on the axes.
    # `cmap` applies our custom colors to the maze cells.
    # `origin='upper'` ensures that the (0,0) coordinate is at the top-left, matching array indexing.
    # `extent` correctly aligns the cell boundaries with the grid lines we'll draw next.
    
    ax.imshow(maze, cmap=cmap, origin='upper', extent=[-0.5, maze.shape[1] - 0.5, maze.shape[0] - 0.5, -0.5])

    # Draw grid lines to clearly delineate each individual cell in the maze.
    # `set_xticks` and `set_yticks` with `minor=True` place ticks at half-unit intervals
    # (e.g., -0.5, 0.5, 1.5), which precisely correspond to the boundaries between cells.
    
    ax.set_xticks(np.arange(-0.5, maze.shape[1], 1), minor=True)
    ax.set_yticks(np.arange(-0.5, maze.shape[0], 1), minor=True)
    ax.grid(which='minor', color='gray', linestyle='-', linewidth=2) # Draw the grid lines in gray.
    ax.tick_params(which='minor', size=0) # Hide the small tick marks themselves, keeping only the grid lines.

    # Mark the start and end points on the maze with distinct, large markers.
    # `ax.plot()` is used to place markers. Matplotlib expects (x, y) coordinates,
    # so we use (column, row) from our (row, column) tuples.
    # 'go' means a green circle marker, 'ro' means a red circle marker.
    # `markersize` controls the size of these circles.
    
    ax.plot(start[1], start[0], 'go', markersize=15, label='Start') # Green circle for the starting point.
    ax.plot(end[1], end[0], 'ro', markersize=15, label='End')     # Red circle for the ending point.

    # This block of code will only execute if a path was successfully found by the A* algorithm.
    
    if path:
        # Extract the X (column) and Y (row) coordinates from the sequence of points in the `path`.
        
        path_y = [p[0] for p in path]
        path_x = [p[1] for p in path]
        
        # Draw the entire calculated path as a dashed blue line.
        # `b--` specifies a blue dashed line. `linewidth` controls its thickness.
        
        ax.plot(path_x, path_y, 'b--', linewidth=1, label='Full Path')

        # Prepare lists to store the starting coordinates (x, y) and the direction vectors (u, v)
        # for each small directional arrow we want to draw along the path.
        
        arrow_x = [] # X-coordinates of where each arrow starts.
        arrow_y = [] # Y-coordinates of where each arrow starts.
        arrow_u = [] # X-component of the arrow's direction (how much it moves horizontally).
        arrow_v = [] # Y-component of the arrow's direction (how much it moves vertically).

        # Iterate through each segment of the path (from one cell to the next)
        # to determine the direction for each arrow.
        
        for i in range(len(path) - 1):
            p1 = path[i]       # The current cell's coordinates.
            p2 = path[i + 1]   # The next cell's coordinates in the path.
            arrow_x.append(p1[1])        # The arrow starts at the X (column) of the current cell.
            arrow_y.append(p1[0])        # The arrow starts at the Y (row) of the current cell.
            arrow_u.append(p2[1] - p1[1]) # Calculate the horizontal change (p2_col - p1_col).
            arrow_v.append(p2[0] - p1[0]) # Calculate the vertical change (p2_row - p1_row).

        # Draw all the directional arrows using `ax.quiver`.
        # We're setting the color back to 'purple' as per your request for the "old direction color"!
        # `angles='xy', scale_units='xy', scale=1` ensures arrows point correctly and stretch across one cell.
        # `width`, `headwidth`, `headlength` are adjusted to make the arrows appear smaller and neat.
        # `zorder=2` ensures the arrows are drawn on top of other elements like the path line.
        
        ax.quiver(arrow_x, arrow_y, arrow_u, arrow_v,
                  color='purple', angles='xy', scale_units='xy', scale=1,
                  width=0.008, headwidth=3, headlength=4, label='Direction', zorder=2)

    # Set the overall title for the plot and labels for the X and Y axes.
    
    ax.set_title("Maze Navigation with Path Directions")
    ax.set_xlabel("Column (X-coordinate)")
    ax.set_ylabel("Row (Y-coordinate)")
    ax.legend() # Display the legend, which shows what the Start, End, Path, and Direction indicators mean.
    
    # Invert the Y-axis. By default, Matplotlib plots (0,0) at the bottom-left.
    # Inverting it makes (0,0) at the top-left, which matches how we typically
    # index 2D arrays (like our maze: row 0 is the top row).
    
    plt.gca().invert_yaxis()

    # Finally, display the generated plot window. This command makes the visualization appear on your screen.
    
    plt.show()

# ______________________________________________________________ Run the A* algorithm and visualize____________________________________________________________

# This section serves as the main execution block of our program. It orchestrates
# the pathfinding and visualization steps.

# First, we call our `astar` function to find the shortest path within the defined maze
# from the `start` point to the `end` point. The result (the path or `None` if no path)
# is stored in the `path` variable.

path = astar(maze, start, end)

# Check if the `astar` algorithm successfully returned a path.

if path:
    
    # If a path was found, print the sequence of coordinates for the user.
    
    print("Path found:", path)

    # Then, call our visualization function to display the maze with the found path and arrows.
    
    visualize_maze_and_path(maze, start, end, path)
else:

    # If `astar` returned `None`, it means no accessible path exists between the start and end points.
    
    print("No path found")

    # Even if no path is found, we still call the visualization function to show the maze layout,
    # but without any path or arrows on it.
    
    visualize_maze_and_path(maze, start, end, None)

# __________________________________________________________________________END OF PROGRAM___________________________________________________________________
