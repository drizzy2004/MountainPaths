import numpy as np
import sys
from heapq import heappush, heappop
import matplotlib.pyplot as plt

# Read the elevation data from file
def read_elevation_data(file_path, num_rows, num_cols):
    grid = [[0] * num_cols for _ in range(num_rows)]

    with open(file_path) as f:
        for i in range(num_rows):
            line = f.readline().strip()
            elevations = list(map(int, line.split()))

            for j in range(num_cols):
                grid[i][j] = elevations[j]

    return grid

# Find the minimum and maximum values in the grid
def find_min_max(grid):
    min_val = float("inf")
    max_val = float("-inf")

    for i in range(len(grid)):
        for j in range(len(grid[0])):
            if grid[i][j] < min_val:
                min_val = grid[i][j]

            if grid[i][j] > max_val:
                max_val = grid[i][j]

    return min_val, max_val

# Example: Draw a path by changing color along a specific row
def draw_path_greedy(grid, start_row):
    num_cols = len(grid[0])
    num_rows = len(grid)
    current_row = start_row
    path = [(start_row, 0)]
    total_cost = 0

    for col in range(num_cols - 1):
        next_col = col + 1

        # Calculate elevation change to move to the next cell
        next_col = col + 1

        # By calculating these 3 different possible moves, we use a greedy algorithm to choose the smallest difference in
        # elevation and make the move (then highlighting a different color to make move visible on map)

        # This up_change should be the difference in elevation from current cell and cell right above
        up_change = abs(grid[current_row - 1][next_col] - grid[current_row][col]) if current_row > 0 else float('inf')

        # This forward_change should be the difference in elevation from current cell and cell right in front
        forward_change = abs(grid[current_row][next_col] - grid[current_row][col])

        # This down_change should be the difference in elevation from current cell and cell right underneath
        down_change = abs(
            grid[current_row + 1][next_col] - grid[current_row][col]) if current_row < num_rows - 1 else float('inf')

        # Move to the cell with the lowest elevation change
        if up_change < forward_change and up_change < down_change:
            current_row -= 1  # Move up
            total_cost += up_change
        elif down_change < forward_change and down_change <= up_change:
            current_row += 1  # Move down
            total_cost += down_change
        else:
            total_cost += forward_change

        path.append((current_row, next_col))

    print("Total cost using Greedy:", total_cost)
    return path


def draw_path_dijkstra(grid, start_row):
    num_cols = len(grid[0])
    num_rows = len(grid)
    cost = [[float("inf")] * num_cols for _ in range(num_rows)]
    cost[start_row][0] = 0
    pq = []
    heappush(pq, (0, start_row, 0))

    # To store the path
    path_from = {}

    while pq:
        current_cost, current_row, current_col = heappop(pq)

        # If we reach the last column, reconstruct the path
        if current_col == num_cols - 1:
            path = reconstruct(path_from, start_row, current_row, current_col)
            print("Total cost using Dijkstra:", current_cost)
            return path

        # Explore neighboring cells
        for move_row in [current_row - 1, current_row, current_row + 1]:
            next_col = current_col + 1

            if 0 <= move_row < num_rows and next_col < num_cols:
                elevation_change = abs(grid[move_row][next_col] - grid[current_row][current_col])
                new_cost = current_cost + elevation_change

                if new_cost < cost[move_row][next_col]:
                    cost[move_row][next_col] = new_cost
                    heappush(pq, (new_cost, move_row, next_col))
                    path_from[(move_row, next_col)] = (current_row, current_col)

    return []

def reconstruct(path_from, start_row, end_row, end_col):
    path = []
    current = (end_row, end_col)
    while current != (start_row, 0):
        path.append(current)
        current = path_from[current]
    path.append((start_row, 0))
    return path[::-1]

def plot_contour_with_paths(grid, dijkstra_path, greedy_path):
    grid = np.array(grid)

    # Create the contour map
    plt.figure(figsize=(10, 8))
    contour = plt.contour(grid, cmap='terrain', levels=5)
    plt.colorbar(contour, label='Elevation')

    # Plot the Dijkstra path in red
    if dijkstra_path:
        dijkstra_coords = list(zip(*dijkstra_path))
        plt.plot(dijkstra_coords[1], dijkstra_coords[0], color='red', label="Dijkstra's Path")
        # Add start and end points for Dijkstra's path with labels
        plt.text(dijkstra_coords[1][0], dijkstra_coords[0][0], 'Start', color='black', fontsize=12, ha='right')
        plt.text(dijkstra_coords[1][-1], dijkstra_coords[0][-1], 'End', color='black', fontsize=12, ha='right')

    # Plot the Greedy path in orange
    if greedy_path:
        greedy_coords = list(zip(*greedy_path))
        plt.plot(greedy_coords[1], greedy_coords[0], color='orange', label="Greedy Path")

    plt.xlabel("Columns")
    plt.ylabel("Rows")
    plt.title("Elevation Contour Map with Paths")
    plt.legend()
    plt.show()



# Main Function
def main():
    # file_path = "lima_peru_9x10.dat"
    # num_rows, num_cols = 9, 10

    file_path = "Colorado_844x480.dat"
    num_rows, num_cols = 480, 844

    # Read the elevation data
    grid = read_elevation_data(file_path, num_rows, num_cols)

    # Find min and max elevation
    min_val, max_val = find_min_max(grid)
    print("--------------------------------------------------------")
    print("These are the lowest and highest elevations: ", min_val, max_val)

    # Get the starting row from arguments
    start_row = int(sys.argv[1])

    # Compute paths
    dijkstra_path = draw_path_dijkstra(grid, start_row)
    greedy_path = draw_path_greedy(grid, start_row)

    # Plot contour map with paths
    plot_contour_with_paths(grid, dijkstra_path, greedy_path)

# Run the main function
if __name__ == "__main__":
    main()
