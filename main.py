import numpy as np

# Define the environment
grid_size = 3
num_states = grid_size * grid_size
gamma = 0.99  # Discount factor
Actions = [(1, 0), (0, -1), (-1, 0), (0, 1)]
convergence = 1e-3


def bellman(grid, r, c, action):
    value = -1
    value += 0.1 * gamma * next_state(grid, r, c, (action - 1) % 4)  # right
    value += 0.8 * gamma * next_state(grid, r, c, action)
    value += 0.1 * gamma * next_state(grid, r, c, (action + 1) % 4)  # left
    return value


def next_state(grid, r, c, action):
    r_action, c_action = Actions[action]
    new_r, new_c = r + r_action, c + c_action
    
    if new_r < 0 or new_c < 0 or new_r >= grid_size or new_c >= grid_size:
        return grid[r][c]  # Stay at the current state if it's a wall
    else:
        return grid[new_r][new_c]


def valueIteration(grid, reward):
    iterations = 0
    while True:
        diff = 0
        temp_grid = [[reward, -1, 10],
                     [-1, -1, -1],
                     [-1, -1, -1]]
        iterations += 1
        for r in range(grid_size):
            for c in range(grid_size):
                if r == 0 and (c == 2 or c == 0):
                    continue
                temp_grid[r][c] = max([bellman(grid, r, c, action) for action in range(4)])
                diff = max(diff, abs(temp_grid[r][c] - grid[r][c]))
        grid = temp_grid
        print("iteration no.", iterations)
        print_grid(grid)
        if diff < convergence:
            break
    return grid, iterations


def print_grid(grid):
    for r in range(grid_size):
        print("|", end="")
        for c in range(grid_size):
            print(f" {grid[r][c]:6.5f}  |", end="")
        print("\n")
    print("-----------------------------------------------\n")


def optimal_policy(grid):
    policy = np.full((grid_size, grid_size), -1, dtype=int)  # Initialize with -1 for terminal states
    for r in range(grid_size):
        for c in range(grid_size):
            if r == 0 and (c == 0 or c == 2):  # Skip terminal states
                continue
            best_action, maxv = None, -float("inf")
            for action in range(4):
                v = bellman(grid, r, c, action)
                if v > maxv:
                    best_action, maxv = action, v
            policy[r, c] = best_action  # Store the best action
    return policy


def print_policy(policy):
    actions = ["Down", "Left", "Up", "Right"]
    for r in range(grid_size):
        print("|", end="")
        for c in range(grid_size):
            if r == 0 and (c == 0 or c == 2):  # Terminal states
                print(f" Terminal  |", end="")
            else:
                action = policy[r, c]
                print(f" {actions[action]:<8} |", end="")
        print("\n")









rewards = [100, 3, 0, -3]
for r in rewards:
    grid = [[r, -1, 10],
            [-1, -1, -1],
            [-1, -1, -1]]
    print (f"Reward: {r}")
    print("initial grid:")
    print_grid(grid)
    grid, iterations = valueIteration(grid, r)
    print("converges in iteration", iterations)
    policy=optimal_policy(grid)
    print("optimal policy:")
    print_policy(policy)
