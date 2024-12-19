import numpy as np
import matplotlib.pyplot as plt
import random, os
from tqdm import tqdm
from roomba_class import Roomba
import csv


# ### Setup Environment

def seed_everything(seed: int):
    """Seed everything for reproducibility."""
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)

def is_obstacle(position):
    """Check if the position is outside the grid boundaries (acting as obstacles)."""
    x, y = position
    return x < 0 or x >= GRID_WIDTH or y < 0 or y >= GRID_HEIGHT

def setup_environment(seed=111):
    """Setup function for grid and direction definitions."""
    global GRID_WIDTH, GRID_HEIGHT, HEADINGS, MOVEMENTS
    GRID_WIDTH = 10
    GRID_HEIGHT = 10
    HEADINGS = ['N', 'E', 'S', 'W']
    MOVEMENTS = {
        'N': (0, -1),
        'E': (1, 0),
        'S': (0, 1),
        'W': (-1, 0),
    }
    print("Environment setup complete with a grid of size {}x{}.".format(GRID_WIDTH, GRID_HEIGHT))
    seed_everything(seed)
    return GRID_WIDTH, GRID_HEIGHT, HEADINGS, MOVEMENTS


# ### Sensor Movements

def simulate_roomba(T, movement_policy,sigma):
    """
    Simulate the movement of a Roomba robot for T time steps and generate noisy observations.

    Parameters:
    - T (int): The number of time steps for which to simulate the Roomba's movement.
    - movement_policy (str): The movement policy dictating how the Roomba moves.
                             Options may include 'straight_until_obstacle' or 'random_walk'.
    - sigma (float): The standard deviation of the Gaussian noise added to the true position 
                     to generate noisy observations.

    Returns:
    - tuple: A tuple containing three elements:
        1. true_positions (list of tuples): A list of the true positions of the Roomba 
                                            at each time step as (x, y) coordinates.
        2. headings (list): A list of headings of the Roomba at each time step.
        3. observations (list of tuples): A list of observed positions with added Gaussian noise,
                                          each as (obs_x, obs_y).
    """
    # Start at the center
    start_pos = (GRID_WIDTH // 2, GRID_HEIGHT // 2)
    start_heading = random.choice(HEADINGS)
    roomba = Roomba(MOVEMENTS, HEADINGS,is_obstacle,start_pos, start_heading, movement_policy)

    true_positions = []
    observations = []
    headings = []

    print(f"Simulating Roomba movement for policy: {movement_policy}")
    for _ in tqdm(range(T), desc="Simulating Movement"):
        position = roomba.move()
        heading = roomba.heading
        true_positions.append(position)
        headings.append(heading)

        # Generate noisy observation
        noise = np.random.normal(0, sigma, 2)
        observed_position = (position[0] + noise[0], position[1] + noise[1])
        observations.append(observed_position)

    return true_positions, headings, observations


# ### Implement Functions

def emission_probability(state, observation,sigma):
    """
    Calculate the emission probability in log form for a given state and observation using a Gaussian distribution.

    Parameters:
    - state (tuple): The current state represented as (position, heading), 
                     where position is a tuple of (x, y) coordinates.
    - observation (tuple): The observed position as a tuple (obs_x, obs_y).
    - sigma (float): The standard deviation of the Gaussian distribution representing observation noise.

    Returns:
    - float: The log probability of observing the given observation from the specified state.
    """
    ###### YOUR CODE HERE ######
    
    position, _ = state
    obs_x, obs_y = observation

    # Calculate the log probability of the observation given the state
    diff_x, diff_y = position[0] - obs_x, position[1] - obs_y
    log_prob = -0.5 * (diff_x**2 + diff_y**2) / sigma**2 - np.log(2 * np.pi * sigma**2) / 2
    return log_prob

def transition_probability(prev_state, curr_state, movement_policy):
    """
    Calculate the transition probability in log form between two states based on a given movement policy.

    Parameters:
    - prev_state (tuple): The previous state represented as (position, heading),
                          where position is a tuple of (x, y) coordinates and heading is a direction.
    - curr_state (tuple): The current state represented as (position, heading),
                          similar to prev_state.
    - movement_policy (str): The movement policy that dictates how transitions are made. 
                             Options are 'straight_until_obstacle' and 'random_walk'.

    Returns:
    - float: The log probability of transitioning from prev_state to curr_state given the movement policy.
             Returns 0.0 (log(1)) for certain transitions, -inf (log(0)) for impossible transitions,
             and a uniform log probability for equal transitions in the case of random walk.
    """
    ###### YOUR CODE HERE ######
    
    prev_pos, prev_heading = prev_state
    curr_pos, curr_heading = curr_state
    dx, dy = MOVEMENTS[prev_heading]
    expected_pos = (prev_pos[0] + dx, prev_pos[1] + dy)

    if movement_policy == 'straight_until_obstacle':
        if curr_pos == expected_pos and curr_heading == prev_heading:
            return 0.0
        elif curr_pos != prev_pos:
            return -np.log(len(HEADINGS))
    elif movement_policy == 'random_walk':
        return -np.log(len(HEADINGS))
    
    return -np.inf

# ### Viterbi Algorithm
def viterbi(observations, start_state, movement_policy,states,sigma):
    """
    Perform the Viterbi algorithm to find the most likely sequence of states given a series of observations.

    Parameters:
    - observations (list of tuples): A list of observed positions, each as a tuple (obs_x, obs_y).
    - start_state (tuple): The initial state represented as (position, heading),
                           where position is a tuple of (x, y) coordinates.
    - movement_policy (str): The movement policy that dictates how transitions are made.
                             Options are 'straight_until_obstacle' and 'random_walk'.
    - states (list of tuples): A list of all possible states, each represented as (position, heading).
    - sigma (float): The standard deviation of the Gaussian distribution representing observation noise.

    Returns:
    - list of tuples: The most probable sequence of states that could have led to the given observations.
    """
    ###### YOUR CODE HERE ######
    
    T = len(observations)
    dp = [{} for _ in range(T)] 
    bp = [{} for _ in range(T)]

    dp[0][start_state] = 0.0
    for state in states:
        if state != start_state:
            dp[0][state] = -np.inf

    for t in range(1, T):
        for curr_state in states:
            max_prob = -np.inf
            best_prev_state = None

            for prev_state in states:
                trans_prob = transition_probability(prev_state, curr_state, movement_policy)
                if trans_prob == -np.inf:
                    continue

                prob = dp[t-1][prev_state] + trans_prob
                if prob > max_prob:
                    max_prob = prob
                    best_prev_state = prev_state

            emission_prob = emission_probability(curr_state, observations[t], sigma)
            dp[t][curr_state] = max_prob + emission_prob
            bp[t][curr_state] = best_prev_state

    max_final_state = max(dp[-1], key=dp[-1].get)
    estimated_path = [max_final_state]

    for t in range(T-1, 0, -1):
        estimated_path.append(bp[t][estimated_path[-1]])

    return estimated_path[::-1]


# ### Evaluation (DO NOT CHANGE THIS)
def getestimatedPath(policy, results, states, sigma):
    """
    Estimate the path of the Roomba using the Viterbi algorithm for a specified policy.

    Parameters:
    - policy (str): The movement policy used during simulation, such as 'random_walk' or 'straight_until_obstacle'.
    - results (dict): A dictionary containing simulation results for different policies. Each policy's data includes:
                      - 'true_positions': List of true positions of the Roomba at each time step.
                      - 'headings': List of headings of the Roomba at each time step.
                      - 'observations': List of noisy observations at each time step.
    - states (list of tuples): A list of all possible states (position, heading) for the Hidden Markov Model.
    - sigma (float): The standard deviation of the Gaussian noise used in the emission probability.

    Returns:
    - tuple: A tuple containing:
        1. true_positions (list of tuples): The list of true positions from the simulation.
        2. estimated_path (list of tuples): The most likely sequence of states estimated by the Viterbi algorithm.
    """
    print(f"\nProcessing policy: {policy}")
    data = results[policy]
    observations = data['observations']
    start_state = (data['true_positions'][0], data['headings'][0])
    estimated_path = viterbi(observations, start_state, policy, states, sigma)
    return data['true_positions'], estimated_path


def evaluate_viterbi(estimated_path, true_positions, T,policy):
    """
    Evaluate the accuracy of the Viterbi algorithm's estimated path compared to the true path.
    """
    correct = 0
    for true_pos, est_state in zip(true_positions, estimated_path):
        if true_pos == est_state[0]:
            correct += 1
    accuracy = correct / T * 100
    # data['accuracy'] = accuracy
    print(f"Tracking accuracy for {policy.replace('_', ' ')} policy: {accuracy:.2f}%")


def plot_results(true_positions, observations, estimated_path, policy, seed):
    """
    Plot the true and estimated paths of the Roomba along with the noisy observations.
    The function plots and saves the graphs of the true and estimated paths.
    """
    # Extract coordinates
    true_x = [pos[0] for pos in true_positions]
    true_y = [pos[1] for pos in true_positions]
    obs_x = [obs[0] for obs in observations]
    obs_y = [obs[1] for obs in observations]
    est_x = [state[0][0] for state in estimated_path]
    est_y = [state[0][1] for state in estimated_path]

    # Identify start and end positions
    start_true = true_positions[0]
    end_true = true_positions[-1]
    start_est = estimated_path[0][0]
    end_est = estimated_path[-1][0]

    # Plotting
    plt.figure(figsize=(10, 10))

    # True Path Plot
    plt.subplot(2, 1, 1)
    plt.plot(true_x, true_y, 'g-', label='True Path', linewidth=2)
    plt.scatter(obs_x, obs_y, c='r', s=10, label='Observations')

    # Mark start and end positions on the true path
    plt.scatter(*start_true, c='b', marker='o', s=100, label='True Start', edgecolors='black')
    plt.scatter(*end_true, c='purple', marker='X', s=100, label='True End', edgecolors='black')

    plt.title(f'Roomba Path Tracking ({policy.replace("_", " ").title()} Policy) - True Path')
    plt.xlabel('X Position')
    plt.ylabel('Y Position')
    plt.legend()
    plt.grid(True)

    # Estimated Path Plot
    plt.subplot(2, 1, 2)
    plt.plot(est_x, est_y, 'b--', label='Estimated Path', linewidth=2)
    plt.scatter(obs_x, obs_y, c='r', s=10, label='Observations')

    # Mark start and end positions on the estimated path
    plt.scatter(*start_est, c='b', marker='o', s=100, label='Estimated Start', edgecolors='black')
    plt.scatter(*end_est, c='purple', marker='X', s=100, label='Estimated End', edgecolors='black')

    plt.title(f'Roomba Path Tracking ({policy.replace("_", " ").title()} Policy) - Estimated Path')
    plt.xlabel('X Position')
    plt.ylabel('Y Position')
    plt.legend()
    plt.grid(True)

    plt.suptitle(f'Roomba Path Tracking ({policy.replace("_", " ").title()} Policy) - Seed: {seed}')
    
    fname = f"{policy.replace('_', ' ')}_Policy_Roomba_Path_Tracking_{seed}.png"
    plt.savefig(fname)



if __name__ == "__main__":
    seeds = [111, 222, 333, 444, 555]
    # 1. Set up the environment, including grid size, headings, and movements.
    output_file = 'estimated_paths.csv'
    with open(output_file, mode='w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(["Seed", "Policy", "Estimated Path"])

    for seed in seeds:
        # seed = 111
        setup_environment(seed)
        sigma = 1.0  # Observation noise standard deviation
        T = 50       # Number of time steps

        # Simulate for both movement policies
        policies = ['random_walk', 'straight_until_obstacle']
        results = {}

        # 2. Loop through each movement policy and simulate the Roomba's movement:
        #    - Generate true positions, headings, and noisy observations.
        #    - Store the results in the dictionary.
        for policy in policies:
            true_positions, headings, observations = simulate_roomba(T, policy,sigma)
            results[policy] = {
                'true_positions': true_positions,
                'headings': headings,
                'observations': observations
            }

        # 3. Define the HMM components
        #   - A list (states) containing all possible states of the Roomba, where each state is represented as a tuple ((x, y), h)
        #   - x, y: The position on the grid.
        #   - h: The heading or direction (e.g., 'N', 'E', 'S', 'W').
        states = []
        ###### YOUR CODE HERE ######
        states = [((x, y), h) for x in range(GRID_WIDTH) for y in range(GRID_HEIGHT) for h in HEADINGS]
        
        # 4. Loop through each policy to estimate the Roomba's path using the Viterbi algorithm:
        #    - Retrieve the true positions and estimated path.
        #    - Evaluate the accuracy of the Viterbi algorithm.
        #    - Plot the true and estimated paths along with the observations.
        # for policy in policies:
        #     true_positions, estimated_path = getestimatedPath(policy,results,states,sigma)
        #     evaluate_viterbi(estimated_path, true_positions, T,policy)
        #     plot_results(true_positions, observations, estimated_path, policy, seed)

        # # saving the paths to a csv file
        # with open(output_file, mode='a', newline='') as file:
        #     writer = csv.writer(file)
        #     for policy in policies:
        #         estimated_path_str = ";".join([f"{pos}" for pos in estimated_path])
        #         writer.writerow([seed, policy, estimated_path_str])
        with open(output_file, mode='a', newline='') as file:
            writer = csv.writer(file)

            for policy in policies:
                # Estimate path using Viterbi
                true_positions, estimated_path = getestimatedPath(policy, results, states, sigma)
                
                # Evaluate and plot
                evaluate_viterbi(estimated_path, true_positions, T, policy)
                plot_results(true_positions, results[policy]['observations'], estimated_path, policy, seed)
                
                # Write to CSV
                estimated_path_str = ";".join([f"{pos}" for pos in estimated_path])
                writer.writerow([seed, policy, estimated_path_str])