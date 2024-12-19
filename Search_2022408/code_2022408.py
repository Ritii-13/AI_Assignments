import numpy as np
import pickle
import heapq
import math
import time
import tracemalloc
import matplotlib.pyplot as plt

# General Notes:
# - Update the provided file name (code_<RollNumber>.py) as per the instructions.
# - Do not change the function name, number of parameters or the sequence of parameters.
# - The expected output for each function is a path (list of node names)
# - Ensure that the returned path includes both the start node and the goal node, in the correct order.
# - If no valid path exists between the start and goal nodes, the function should return None.

# --------------------------------------------------------------------------------------------------------------------

def measure_performance(func, *args):
    start_time = time.time()
    tracemalloc.start()

    result = func(*args)

    current, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    end_time = time.time()

    total_time = end_time - start_time
    memory_usage = peak / 1024 

    return result, total_time, memory_usage

def compare_algorithms(adj_matrix, num_nodes):
    algo1_times = []
    algo1_memory = []
    algo2_times = []
    algo2_memory = []
    algo3_times = []
    algo3_memory = []
    algo4_times = []
    algo4_memory = []

    for i in range(num_nodes):
        for j in range(num_nodes):
            if i != j:
                print(f"Running algorithms for pair ({i}, {j})")

                _, algo1_time, algo1_mem = measure_performance(get_ids_path, adj_matrix, i, j)
                algo1_times.append(algo1_time)
                algo1_memory.append(algo1_mem)
                print(f"Iterative Deepening Search: {algo1_time} seconds, {algo1_mem} KB")

                _, algo2_time, algo2_mem = measure_performance(get_bidirectional_search_path, adj_matrix, i, j)
                algo2_times.append(algo2_time)
                algo2_memory.append(algo2_mem)
                print(f"Bidirectional Search: {algo2_time} seconds, {algo2_mem} KB")

                _, algo3_time, algo3_mem = measure_performance(get_astar_search_path, adj_matrix, node_attributes, i, j)
                algo3_times.append(algo3_time)
                algo3_memory.append(algo3_mem)
                print(f"Astar Search: {algo3_time} seconds, {algo3_mem} KB")

                _, algo4_time, algo4_mem = measure_performance(get_bidirectional_heuristic_search_path, adj_matrix, node_attributes, i, j)
                algo4_times.append(algo4_time)
                algo4_memory.append(algo4_mem)
                print(f"Bidirectional Heuristic Search: {algo4_time} seconds, {algo4_mem} KB")

                print("--------------------------------------------------")

    print("Total time for all pairs:")
    print(f"Iterative Deepening Search: {sum(algo1_times)} seconds")
    print(f"Bidirectional Search: {sum(algo2_times)} seconds")
    print(f"Astar Search: {sum(algo3_times)} seconds")
    print(f"Bidirectional Heuristic Search: {sum(algo4_times)} seconds")

    print("Total memory for all pairs:")
    print(f"Iterative Deepening Search: {sum(algo1_memory)} KB")
    print(f"Bidirectional Search: {sum(algo2_memory)} KB")
    print(f"Astar Search: {sum(algo3_memory)} KB")
    print(f"Bidirectional Heuristic Search: {sum(algo4_memory)} KB")

    return algo1_times, algo1_memory, algo2_times, algo2_memory, algo3_times, algo3_memory, algo4_times, algo4_memory

def plot_comparison(time_data1, memory_data1, time_data2, memory_data2, time_data3, memory_data3, time_data4, memory_data4, label1, label2, label3, label4):
    plt.figure(figsize=(10, 6))

    # Plot for Algorithm 1
    plt.scatter(memory_data1, time_data1, color='r', label=label1)

    # Plot for Algorithm 2
    plt.scatter(memory_data2, time_data2, color='b', label=label2)

    # Plot for Algorithm 3
    plt.scatter(memory_data3, time_data3, color='g', label=label3)

    # Plot for Algorithm 4
    plt.scatter(memory_data4, time_data4, color='o', label=label4)

    plt.title('Comparison of Algorithm Performance: Time vs Memory Usage')
    plt.xlabel('Memory Usage (KB)')
    plt.ylabel('Time (seconds)')
    plt.legend()
    plt.grid(True)
    plt.show()

# --------------------------------------------------------------------------------------------------------------------

# Algorithm: Iterative Deepening Search (IDS)

# Input:
#   - adj_matrix: Adjacency matrix representing the graph.
#   - start_node: The starting node in the graph.
#   - goal_node: The target node in the graph.

# Return:
#   - A list of node names representing the path from the start_node to the goal_node.
#   - If no path exists, the function should return None.

# Sample Test Cases:

#   Test Case 1:
#     - Start node: 1, Goal node: 2
#     - Return: [1, 7, 6, 2]

#   Test Case 2:
#     - Start node: 5, Goal node: 12
#     - Return: [5, 97, 98, 12]

#   Test Case 3:
#     - Start node: 12, Goal node: 49
#     - Return: None

#   Test Case 4:
#     - Start node: 4, Goal node: 12
#     - Return: [4, 6, 2, 9, 8, 5, 97, 98, 12]

def depth_limited_dfs(adj_matrix, node, goal_node, depth, visited):
    if node == goal_node:
        return [node]

    if depth <= 0:
        return None

    visited.add(node)
    
    for neighbor, connected in enumerate(adj_matrix[node]):
        if connected and neighbor not in visited:
            result = depth_limited_dfs(adj_matrix, neighbor, goal_node, depth - 1, visited)
            if result is not None:
                return [node] + result
    
    # visited.remove(node)
    return None

def get_ids_path(adj_matrix, start_node, goal_node):
    n = len(adj_matrix)
    # print(n)
    for depth in range(n):
        visited = set()
        # path_visited = set()
        visited.add(start_node)
        # print(f"Trying depth: {depth}")
        result = depth_limited_dfs(adj_matrix, start_node, goal_node, depth, visited)
        if result is not None:
            return result
    return None


# --------------------------------------------------------------------------------------------------------------------

# Algorithm: Bi-Directional Search

# Input:
#   - adj_matrix: Adjacency matrix representing the graph.
#   - start_node: The starting node in the graph.
#   - goal_node: The target node in the graph.

# Return:
#   - A list of node names representing the path from the start_node to the goal_node.
#   - If no path exists, the function should return None.

# Sample Test Cases:

#   Test Case 1:
#     - Start node: 1, Goal node: 2
#     - Return: [1, 7, 6, 2]

#   Test Case 2:
#     - Start node: 5, Goal node: 12
#     - Return: [5, 97, 98, 12]

#   Test Case 3:
#     - Start node: 12, Goal node: 49
#     - Return: None

#   Test Case 4:
#     - Start node: 4, Goal node: 12
#     - Return: [4, 6, 2, 9, 8, 5, 97, 98, 12]

def reconstruct_path(reachedF, reachedB, meeting_node):
  fwd_path = []
  node = meeting_node
  while node is not None:
    fwd_path.append(node)
    node = reachedF[node][1]
  fwd_path.reverse()

  bwd_path = []
  node = reachedB[meeting_node][1]
  while node is not None:
    bwd_path.append(node)
    node = reachedB[node][1]

  return fwd_path + bwd_path

def proceed(adj_matrix, frontier, reached, opposite_reached):
  _, cost, node, parent = heapq.heappop(frontier)

  for neighbour, connected in enumerate(adj_matrix[node]):
    if connected:
      new_cost = cost + 1
      if neighbour not in reached or new_cost < reached[neighbour][0]:
        heapq.heappush(frontier, (new_cost, new_cost, neighbour, node))
        reached[neighbour] = (new_cost, node)

        if neighbour in opposite_reached:
          return neighbour
          
  return None

def get_bidirectional_search_path(adj_matrix, start_node, goal_node):
  if start_node == goal_node:
    return [start_node]
  
  frontierF = [(0, 0, start_node, None)]
  frontierB = [(0, 0, goal_node, None)]

  reachedF = {start_node: (0, None)}
  reachedB = {goal_node: (0, None)}

  while frontierF and frontierB:
    if frontierF[0][1] <= frontierB[0][1]:
      meeting_node = proceed(adj_matrix, frontierF, reachedF, reachedB)
    else:
      meeting_node = proceed(adj_matrix, frontierB, reachedB, reachedF)

    if meeting_node is not None:
      return reconstruct_path(reachedF, reachedB, meeting_node)
    
  return None

# --------------------------------------------------------------------------------------------------------------------

# Algorithm: A* Search Algorithm

# Input:
#   - adj_matrix: Adjacency matrix representing the graph.
#   - node_attributes: Dictionary of node attributes containing x, y coordinates for heuristic calculations.
#   - start_node: The starting node in the graph.
#   - goal_node: The target node in the graph.

# Return:
#   - A list of node names representing the path from the start_node to the goal_node.
#   - If no path exists, the function should return None.

# Sample Test Cases:

#   Test Case 1:
#     - Start node: 1, Goal node: 2
#     - Return: [1, 27, 9, 2]

#   Test Case 2:
#     - Start node: 5, Goal node: 12
#     - Return: [5, 97, 28, 10, 12]

#   Test Case 3:
#     - Start node: 12, Goal node: 49
#     - Return: None

#   Test Case 4:
#     - Start node: 4, Goal node: 12
#     - Return: [4, 6, 27, 9, 8, 5, 97, 28, 10, 12]

def euclidean_distance(node1, node2, node_data):
    try:
        x1, y1 = node_data[node1]['x'], node_data[node1]['y']
        x2, y2 = node_data[node2]['x'], node_data[node2]['y']
        return math.sqrt((x1 - x2)**2 + (y1 - y2)**2)
    except KeyError as e:
        print(f"Missing node data: {e}")
        return float('inf')

def heuristic(current, start, goal, node_data):
    dist_start_to_current = euclidean_distance(start, current, node_data)  
    dist_current_to_goal = euclidean_distance(current, goal, node_data)  

    return dist_start_to_current + dist_current_to_goal
    
def get_astar_search_path(adj_matrix, node_attributes, start_node, goal_node):
  frontier = [(0, start_node, [])]  # (priority, current_node, path)
  explored = set()
  costs = {start_node: 0}

  while frontier:
    cost, current_node, path = heapq.heappop(frontier)
    path = path + [current_node]

    if current_node == goal_node:
      # print(f"Final path: {path}, Total cost: {cost}")
      return path

    explored.add(current_node)

    for neighbor, travel_cost in enumerate(adj_matrix[current_node]):
      if travel_cost > 0 and neighbor not in explored:
        new_cost = costs[current_node] + travel_cost  
        heuristic_cost = heuristic(neighbor, start_node, goal_node, node_attributes)

        if heuristic_cost == float('inf'):
            continue 

        total_cost = new_cost + heuristic_cost  

        if neighbor not in costs or new_cost < costs[neighbor]:
            costs[neighbor] = new_cost
            heapq.heappush(frontier, (total_cost, neighbor, path))

  return None

# --------------------------------------------------------------------------------------------------------------------

# Algorithm: Bi-Directional Heuristic Search

# Input:
#   - adj_matrix: Adjacency matrix representing the graph.
#   - node_attributes: Dictionary of node attributes containing x, y coordinates for heuristic calculations.
#   - start_node: The starting node in the graph.
#   - goal_node: The target node in the graph.

# Return:
#   - A list of node names representing the path from the start_node to the goal_node.
#   - If no path exists, the function should return None.

# Sample Test Cases:

#   Test Case 1:
#     - Start node: 1, Goal node: 2
#     - Return: [1, 27, 6, 2]

#   Test Case 2:
#     - Start node: 5, Goal node: 12
#     - Return: [5, 97, 98, 12]

#   Test Case 3:
#     - Start node: 12, Goal node: 49
#     - Return: None

#   Test Case 4:
#     - Start node: 4, Goal node: 12
#     - Return: [4, 34, 33, 11, 32, 31, 3, 5, 97, 28, 10, 12]


# Heuristic function for the forward direction (hF)
def heuristic_bidirectional(current, start, goal, node_attributes):
  dist_start_to_current = euclidean_distance(start, current, node_attributes) 
  dist_current_to_goal = euclidean_distance(current, goal, node_attributes)  

  return dist_start_to_current + dist_current_to_goal

# f2 function that combines g(n) and h(n) into a priority value
def f2(g, h):
  return max(2 * g, g + h)

# Expanding nodes in one direction
def proceed_bidirectional(frontier, reached, opposite_reached, adj_matrix, node_attributes, target_node, g_opposite, is_forward=True):
  if not frontier:
    return None
  
  _, cost, node, parent = heapq.heappop(frontier)

  for neighbor, travel_cost in enumerate(adj_matrix[node]):
    if travel_cost > 0:
      new_cost = cost + travel_cost

      if neighbor not in reached or new_cost < reached[neighbor][0]:
        if is_forward:
          heuristic_cost = heuristic_bidirectional(neighbor, node, target_node, node_attributes)
        else:
          heuristic_cost = heuristic_bidirectional(neighbor, target_node, node, node_attributes)

        if heuristic_cost == float('inf'):
          continue

        f_cost = f2(new_cost, heuristic_cost)

        if neighbor in opposite_reached:
          g_opposite_cost = g_opposite.get(neighbor, (float('inf'),))[0]  
          if g_opposite_cost != float('inf'):
            f2_cost = max(new_cost + g_opposite_cost, f_cost, opposite_reached[neighbor][1] if opposite_reached[neighbor][1] is not None else float('inf'))
          else:
            f2_cost = f_cost
        else:
          f2_cost = f_cost

        heapq.heappush(frontier, (f2_cost, new_cost, neighbor, node))
        reached[neighbor] = (new_cost, node)
        g_opposite[neighbor] = (new_cost, f_cost)

        if neighbor in opposite_reached:
          return neighbor

  return None 

# Path reconstruction function
def reconstruct_bidirectional_path(reachedF, reachedB, meeting_node):
  fwd_path = []
  node = meeting_node
  while node is not None:
    fwd_path.append(node)
    node = reachedF[node][1]
  fwd_path.reverse()

  bwd_path = []
  node = reachedB[meeting_node][1]
  while node is not None:
    bwd_path.append(node)
    node = reachedB[node][1]

  return fwd_path + bwd_path

def get_bidirectional_heuristic_search_path(adj_matrix, node_attributes, start_node, goal_node):
  if start_node == goal_node:
    return [start_node]

  # Initialize frontiers and cost tracking for forward and backward searches
  frontierF = [(0, 0, start_node, None)]  # (f2_cost, g_cost, node, parent)
  frontierB = [(0, 0, goal_node, None)]
  reachedF = {start_node: (0, None)}
  reachedB = {goal_node: (0, None)}
  g_oppositeF = {start_node: (0, None)}
  g_oppositeB = {goal_node: (0, None)}

  # Process frontiers until they meet
  while frontierF and frontierB:
    # Expand forward
    meeting_node = proceed_bidirectional(frontierF, reachedF, reachedB, adj_matrix, node_attributes, goal_node, g_oppositeB, is_forward=True)
    if meeting_node is not None:
      return reconstruct_bidirectional_path(reachedF, reachedB, meeting_node)

    # Expand backward
    meeting_node = proceed_bidirectional(frontierB, reachedB, reachedF, adj_matrix, node_attributes, start_node, g_oppositeF, is_forward=False)
    if meeting_node is not None:
      return reconstruct_bidirectional_path(reachedF, reachedB, meeting_node)

  return None  

# --------------------------------------------------------------------------------------------------------------------

# Bonus Problem
 
# Input:
# - adj_matrix: A 2D list or numpy array representing the adjacency matrix of the graph.

# Return:
# - A list of tuples where each tuple (u, v) represents an edge between nodes u and v.
#   These are the vulnerable roads whose removal would disconnect parts of the graph.

# Note:
# - The graph is undirected, so if an edge (u, v) is vulnerable, then (v, u) should not be repeated in the output list.
# - If the input graph has no vulnerable roads, return an empty list [].

def check_connectivity(adj_matrix, start_node):
  visited = set()
  stack = [start_node]
  while stack:
    node = stack.pop()
    if node not in visited:
      visited.add(node)
      # Add connected nodes to the stack
      for neighbor, connected in enumerate(adj_matrix[node]):
        if connected > 0 and neighbor not in visited:
            stack.append(neighbor)
  return visited

def find_connected_components(adj_matrix):
    visited = set()
    components = []

    for node in range(len(adj_matrix)):
        if node not in visited:
            component = check_connectivity(adj_matrix, node)
            components.append(component)
            visited.update(component)
    
    return components

def print_components(bridges, adj_matrix):
  for u,v in bridges:
    reachable = check_connectivity(adj_matrix, u)
    if len(reachable) != len(adj_matrix):
      # components = find_connected_components(adj_matrix)
      print(f"Edge ({u}, {v}) is a bridge")
    else:
      print("not a bridge")

def bonus_problem(adj_matrix):
  n = len(adj_matrix)
  bridges = []
  discovery_time = [-1] * n
  low = [-1] * n
  parent = [-1] * n
  time = [0] 

  def dfs(u):
    discovery_time[u] = low[u] = time[0]
    time[0] += 1

    for v, connected in enumerate(adj_matrix[u]):
      if connected:
        if discovery_time[v] == -1: 
          parent[v] = u
          dfs(v)
          low[u] = min(low[u], low[v])

          if low[v] > discovery_time[u]:
              bridges.append((u, v))
        elif v != parent[u]:
          low[u] = min(low[u], discovery_time[v])

  for i in range(n):
    if discovery_time[i] == -1:
      dfs(i)

  # print_components(bridges, adj_matrix)

  return bridges

# --------------------------------------------------------------------------------------------------------------------

if __name__ == "__main__":
  adj_matrix = np.load('IIIT_Delhi.npy')
  with open('IIIT_Delhi.pkl', 'rb') as f:
    node_attributes = pickle.load(f)

  start_node = int(input("Enter the start node: "))
  end_node = int(input("Enter the end node: "))

  # # print(node_attributes)

  print(f'Iterative Deepening Search Path: {get_ids_path(adj_matrix,start_node,end_node)}')
  print(f'Bidirectional Search Path: {get_bidirectional_search_path(adj_matrix,start_node,end_node)}')
  print(f'A* Path: {get_astar_search_path(adj_matrix,node_attributes,start_node,end_node)}')
  print(f'Bidirectional Heuristic Search Path: {get_bidirectional_heuristic_search_path(adj_matrix,node_attributes,start_node,end_node)}')
  print(f'Bonus Problem: {bonus_problem(adj_matrix)}')

  # num_nodes = len(adj_matrix)
  # algo1_times, algo1_memory, algo2_times, algo2_memory, algo3_times, algo3_memory, algo4_times, algo4_memory = compare_algorithms(adj_matrix, num_nodes)

  # # Plot comparison between the two algorithms
  # plot_comparison(algo1_times, algo1_memory, algo2_times, algo2_memory, algo3_times, algo3_memory, algo4_times, algo4_memory, 'Iterative Deepening Search', 'Bidirectional Search', 'Astar Search', 'Bidirectional Heuristic Search')
