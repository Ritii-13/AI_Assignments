# Boilerplate for AI Assignment — Knowledge Representation, Reasoning and Planning
# CSE 643

# Import necessary libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import plotly.graph_objects as go
import networkx as nx
from pyDatalog import pyDatalog
from collections import defaultdict, deque
import time 
import tracemalloc

## ****IMPORTANT****
## Don't import or use any other libraries other than defined above
## Otherwise your code file will be rejected in the automated testing

# ------------------ Global Variables ------------------
route_to_stops = defaultdict(list)  # Mapping of route IDs to lists of stops
trip_to_route = {}                   # Mapping of trip IDs to route IDs
stop_trip_count = defaultdict(int)    # Count of trips for each stop
fare_rules = {}                      # Mapping of route IDs to fare information
merged_fare_df = None                # To be initialized in create_kb()

# Load static data from GTFS (General Transit Feed Specification) files
df_stops = pd.read_csv('GTFS/stops.txt')
df_routes = pd.read_csv('GTFS/routes.txt')
df_stop_times = pd.read_csv('GTFS/stop_times.txt')
df_fare_attributes = pd.read_csv('GTFS/fare_attributes.txt')
df_trips = pd.read_csv('GTFS/trips.txt')
df_fare_rules = pd.read_csv('GTFS/fare_rules.txt')

# df_stop_times['trip_id'] = df_stop_times['trip_id'].astype(str)
# df_stop_times['stop_id'] = df_stop_times['stop_id'].astype(str)

# df_trips['trip_id'] = df_trips['trip_id'].astype(str)
# df_trips['route_id'] = df_trips['route_id'].astype(str)
# ------------------ Function Definitions ------------------

# Function to create knowledge base from the loaded data
def create_kb():
    """
    Create knowledge base by populating global variables with information from loaded datasets.
    It establishes the relationships between routes, trips, stops, and fare rules.
    
    Returns:
        None
    """
    global route_to_stops, trip_to_route, stop_trip_count, fare_rules, merged_fare_df

    # Create trip_id to route_id mapping
    for index, row in df_trips.iterrows():
        trip_to_route[row['trip_id']] = row['route_id']
    # print("Trip to route mapping created")

    # Map route_id to a list of stops
    for index, row in df_stop_times.iterrows():
        route_id = trip_to_route[row['trip_id']]
        stop_id = row['stop_id']
        route_to_stops[route_id].append(stop_id)
        stop_trip_count[stop_id] += 1
    # print("Route to stops mapping created")

    # Ensure each route only has unique stops
    for route_id, stops in route_to_stops.items():
        route_to_stops[route_id] = list(dict.fromkeys(stops))
    # print("Unique stops ensured for each route")
    
    # # Count trips per stop
    # # for index, row in df_stop_times.iterrows():
    # #     stop_trip_count[row['stop_id']] += 1

    # Create fare rules for routes
    fare_rules = df_fare_rules.groupby('route_id').apply(lambda x: x.to_dict(orient='records')).to_dict()
    # print("Fare rules created")

    # Merge fare rules and attributes into a single DataFrame
    merged_fare_df = pd.merge(df_fare_rules, df_fare_attributes, on='fare_id', how='left')
    # print("Fare rules and attributes merged")

# Function to find the top 5 busiest routes based on the number of trips
def get_busiest_routes():
    """
    Identify the top 5 busiest routes based on trip counts.

    Returns:
        list: A list of tuples, where each tuple contains:
              - route_id (int): The ID of the route.
              - trip_count (int): The number of trips for that route.
    """
    route_trip_counts = defaultdict(int)
    for route_id in trip_to_route.values():
        route_trip_counts[route_id] += 1
    busiest_routes = sorted(route_trip_counts.items(), key=lambda x: x[1], reverse=True)[:5]
    return busiest_routes

# Function to find the top 5 stops with the most frequent trips
def get_most_frequent_stops():
    """
    Identify the top 5 stops with the highest number of trips.

    Returns:
        list: A list of tuples, where each tuple contains:
              - stop_id (int): The ID of the stop.
              - trip_count (int): The number of trips for that stop.
    """
    most_frequent_stops = sorted(stop_trip_count.items(), key=lambda x: x[1], reverse=True)[:5]
    return most_frequent_stops

# Function to find the top 5 busiest stops based on the number of routes passing through them
def get_top_5_busiest_stops():
    """
    Identify the top 5 stops with the highest number of different routes.

    Returns:
        list: A list of tuples, where each tuple contains:
              - stop_id (int): The ID of the stop.
              - route_count (int): The number of routes passing through that stop.
    """
    stop_route_counts = defaultdict(set)
    for route_id, stops in route_to_stops.items():
        for stop_id in stops:
            stop_route_counts[stop_id].add(route_id)
    busiest_stops = sorted([(stop_id, len(routes)) for stop_id, routes in stop_route_counts.items()], key=lambda x: x[1], reverse=True)[:5]
    return busiest_stops

# Function to identify the top 5 pairs of stops with only one direct route between them
def get_stops_with_one_direct_route():
    """
    Identify the top 5 pairs of consecutive stops (start and end) connected by exactly one direct route. 
    The pairs are sorted by the combined frequency of trips passing through both stops.

    Returns:
        list: A list of tuples, where each tuple contains:
              - pair (tuple): A tuple with two stop IDs (stop_1, stop_2).
              - route_id (int): The ID of the route connecting the two stops.
    """
    direct_routes = defaultdict(list)
    for route_id, stops in route_to_stops.items():
        for i in range(len(stops) - 1):
            start, end = stops[i], stops[i + 1]
            direct_routes[(start, end)].append(route_id)

    one_route_pairs = [(pair, routes[0]) for pair, routes in direct_routes.items() if len(routes) == 1]
    one_route_pairs = sorted(one_route_pairs, key=lambda x: sum(stop_trip_count[stop] for stop in x[0]), reverse=True)[:5]
    return one_route_pairs

# Function to get merged fare DataFrame
# No need to change this function
def get_merged_fare_df():
    """
    Retrieve the merged fare DataFrame.

    Returns:
        DataFrame: The merged fare DataFrame containing fare rules and attributes.
    """
    global merged_fare_df
    return merged_fare_df

# Visualize the stop-route graph interactively
def visualize_stop_route_graph_interactive(route_to_stops):
    """
    Visualize the stop-route graph using Plotly for interactive exploration.

    Args:
        route_to_stops (dict): A dictionary mapping route IDs to lists of stops.

    Returns:
        None
    """
    G = nx.Graph()

    for route_id, stops in route_to_stops.items():
        for i in range(len(stops) - 1):
            G.add_edge(stops[i], stops[i + 1], route=route_id)

    pos = nx.spring_layout(G, seed=42)
    
    plt.figure(figsize=(200, 200))
    nx.draw_networkx_nodes(G, pos, node_size=15, node_color='skyblue')
    nx.draw_networkx_edges(G, pos, edge_color='blue', width=1.5, alpha=0.6)
    nx.draw_networkx_labels(G, pos, font_size=8)
    plt.title("Delhi's Open Transit Data")
    # plt.axis('off')
    # plt.colorbar(plt.cm.ScalarMappable(cmap=plt.cm.Blues), label="Node Value")
    plt.show()

# def measure_performance(func, *args):
#     start_time = time.time()
#     tracemalloc.start()

#     result = func(*args)

#     current, peak = tracemalloc.get_traced_memory()
#     tracemalloc.stop()
#     end_time = time.time()

#     total_time = end_time - start_time
#     memory_usage = peak / 1024 

#     return result, total_time, memory_usage

# unique_stops = list(route_to_stops.values())[0]

# random_pairs = [(random.choice(unique_stops), random.choice(unique_stops)) for _ in range(100)]

# # Store results
# brute_force_times = []
# brute_force_memory = []
# pydatalog_times = []
# pydatalog_memory = []

# for start_stop, end_stop in random_pairs:
#     # Measure brute-force performance
#     res, bf_time, bf_mem = measure_performance(direct_route_brute_force, start_stop, end_stop)
#     brute_force_times.append(bf_time)
#     brute_force_memory.append(bf_mem)
    
#     # Measure PyDatalog performance
#     res2, pd_time, pd_mem = measure_performance(query_direct_routes, start_stop, end_stop)
#     pydatalog_times.append(pd_time)
#     pydatalog_memory.append(pd_mem)

# # Calculate averages
# avg_bf_time = sum(brute_force_times) / len(brute_force_times)
# avg_bf_memory = sum(brute_force_memory) / len(brute_force_memory)
# avg_pd_time = sum(pydatalog_times) / len(pydatalog_times)
# avg_pd_memory = sum(pydatalog_memory) / len(pydatalog_memory)
    
# Print average results
# print(f"Brute-Force Average Execution Time: {avg_bf_time} seconds")
# print(f"Brute-Force Average Memory Usage: {avg_bf_memory} MB")
# print(f"PyDatalog Average Execution Time: {avg_pd_time} seconds")
# print(f"PyDatalog Average Memory Usage: {avg_pd_memory} MB")

# # Plotting the results
# plt.figure(figsize=(14, 6))

# # Execution time comparison
# plt.subplot(1, 2, 1)
# plt.plot(range(100), brute_force_times, color='blue', label='Brute-Force', alpha=0.7)
# plt.plot(range(100), pydatalog_times, color='green', label='PyDatalog', alpha=0.7)
# plt.xlabel('Test Case Index')
# plt.ylabel('Execution Time (seconds)')
# plt.title('Execution Time Comparison (100 Random Pairs)')
# plt.legend()

# # Memory usage comparison
# plt.subplot(1, 2, 2)
# plt.plot(range(100), brute_force_memory, color='blue', label='Brute-Force', alpha=0.7)
# plt.plot(range(100), pydatalog_memory, color='green', label='PyDatalog', alpha=0.7)
# plt.xlabel('Test Case Index')
# plt.ylabel('Memory Usage (MB)')
# plt.title('Memory Usage Comparison (100 Random Pairs)')
# plt.legend()

# random_test_cases = [(random.randint(2000, 30000), random.randint(2000, 30000), random.randint(1000, 5000), random.randint(1, 5)) for _ in range(100)]

# # Store results
# backward_times = []
# backward_memory = []
# forward_times = []
# forward_memory = []

# # Run tests for backward and forward chaining on 100 random pairs
# for start_stop_id, end_stop_id, stop_id_to_include, max_transfers in random_test_cases:
#     # Measure performance for backward chaining
#     res_bwd, exec_time_bwd, mem_usage_bwd = measure_performance(backward_chaining, start_stop_id, end_stop_id, stop_id_to_include, max_transfers)
#     backward_times.append(exec_time_bwd)
#     backward_memory.append(mem_usage_bwd)

#     # Measure performance for forward chaining
#     res_fwd, exec_time_fwd, mem_usage_fwd = measure_performance(forward_chaining, start_stop_id, end_stop_id, stop_id_to_include, max_transfers)
#     forward_times.append(exec_time_fwd)
#     forward_memory.append(mem_usage_fwd)

# print(f"Backward Chaining Average Execution Time: {sum(backward_times) / 100} seconds")
# print(f"Backward Chaining Average Memory Usage: {sum(backward_memory) / 100} MB")
# print(f"Forward Chaining Average Execution Time: {sum(forward_times) / 100} seconds")
# print(f"Forward Chaining Average Memory Usage: {sum(forward_memory) / 100} MB")

# # Plotting the results
# plt.figure(figsize=(14, 6))

# # Execution time comparison
# plt.subplot(1, 2, 1)
# plt.plot(range(100), backward_times, color='red', label='Backward Chaining', alpha=0.7)
# plt.plot(range(100), forward_times, color='blue', label='Forward Chaining', alpha=0.7)
# plt.xlabel('Test Case Index')
# plt.ylabel('Execution Time (seconds)')
# plt.title('Execution Time Comparison (100 Random Pairs)')
# plt.legend()

# # Memory usage comparison
# plt.subplot(1, 2, 2)
# plt.plot(range(100), backward_memory, color='red', label='Backward Chaining', alpha=0.7)
# plt.plot(range(100), forward_memory, color='blue', label='Forward Chaining', alpha=0.7)
# plt.xlabel('Test Case Index')
# plt.ylabel('Memory Usage (MB)')
# plt.title('Memory Usage Comparison (100 Random Pairs)')
# plt.legend()

# plt.tight_layout()
# plt.show()

# Brute-Force Approach for finding direct routes
def direct_route_brute_force(start_stop, end_stop):
    """
    Find all valid routes between two stops using a brute-force method.

    Args:
        start_stop (int): The ID of the starting stop.
        end_stop (int): The ID of the ending stop.

    Returns:
        list: A list of route IDs (int) that connect the two stops directly.
    """
    direct_routes = defaultdict(list)
    for route_id, stops in route_to_stops.items():
        if start_stop in stops and end_stop in stops:
            start_index, end_index = stops.index(start_stop), stops.index(end_stop)
            if start_index < end_index:
                direct_routes[(start_stop, end_stop)].append(route_id)
    return direct_routes[(start_stop, end_stop)]

pyDatalog.create_terms('RouteHasStop, DirectRoute, OptimalRoute, State, Action, GoalState, Start, Destination, Board, Transfer, X, Y, Z, R, R1, R2')     

# Adding route data to Datalog
def add_route_data(route_to_stops):
    """
    Add the route data to Datalog for reasoning.

    Args:
        route_to_stops (dict): A dictionary mapping route IDs to lists of stops.

    Returns:
        None
    """
    for route_id, stops in route_to_stops.items():
        for stop_id in stops:
            +RouteHasStop(route_id, stop_id)
    # print("Route data added to Datalog") 

# Initialize Datalog predicates for reasoning
def initialize_datalog():
    """
    Initialize Datalog terms and predicates for reasoning about routes and stops.

    Returns:
        None
    """
    pyDatalog.clear()  # Clear previous terms
    print("Terms initialized: DirectRoute, RouteHasStop, OptimalRoute")  # Confirmation print

    # Define Datalog predicates
    DirectRoute(X, Y, R) <= RouteHasStop(R, X) & RouteHasStop(R, Y) & (X != Y)
    OptimalRoute(X, Y, R1, Z, R2) <= DirectRoute(X, Z, R1) & DirectRoute(Z, Y, R2) & (R1 != R2)

    create_kb()  # Populate the knowledge base
    add_route_data(route_to_stops)  # Add route data to Datalog


# Function to query direct routes between two stops
def query_direct_routes(start, end):
    """
    Query for direct routes between two stops.

    Args:
        start (int): The ID of the starting stop.
        end (int): The ID of the ending stop.

    Returns:
        list: A sorted list of route IDs (str) connecting the two stops.
    """
    res = DirectRoute(start, end, R)
    # print(res)
    return sorted(set([r[0] for r in res]))

# Forward chaining for optimal route planning
def forward_chaining(start_stop_id, end_stop_id, stop_id_to_include, max_transfers):
    """
    Perform forward chaining to find optimal routes considering transfers.

    Args:
        start_stop_id (int): The starting stop ID.
        end_stop_id (int): The ending stop ID.
        stop_id_to_include (int): The stop ID where a transfer occurs.
        max_transfers (int): The maximum number of transfers allowed.

    Returns:
        list: A list of unique paths (list of tuples) that satisfy the criteria, where each tuple contains:
              - route_id1 (int): The ID of the first route.
              - stop_id (int): The ID of the intermediate stop.
              - route_id2 (int): The ID of the second route.
    """
    # pyDatalog.clear()

    # initialize_datalog()

    result = OptimalRoute(start_stop_id, end_stop_id, R1, stop_id_to_include, R2) 
    res= [(i[0], stop_id_to_include, i[1]) for i in result if max_transfers == 1] 
    return res

# Backward chaining for optimal route planning
def backward_chaining(start_stop_id, end_stop_id, stop_id_to_include, max_transfers):
    """
    Perform backward chaining to find optimal routes considering transfers.

    Args:
        start_stop_id (int): The starting stop ID.
        end_stop_id (int): The ending stop ID.
        stop_id_to_include (int): The stop ID where a transfer occurs.
        max_transfers (int): The maximum number of transfers allowed.

    Returns:
        list: A list of unique paths (list of tuples) that satisfy the criteria, where each tuple contains:
              - route_id1 (int): The ID of the first route.
              - stop_id (int): The ID of the intermediate stop.
              - route_id2 (int): The ID of the second route.
    """

    result = OptimalRoute(end_stop_id, start_stop_id, R2, stop_id_to_include, R1)
    res= [(i[0], stop_id_to_include, i[1]) for i in result if max_transfers == 1]
    return res

# PDDL-style planning for route finding
def pddl_planning(start_stop_id, end_stop_id, stop_id_to_include, max_transfers):
    """
    Implement PDDL-style planning to find routes with optional transfers.

    Args:
        start_stop_id (int): The starting stop ID.
        end_stop_id (int): The ending stop ID.
        stop_id_to_include (int): The stop ID for a transfer.
        max_transfers (int): The maximum number of transfers allowed.

    Returns:
        list: A list of unique paths (list of tuples) that satisfy the criteria, where each tuple contains:
              - route_id1 (int): The ID of the first route.
              - stop_id (int): The ID of the intermediate stop.
              - route_id2 (int): The ID of the second route.
    """
    pass # Implementation here


# Function to filter fare data based on an initial fare limit
def prune_data(merged_fare_df, initial_fare):
    """
    Filter fare data based on an initial fare limit.

    Args:
        merged_fare_df (DataFrame): The merged fare DataFrame.
        initial_fare (float): The maximum fare allowed.

    Returns:
        DataFrame: A filtered DataFrame containing only routes within the fare limit.
    """
    pass  # Implementation here

# Pre-computation of Route Summary
def compute_route_summary(pruned_df):
    """
    Generate a summary of routes based on fare information.

    Args:
        pruned_df (DataFrame): The filtered DataFrame containing fare information.

    Returns:
        dict: A summary of routes with the following structure:
              {
                  route_id (int): {
                      'min_price': float,          # The minimum fare for the route
                      'stops': set                # A set of stop IDs for that route
                  }
              }
    """
    pass  # Implementation here

# BFS for optimized route planning
def bfs_route_planner_optimized(start_stop_id, end_stop_id, initial_fare, route_summary, max_transfers=3):
    """
    Use Breadth-First Search (BFS) to find the optimal route while considering fare constraints.

    Args:
        start_stop_id (int): The starting stop ID.
        end_stop_id (int): The ending stop ID.
        initial_fare (float): The available fare for the trip.
        route_summary (dict): A summary of routes with fare and stop information.
        max_transfers (int): The maximum number of transfers allowed (default is 3).

    Returns:
        list: A list representing the optimal route with stops and routes taken, structured as:
              [
                  (route_id (int), stop_id (int)),  # Tuple for each stop taken in the route
                  ...
              ]
    """
    pass  # Implementation here
# print("Hi")
# create_kb()
# print(merged_fare_df.head())
