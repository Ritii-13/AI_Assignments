#############
## Imports ##
#############

import pickle
import pandas as pd
import numpy as np
import bnlearn as bn
from test_model import test_model
# import time
# import tracemalloc

######################
## Boilerplate Code ##
######################

def load_data():
    """Load train and validation datasets from CSV files."""
    # Implement code to load CSV files into DataFrames
    # Example: train_data = pd.read_csv("train_data.csv")
    train_df = pd.read_csv("train_data.csv")
    val_df = pd.read_csv("validation_data.csv")

    return train_df, val_df

def make_network(df):
    """Define and fit the initial Bayesian Network."""
    # Code to define the DAG, create and fit Bayesian Network, and return the model
    # Define the edges manually
    edges = [
        ('Start_Stop_ID', 'End_Stop_ID'),
        ('Start_Stop_ID', 'Distance'),
        ('Start_Stop_ID', 'Zones_Crossed'),
        ('Start_Stop_ID', 'Route_Type'),
        ('Start_Stop_ID', 'Fare_Category'),
        ('End_Stop_ID', 'Distance'),
        ('End_Stop_ID', 'Zones_Crossed'),
        ('End_Stop_ID', 'Route_Type'),
        ('End_Stop_ID', 'Fare_Category'),
        ('Distance', 'Zones_Crossed'),
        ('Distance', 'Route_Type'),
        ('Distance', 'Fare_Category'),
        ('Zones_Crossed', 'Route_Type'),
        ('Zones_Crossed', 'Fare_Category'),
        ('Route_Type', 'Fare_Category')
    ]

    # Create the DAG
    dag = bn.make_DAG(edges)

    # Create and fit the Bayesian Network
    model = bn.parameter_learning.fit(dag, df)

    return model

def visualize_network(model):
    """Visualize the Bayesian Network."""
    # Code to visualize the Bayesian Network

    # Visualize the Bayesian Network
    bn.plot(model)

def make_pruned_network(df):
    """Define and fit a pruned Bayesian Network."""
    # Code to create a pruned network, fit it, and return the pruned model

    edges = [
        ('Start_Stop_ID', 'End_Stop_ID'),
        ('Start_Stop_ID', 'Distance'),
        ('Start_Stop_ID', 'Zones_Crossed'),
        ('Start_Stop_ID', 'Route_Type'),
        ('Start_Stop_ID', 'Fare_Category'),
        ('End_Stop_ID', 'Distance'),
        ('End_Stop_ID', 'Zones_Crossed'),
        ('End_Stop_ID', 'Route_Type'),
        ('End_Stop_ID', 'Fare_Category'),
        ('Distance', 'Zones_Crossed'),
        ('Distance', 'Route_Type'),
        ('Distance', 'Fare_Category'),
        ('Zones_Crossed', 'Route_Type'),
        ('Zones_Crossed', 'Fare_Category'),
        ('Route_Type', 'Fare_Category')
    ]

    # Create the DAG
    dag = bn.make_DAG(edges)

    model = bn.independence_test(dag, df, alpha=0.5, prune=True)

    # print(model.keys())
    # print(model['structure_scores'])
    # print(model['independence_test'])

    pruned_model = bn.parameter_learning.fit(model, df)

    # print(pruned_model['structure_scores'])
    # print(pruned_model['independence_test'])

    return pruned_model

def make_optimized_network(df):
    """Perform structure optimization and fit the optimized Bayesian Network."""
    # Code to optimize the structure, fit it, and return the optimized model
    dag_optimized = bn.structure_learning.fit(df, methodtype='hc', scoretype='bic')
    model = bn.independence_test(dag_optimized, df, alpha=0.5, prune=True)
    optimized_model = bn.parameter_learning.fit(model, df)

    return optimized_model

def save_model(fname, model):
    """Save the model to a file using pickle."""
    with open(fname, 'wb') as f:
        pickle.dump(model, f)

def evaluate(model_name, val_df):
    """Load and evaluate the specified model."""
    with open(f"{model_name}.pkl", 'rb') as f:
        model = pickle.load(f)
        correct_predictions, total_cases, accuracy = test_model(model, val_df)
        print(f"Total Test Cases: {total_cases}")
        print(f"Total Correct Predictions: {correct_predictions} out of {total_cases}")
        print(f"Model accuracy on filtered test cases: {accuracy:.2f}%")


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

############
## Driver ##
############

def main():
    # Load data
    train_df, val_df = load_data()

    # Create and save base model
    base_model = make_network(train_df.copy())
    save_model("base_model.pkl", base_model)

    # Create and save pruned model
    pruned_network = make_pruned_network(train_df.copy())
    save_model("pruned_model.pkl", pruned_network)

    # Create and save optimized model
    optimized_network = make_optimized_network(train_df.copy())
    save_model("optimized_model.pkl", optimized_network)

    # Evaluate all models on the validation set
    evaluate("base_model", val_df)
    evaluate("pruned_model", val_df)
    evaluate("optimized_model", val_df)

    # Measure performance of all models
    # base_model_performance = measure_performance(make_network, train_df.copy())
    # pruned_model_performance = measure_performance(make_pruned_network, train_df.copy())
    # optimized_model_performance = measure_performance(make_optimized_network, train_df.copy())
    
    # print("\nPerformance Metrics:")
    # print(f"Base Model: Time = {base_model_performance[1]:.2f}s, Memory = {base_model_performance[2]:.2f}KB")
    # print(f"Pruned Model: Time = {pruned_model_performance[1]:.2f}s, Memory = {pruned_model_performance[2]:.2f}KB")
    # print(f"Optimized Model (Constraint Based): Time = {optimized_model_performance[1]:.2f}s, Memory = {optimized_model_performance[2]:.2f}KB")

    print("[+] Done")

if __name__ == "__main__":
    main()

