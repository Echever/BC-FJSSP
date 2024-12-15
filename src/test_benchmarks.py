import os
import torch
import numpy as np
import pandas as pd
import json
import time
import pickle
import argparse
from datetime import datetime
from actor import Model
from env import FJSSPEnv
from generate_instances.parsedata import get_file_data
from generate_instances.solver import flexible_jobshop
from sklearn.ensemble import GradientBoostingRegressor

device = torch.device('cpu')
if(torch.cuda.is_available()): 
    device = torch.device('cuda:0') 
    torch.cuda.empty_cache()
else:
    pass

def load_model(model_path):
    with open(model_path, 'rb') as f:
        return pickle.load(f)

def get_features(jobs, operations, tenv):
    new_operations = []
    new_op_id = 0
    for j_id in range(len(jobs)):
        adv_op = tenv.advance_operations[j_id]
        for op_id in jobs[j_id][adv_op:]:
            new_op_id += 1
            new_operations.append(operations[op_id].copy())

    feature = [len(jobs), len(new_operations), len(new_operations[0])]
    empty = np.zeros(len(new_operations[0]))
    for o in new_operations:
        empty[np.array(o) != 0] += 1
    
    all_operations = [item for row in new_operations for item in row if item != 0]
    feature.extend([
        np.min(all_operations),
        np.max(all_operations),
        np.max(all_operations) - np.min(all_operations),
        np.sum(empty),
        np.mean(empty),
        np.std(empty),
        np.mean((empty - np.mean(empty)) ** 3) / (np.var(empty) ** 1.5 + 1e-7)
    ])
    return feature

def predict_model(features, model, thres, num_steps, num_operations):
    prediction = model.predict(np.array([features]).reshape(1, -1))[0]
    remaining_steps = num_operations - num_steps
    should_break = (prediction > thres or remaining_steps < 40) and remaining_steps < 120
    return should_break

def process_instance(instance_path, complex_model, thres, time_cp, use_predictor):
    info = get_file_data(instance_path)
    jobs, operations = info[0], info[1]
    instances = [{"jobs": jobs, "operations": operations}]
    tenv = FJSSPEnv(instances)
    obs = tenv.reset()

    model = Model(hidden_channels=128, metadata=obs.metadata(), num_layers=3, heads=3)
    model.load_state_dict(torch.load(args.model_path, map_location=device))
    model.eval()

    model = model.to(device)
    done = False

    start = time.time()

    with torch.no_grad():
        while True:
            if use_predictor:
                if predict_model(get_features(jobs, operations, tenv), complex_model, thres, tenv.num_steps, len(operations)):
                    break

            if done:
                return {
                    "score": tenv.mk,
                    "name": os.path.basename(instance_path),
                    "time": time.time() - start,
                }
                
            
            nobs = tenv.normalize_state(obs).to(device)
            action = torch.argmax(model(nobs).T[0].cpu())
            obs = obs.to('cpu')
            obs, _, done, _ = tenv.step(action)

    new_jobs = []
    new_operations = []
    new_op_id = 0
    jobs_starts = []
    for j_id in range(len(jobs)):
        adv_op = tenv.advance_operations[j_id]
        aux = []
        for op_id in jobs[j_id][adv_op:]:
            aux.append(new_op_id)
            new_op_id += 1
            new_operations.append(operations[op_id].copy())
        if aux:
            new_jobs.append(aux)
            jobs_starts.append(int(tenv.operations_ends[j_id]))

    machines_starts = [int(x) for x in obs["machine"].x[:, 0]]
    resuls_or = []
    for _ in range(1):
        jobs_or = []
        for job in new_jobs:
            job_info = []
            for o_id in job:
                ops_info = [(int(new_operations[o_id][i]), i) for i in range(len(new_operations[o_id])) if new_operations[o_id][i] != 0]
                if ops_info:
                    job_info.append(ops_info)
            if job_info:
                jobs_or.append(job_info)
        _, b, _ = flexible_jobshop(jobs_or, len(new_operations[0]), (len(operations) - tenv.num_steps) * time_cp, machines_starts, jobs_starts)
        resuls_or.append(b)

    result = np.mean(resuls_or)
    elapsed_time = time.time() - start

    return {
        "score": result,
        "name": os.path.basename(instance_path),
        "time": elapsed_time,
    }

def main(args):

    complex_model = load_model(args.complex_model_path) if args.use_predictor else None

    results = []
    for instance in os.listdir(args.benchmark_folder):
        instance_path = os.path.join(args.benchmark_folder, instance)
        if os.path.isfile(instance_path):
            result = process_instance(instance_path, complex_model, args.threshold, args.time_cp, args.use_predictor)
            results.append(result)
            print(f"Processed {instance}: Score = {result['score']}, Time = {result['time']:.2f}s")

    # Create model_results folder if it doesn't exist
    os.makedirs("model_results", exist_ok=True)

    # Save results in model_results folder
    output_path = os.path.join("model_results", args.output_file)
    with open(output_path, "w") as outfile:
        json.dump(results, outfile, indent=2)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="FJSSP Solver")
    parser.add_argument("--benchmark_folder", type=str, default="data/benchmarks/brandimarte", help="Path to benchmark folder")
    parser.add_argument("--model_path", type=str, default="models/model.pt", help="Path to trained model")
    parser.add_argument("--complex_model_path", type=str, default="models/model_complex.pkl", help="Path to complex model")
    parser.add_argument("--threshold", type=float, default=0.98, help="Threshold for switching to OR solver")
    parser.add_argument("--time_cp", type=float, default=0.01, help="Time limit for CP solver")
    parser.add_argument("--output_file", type=str, default="results.json", help="Output file for results")
    parser.add_argument("--use_predictor", type=bool, default=True, help="Use CP capability predictor")

    args = parser.parse_args()
    main(args)
