import os
import torch
import torch.nn.functional as F
import numpy as np
from datetime import datetime
from actor import Model
from torch_geometric.loader import DataLoader
from torch_geometric.nn import to_hetero
from generate_instances.generator import generate_instance_list
from generate_instances.parsedata import get_data, parse, get_file_data
from generate_instances.solver import solve_fjsp
import json
from env import FJSSPEnv
import pandas as pd

device = torch.device('cpu')

if(torch.cuda.is_available()):
    device = torch.device('cuda:0') 
    torch.cuda.empty_cache()
else:
    pass

def evaluate_val(model, file_name):

    all_results = []
     
    with open(file_name, "r") as f:
        data = json.load(f)

    for index, value in enumerate(data):
        jobs, operations = value["jobs"], value["operations"]
        instances = [{"jobs": jobs, "operations": operations}]
        tenv = FJSSPEnv(instances)
        obs = tenv.reset()

        model = model.to(device)
        with torch.no_grad():
            while True:
                nobs = tenv.normalize_state(obs)
                nobs = nobs.to(device)
                action = model(nobs).T[0]
                action = action.cpu()
                action = np.argmax(action)
                obs, _ ,done, _ = tenv.step(action) 
                if done:
                    break
        all_results.append(round((tenv.mk/value["result"][1] - 1),10))

    return round(np.mean(all_results), 2), all_results


def evaluate_model(model_name = None, folder = "data"):

    all_results = []
    dir_path = folder

    try:
        opts = list(pd.read_csv(folder + "/optimum/optimum.csv").iloc[:,1])
    except:
        opts = [1]
    onlyfiles = []

    # Iterate directory
    for path in os.listdir(dir_path):
        # check if current path is a file
        if os.path.isfile(os.path.join(dir_path, path)):
            onlyfiles.append(path)

    for index, value in enumerate(onlyfiles[0:len(opts)]):
        info = get_file_data(value, folder)
        jobs, operations = info[0], info[1]
        instances = [{"jobs": jobs, "operations": operations}]
        tenv = FJSSPEnv(instances)
        obs = tenv.reset()

        if model_name is None:
            model = Model(hidden_channels=128, metadata=obs.metadata(), num_layers= 3)
            model.load_state_dict(torch.load('models/model.pt'))
        else:
            model = model_name

        model = model.to(device)
        with torch.no_grad():
            while True:
                nobs = tenv.normalize_state(obs)
                nobs = nobs.to(device)
                action = model(nobs).T[0]
                #action[nobs[('machine','exec','job')].mask] = float("-inf")
                action = action.cpu()
                action = np.argmax(action)
                eaction = obs['machine', 'exec', 'job'].edge_index[:,action]
                #print("action", action)
                sel_job = eaction[1]
                sel_mach = eaction[0]
                obs, _ ,done, _ = tenv.step(action) 
                #print(float(torch.max(obs["machine"].x[:,0])) )
                if done:
                    break
        all_results.append(round((tenv.mk/opts[index] - 1),10))

    return round(np.mean(all_results), 2), all_results

if __name__ == "__main__":
    evaluate_model()