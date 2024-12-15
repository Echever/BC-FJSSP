import os
import torch
import torch.nn.functional as F
import numpy as np
import argparse
import copy
import random
import json
from datetime import datetime
from actor import Model
from torch_geometric.loader import DataLoader
from torch_geometric.nn import to_hetero
from generate_instances.generator import generate_instance_list
from generate_instances.parsedata import get_data, parse
from generate_instances.solver import solve_fjsp
from env import FJSSPEnv
from evaluate_model import evaluate_model, evaluate_val

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
if torch.cuda.is_available():
    torch.cuda.empty_cache()

def parse_arguments():
    parser = argparse.ArgumentParser(description='FJSSP Model Training')
    parser.add_argument('--hidden_channels', type=int, default=128, help='Number of hidden channels')
    parser.add_argument('--num_layers', type=int, default=3, help='Number of layers')
    parser.add_argument('--learning_rate', type=float, default=0.0002, help='Learning rate')
    parser.add_argument('--heads', type=int, default=3, help='Number of attention heads')
    parser.add_argument('--episodes', type=int, default=25, help='Number of episodes')
    parser.add_argument('--num_cases', type=int, default=50, help='Number of cases')
    parser.add_argument('--batch_size', type=int, default=64, help='Batch size')
    parser.add_argument('--epochs', type=int, default=4, help='Number of epochs')
    parser.add_argument('--number_group', type=int, default=3, help='Number of groups')
    return parser.parse_args()

def generate_train_instances(train_config):
    list_instances = generate_instance_list(**train_config)
    instances = []
    for instance in list_instances:
        jobs, operations, info, maximum = get_data(parse(instance))
        instances.append({
            "jobs": jobs,
            "operations": operations,
            "maximum": maximum,
            "num_machines": info["machinesNb"],
            "result": solve_fjsp(jobs, operations)
        })
    return instances

def train(model, optimizer, train_loader):
    kl_loss = torch.nn.KLDivLoss(reduction="batchmean", log_target=False)
    model.train()
    total_examples, total_loss, total_acc = 0, 0, 0
    for batch in train_loader:
        optimizer.zero_grad()
        batch = batch.to(device)
        res = model(batch)
        list_action_prob = []
        list_targets = []
        row, col = batch[('machine', 'exec', 'job')].edge_index
        batch_index = batch["machine"].batch[row]
        for i in range(batch["operation"].batch[-1] + 1):
            action_probs = res[batch_index==i].T[0]
            action_probs = F.log_softmax(action_probs, dim=0)
            list_action_prob.append(action_probs)
            list_targets.append(batch[('machine','exec','job')].y[batch_index==i])
        acc = get_accuracy(list_action_prob, list_targets)
        list_action_prob = torch.stack(apply_padding(list_action_prob))
        list_targets = torch.stack(apply_padding(list_targets))
        loss = kl_loss(list_action_prob, list_targets)
        loss.backward()
        optimizer.step()
        len_bach_size = len(batch)
        total_examples += len_bach_size
        total_loss += float(loss) * len_bach_size
        total_acc += float(acc) * len_bach_size
    return total_loss / total_examples, total_acc / total_examples

def get_accuracy(list_action_prob, list_targets):
    acc = 0
    for i in range(len(list_action_prob)):
        ind = int(torch.argmax(list_action_prob[i]))
        if float(list_targets[i][ind]) != 0:
            acc += 1
    acc = acc / len(list_action_prob)
    return acc

def apply_padding(tensor_list):
    padding = int(max([i.size() for i in tensor_list])[0])
    return [F.pad(tensor, (0, padding - tensor.size()[0]), mode="constant", value=0) for tensor in tensor_list]

def generate_expert_observations(instance):
    jobs, operations = instance["jobs"], instance["operations"]
    instances = [instance]
    step_list = instance["result"][2]
    env = FJSSPEnv(instances)
    obs = env.reset()
    for s in step_list:
        s["pending"] = env.all_pendings[env.jobs[int(s["job_id"])][s["task_id"]]]
    sorted_steps = sorted(step_list, key=lambda k: (k["start"], k["end"], -k["pending"]))
    expert_observations = []
    index_step = 0
    num_actions = []
    while True:
        selected_jobs = []
        selected_machines = []
        options = obs[('machine', 'exec', 'job')]
        sel_index = []
        counter = 0
        first_end = None
        counter_wrong = 0
        for i in range(index_step, len(sorted_steps)):
            op_s = sorted_steps[i]
            if int(op_s["machine"]) in selected_machines or int(op_s["job_id"]) in selected_jobs or counter > 1:
                counter_wrong += 1
                break
            for o_i in range(len(options["edge_index"][0])):
                o = options["edge_index"][:, o_i]
                if o[0] == int(op_s["machine"]) and o[1] == int(op_s["job_id"]):
                    sel_index.append(o_i)
                    selected_jobs.append(int(op_s["job_id"]))
                    selected_machines.append(int(op_s["machine"]))
                    counter += 1
                    if first_end is None:
                        first_end = int(op_s["end"])
                    break
        new_obs = env.normalize_state(obs)
        if int(obs['machine', 'exec', 'job'].edge_index.shape[1]) > 4:
            aux = torch.zeros(obs['machine', 'exec', 'job'].edge_index.shape[1])
            aux[sel_index] = 1/len(sel_index)
            new_obs['machine', 'exec', 'job'].y = copy.deepcopy(aux)
            expert_observations.append(copy.deepcopy(new_obs))
            num_actions.append(int(obs['machine', 'exec', 'job'].edge_index.shape[1]))
        else:
            break
        obs, _, done, _ = env.step(sel_index[0])
        index_step += 1
        if index_step == len(sorted_steps):
            break
    return expert_observations, num_actions

def generate_instances_train(n_cases, episode, number_group):
    all_data = []
    for i in range(1, number_group):
        file_path = f"data/opt_results/opt_results_{episode+i}.json"
        if not os.path.exists(file_path):
            max_machines = random.randint(4, 9)
            meq = random.randint(2, max_machines)
            max_opers = random.randint(6, 9)
            desv_opers = random.randint(0, 3)
            max_processing = random.randint(5, 10)
            desv_proc = 0.2
            train_config = {
                "n_cases": n_cases,
                "range_jobs": (12, 13),
                "range_machines": (max_machines, max_machines+1),
                "range_op_per_job": (max_opers - desv_opers, max_opers),
                "meq": meq,
                "max_processing": max_processing,
                "desv_proc": desv_proc
            }
            data = generate_train_instances(train_config)
            with open(file_path, "w") as outfile:
                json.dump(data, outfile)
        else:
            with open(file_path, "r") as f:
                data = json.load(f)
        all_data.extend(data)
    
    expert_observations = []
    num_actions = []
    for d in all_data:
        a, b = generate_expert_observations(d)
        expert_observations.extend(a)
        num_actions.extend(b)
    expert_observations = [expert_observations[x] for x in np.argsort(num_actions)]
    return expert_observations

def start_train(args):
    os.makedirs("data/opt_results/", exist_ok=True)
    os.makedirs("./models/", exist_ok=True)

    expert_observations = generate_instances_train(50, -1, 2)
    model = Model(hidden_channels=args.hidden_channels, metadata=expert_observations[0].metadata(), num_layers=args.num_layers, heads=args.heads)
    model = model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate)

    best_val_result = float('inf')
    best_model = None

    for episode in range(args.episodes):
        model = model.to(device)
        expert_observations = generate_instances_train(args.num_cases, ((args.number_group-1)*episode)%(200-args.number_group-2)+1, args.number_group)
        train_loader = DataLoader(expert_observations, batch_size=args.batch_size)

        for epoch in range(1, args.epochs + 1):
            loss, acc = train(model, optimizer, train_loader)
            print(f"Episode: {episode}, Epoch: {epoch}, Loss: {loss:.4f}, Acc: {acc:.4f}")

        if episode > 20:
            val_res = evaluate_val(model, "data/opt_results/opt_results_0.json")
            print(f"Episode: {episode}, Val: {val_res[0]}")

            if val_res[0] < best_val_result:
                best_val_result = val_res[0]
                best_model = copy.deepcopy(model)


    # Save the best model
    if best_model is not None:
        best_model_path = f"./models/model.pt"
        torch.save(best_model.state_dict(), best_model_path)
        print(f"Best model saved with validation result: {best_val_result:.4f}")
    
    return best_val_result

if __name__ == "__main__":
    args = parse_arguments()
    best_val_result = start_train(args)
    print(f"Training completed. Best validation result: {best_val_result:.4f}")
