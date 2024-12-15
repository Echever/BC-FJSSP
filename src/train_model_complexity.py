import json
import os
import pickle
from typing import List, Dict
import numpy as np
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.model_selection import train_test_split

def get_file_list(directory: str) -> List[str]:
    """Get a list of files in the specified directory."""
    return [f for f in os.listdir(directory) if os.path.isfile(os.path.join(directory, f))]

def load_json_data(file_path: str) -> Dict:
    """Load JSON data from a file."""
    with open(file_path, 'r') as json_file:
        return json.load(json_file)

def extract_features(data: Dict) -> List[float]:
    """Extract features from the data."""
    feature = [
        len(data["jobs"]),
        len(data["operations"]),
        len(data["operations"][0])
    ]

    empty = np.zeros(len(data["operations"][0]))
    mean_options = []
    for o in data["operations"]:
        arro = np.array(o)
        empty[arro != 0] += 1
        mean_options.append(np.sum(arro != 0))

    feature.append(np.mean(mean_options))
    
    all_operations = [item for row in data["operations"] for item in row if item != 0]

    feature.extend([
        np.min(all_operations),
        np.max(all_operations),
        np.max(all_operations) - np.min(all_operations),
        np.mean(empty),
        np.std(empty),
        np.mean((empty - np.mean(empty)) ** 3) / (np.var(empty) ** 1.5 + 1e-7)
    ])

    return feature

def calculate_y(data: Dict) -> float:
    """Calculate the Y value."""
    for i, v in enumerate(np.array(data["solution_sec"]) < len(data["operations"]) * 0.01):
        if not v:
            break
    return data["solution_score"][-1] / data["solution_score"][i]

def main():
    dir_path = 'data/data_complexity'
    res = get_file_list(dir_path)
    print(f"Files found: {res}")

    all_features = []
    Ys = []

    for r in res:
        try:
            data = load_json_data(os.path.join(dir_path, r))
            all_features.append(extract_features(data))
            Ys.append(calculate_y(data))
        except Exception as e:
            print(f"Error processing file {r}: {e}")

    X = np.array(all_features)
    y = np.array(Ys)

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, shuffle=False)

    model = GradientBoostingRegressor()
    model.fit(X_train, y_train)

    try:
        with open('models/model_complex.pkl', 'wb') as f:
            pickle.dump(model, f)
        print("Model saved successfully.")
    except Exception as e:
        print(f"Error saving model: {e}")

if __name__ == "__main__":
    main()
