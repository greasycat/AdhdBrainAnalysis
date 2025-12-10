from pathlib import Path
import torch
from cnn.dataset import EstimateDataset
from torch.utils.data import DataLoader, Subset
from cnn.train import train_model
from cnn.model3 import CNN
from cnn.model2 import ViT3D
import pandas as pd
import argparse

parser = argparse.ArgumentParser(description='Perform analysis on CNP task data')
parser.add_argument('--demo', action='store_true',
                    help='run in demo mode')

def main():
    args = parser.parse_args()
    dataset = EstimateDataset(Path("derivatives/task"))
    input_dim = dataset.get_input_dim()
    df = pd.DataFrame()
    if args.demo:
        model = CNN(input_dim=input_dim, input_size=(65, 77, 49), num_classes=2)
        train_subset, val_subset = dataset.train_test_split(test_size=0.2, random_state=256)
        results = train_model(model, train_subset, val_subset, batch_size=4, epochs=20)
        results = pd.DataFrame(results)
        results.to_csv("demo_results.csv", index=False)
        print("Demo results saved to demo_results.csv. They do not reflect the model performance and no cross-validation is performed!! \n if you want to run the full analysis, do not use the --demo flag.")
        pass
    else:
        for i, (train_index, val_index) in enumerate(dataset.cross_validation_split(n_splits=5, random_state=256)):
            print(len(train_index), len(val_index))
            model = ViT3D(input_dim=input_dim, input_size=(65, 77, 49), num_classes=2)
            model.to("cuda" if torch.cuda.is_available() else "cpu")
            train_subset = Subset(dataset, train_index.tolist())
            val_subset = Subset(dataset, val_index.tolist())
            results = train_model(model, train_subset, val_subset, batch_size=4, epochs=20)
            results = pd.DataFrame(results)
            results["train_index"] = i
            df = pd.concat([df, results])
        df.to_csv("results.csv", index=False)



if __name__ == "__main__":
    main()
