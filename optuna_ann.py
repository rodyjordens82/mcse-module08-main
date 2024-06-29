""" Simple ANN script """

import time
import joblib
import torch
import torch.nn as nn
from torch.cuda.amp import GradScaler
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from helper_funcs import load_and_encode, print_and_calc_metrics
from own_models import ANN, Own_Models
from config import (
    USE_SAVED_DATA,
    CSV_FILE,
    ENCODED_FILE,
    TRAINING_STATS,
    VALIDATE_USE_SAVED_DATA,
    VALIDATE_CSV,
    VALIDATE_ENCODED_FILE,
    SEPERATOR,
    VALIDATE_SEPERATOR,
    CLEANED,
)


import optuna
def append_to_file(file_path, data):
    try:
        with open(file_path, 'a') as file:  # 'a' mode opens the file for appending
            file.write(data + '\n')  # Appends the data with a newline character
        print("Data appended successfully.")
    except Exception as e:
        print(f"An error occurred: {e}")


class Objective:
    def __init__(
        self,
        model: Own_Models,
        input_size: int,
        output_size: int,
        x: torch.Tensor,
        y: torch.Tensor,
        x_val: torch.Tensor,
        y_val: torch.Tensor,
        epochs: int,
        device: torch.device,
        start_time: float,
    ):
        self.model = model
        self.input_size = input_size
        self.output_size = output_size
        self.x = x
        self.y = y
        self.x_val = x_val
        self.y_val = y_val
        self.epochs = epochs
        self.device = device
        self.start_time = start_time

    def __call__(self, trial: optuna.Trial):
        # Hyperparameter search space
        hidden_size = trial.suggest_int("hidden_size", 16, 128, log=True)
        learning_rate = trial.suggest_float("learning_rate", 1e-4, 1e-1, log=True)
        weight_decay = trial.suggest_float('weight_decay', 1e-4, 1e-1, log=True)
        epochs = trial.suggest_int("epochs", 1, 200)

        model = self.model(
            input_dim=self.input_size,
            hidden_dim=hidden_size,
            output_dim=self.output_size,
            start_time=self.start_time,
            device=self.device,
        )
        optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
        criterion = nn.CrossEntropyLoss()
        gradscaler = GradScaler()

        # Training loop
        model.train_model(
            criterion=criterion,
            optimizer=optimizer,
            scaler=gradscaler,
            x_tensor=self.x,
            y_tensor=self.y,
            num_epochs=epochs, # self.epochs,
            dataloader_trainer=False,
            training_stats=TRAINING_STATS,
            do_not_eval=True,
            trial=trial,
        )
        
        # Example usage
        file_path = r'./ann_results.txt'
        data = f'{hidden_size=}\t{learning_rate=}\t{weight_decay=}\t{epochs=}\n{model.training_stats}'
        print(data)
        append_to_file(file_path, data)

        # Validation (use your own validation dataset here)
        # For simplicity, we use the training loss as a proxy
        pred, y, loss = model.validate_model(criterion, self.x_val, self.y_val)
        accuracy, precision, recall, f1, uac, _ = print_and_calc_metrics(
            y, pred, "train"
        )
        return loss


#######################################################################################################################

if __name__ == "__main__":
    # Set default device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.set_default_device(device)

    start_time = time.time()
    if not USE_SAVED_DATA:
        X, y, label_encoders, label_encoder_y = load_and_encode(
            CSV_FILE, ENCODED_FILE, start_time, sep=SEPERATOR, cleaned=CLEANED
        )
    else:
        print(
            f"{(time.time() - start_time):.5f}\t Loading encoded dataset from {ENCODED_FILE}"
        )
        X, y, label_encoders, label_encoder_y = joblib.load(ENCODED_FILE)

    # Split the dataset into training and testing sets
    print(f"{(time.time() - start_time):.5f}\t Splitting test and train")
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

    # Scale the features
    print(f"{(time.time() - start_time):.5f}\t Performing standardscaler")
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    # Convert data to PyTorch tensors
    print(f"{(time.time() - start_time):.5f}\t Converting to tensors")
    X_train_tensor = torch.tensor(X_train, dtype=torch.float32).to("cpu").detach()
    y_train_tensor = torch.tensor(y_train, dtype=torch.long).to("cpu").detach()
    X_test_tensor = torch.tensor(X_test, dtype=torch.float32).to("cpu").detach()
    y_test_tensor = torch.tensor(y_test, dtype=torch.long).to("cpu").detach()

    if not VALIDATE_USE_SAVED_DATA:
        print(f"{(time.time() - start_time):.5f}\t Loading dataset from {VALIDATE_CSV}")
        (
            X_validate,
            y_validate,
            label_encoders_validate,
            label_encoder_y_validate,
        ) = load_and_encode(
            VALIDATE_CSV,
            VALIDATE_ENCODED_FILE,
            start_time,
            label_encoders=(label_encoders, label_encoder_y),
            sep=VALIDATE_SEPERATOR,
            cleaned=CLEANED,
        )
    else:
        print(
            f"{(time.time() - start_time):.5f}\t Loading encoded dataset from {VALIDATE_ENCODED_FILE}"
        )
        (
            X_validate,
            y_validate,
            label_encoders_validate,
            label_encoder_y_validate,
        ) = joblib.load(VALIDATE_ENCODED_FILE)

    print(
        f"{(time.time() - start_time):.5f}\t Performing standardscaler for validation set"
    )
    # Scale the features
    X_validate = scaler.transform(X_validate)

    print(f"{(time.time() - start_time):.5f}\t Converting validation set to tensors")
    # Convert data to PyTorch tensors
    X_validate_tensor = torch.tensor(X_validate, dtype=torch.float32).to("cpu").detach()
    y_validate_tensor = torch.tensor(y_validate, dtype=torch.long).to("cpu").detach()

    # Define hyperparameters
    INPUT_DIM = X_train.shape[1]
    OUTPUT_DIM = len(np.unique(y))

    # Optuna
    objective = Objective(
        model=ANN,
        input_size=INPUT_DIM,
        output_size=OUTPUT_DIM,
        x=X_train_tensor,
        y=y_train_tensor,
        x_val=X_validate_tensor,
        y_val=y_validate_tensor,
        epochs=1,
        device=device,
        start_time=start_time,
    )

    study = optuna.create_study(direction="minimize")
    study.optimize(objective, n_trials=100)  # Number of trials

    print("Best trial:")
    trial = study.best_trial
    print("  Value: {}".format(trial.value))
    print("  Params: ")
    for key, value in trial.params.items():
        print(f"\t{key}: {value}")
