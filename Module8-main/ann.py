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
from own_models import ANN
from config import (
    USE_SAVED_DATA,
    CSV_FILE,
    ENCODED_FILE,
    TRAINING_STATS,
    VALIDATE_SET,
    VALIDATE_USE_SAVED_DATA,
    VALIDATE_CSV,
    VALIDATE_ENCODED_FILE,
    SEPERATOR,
    VALIDATE_SEPERATOR,
    CLEANED,
    # AMOUUNT_OF_BATCHES,
    # NUM_WORKERS,    
)


if __name__ == "__main__":
    # Set default device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.set_default_device(device)
    
    # Variables/hyperparameters
    HIDDEN_DIM = 52
    LEARNING_RATE = 0.00139089817931236
    NUM_EPOCHS = 193
    WEIGHT_DECAY = 0.00921546221157035

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

    # Define hyperparameters
    INPUT_DIM = X_train.shape[1]
    OUTPUT_DIM = len(np.unique(y))

    # Initialize the model
    print(f"{(time.time() - start_time):.5f}\t Initializing model")
    model = ANN(INPUT_DIM, HIDDEN_DIM, OUTPUT_DIM, start_time=start_time, device=device)

    # Define loss function and optimizer
    print(f"{(time.time() - start_time):.5f}\t Defining loss model and optimizing")
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
    gradscaler = GradScaler()
    
    train_pred, y_train = model.train_model(
        criterion=criterion,
        optimizer=optimizer,
        scaler=gradscaler,
        x_tensor=X_train_tensor,
        y_tensor=y_train_tensor,
        num_epochs=NUM_EPOCHS,
        dataloader_trainer=False,
        training_stats=TRAINING_STATS,
    )
    print(model.get_training_stats())

    y_test, test_pred = model.eval_model(X_test_tensor, y_test_tensor)

    # Calculate metrics on training set
    print(f"{(time.time() - start_time):.5f}\t Train stats:")
    print_and_calc_metrics(y_train, train_pred, "train")

    # Calculate metrics on test set
    print(f"{(time.time() - start_time):.5f}\t Test stats:")
    print_and_calc_metrics(y_test, test_pred, "test")

    if VALIDATE_SET:
        if not VALIDATE_USE_SAVED_DATA:
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

        print(
            f"{(time.time() - start_time):.5f}\t Converting validation set to tensors"
        )
        # Convert data to PyTorch tensors
        X_validate_tensor = torch.tensor(X_validate, dtype=torch.float32).to("cpu").detach()
        y_validate_tensor = torch.tensor(y_validate, dtype=torch.long).to("cpu").detach()

        validate_predicted, y_validate = model.eval_model(
            X_validate_tensor, y_validate_tensor
        )

        # Calculate metrics on validation set
        print(f"{(time.time() - start_time):.5f}\t Validation stats:")
        print_and_calc_metrics(y_validate, validate_predicted, "validate")
