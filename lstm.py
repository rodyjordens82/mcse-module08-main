import time
import joblib
import torch
import torch.nn as nn
from torch.cuda.amp import GradScaler
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from helper_funcs import load_and_encode, print_and_calc_metrics
from own_models import LSTM
from config import (
    USE_SAVED_DATA,
    CSV_FILE,
    ENCODED_FILE,
    TRAINING_STATS,
    VALIDATE_SET,
    VALIDATE_USE_SAVED_DATA,
    VALIDATE_CSV,
    VALIDATE_ENCODED_FILE,
    VALIDATE_SEPERATOR,
    SEPERATOR,
    CLEANED,
    # BATCH_SIZE,
    # NUM_WORKERS,
    # AMOUNT_OF_BATCHES,
)


if __name__ == "__main__":
    # Set default device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.set_default_device(device)
    
    # Variables/hyperparameters
    HIDDEN_SIZE = 64
    NUM_LAYERS = 2
    LEARNING_RATE = int(1e-2)
    NUM_EPOCHS = int(5e0)

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

    print(f"{(time.time() - start_time):.5f}\t Splitting Test and Train Data")
    # Split the dataset into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    print(f"{(time.time() - start_time):.5f}\t Performing StandardScaler")
    # Scale the features
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    print(f"{(time.time() - start_time):.5f}\t Converting to Tensors")
    # Convert data to PyTorch tensors
    X_train_tensor = torch.tensor(X_train, dtype=torch.float32).to("cpu").detach()
    y_train_tensor = torch.tensor(y_train, dtype=torch.long).to("cpu").detach()
    X_test_tensor = torch.tensor(X_test, dtype=torch.float32).to("cpu").detach()
    y_test_tensor = torch.tensor(y_test, dtype=torch.long).to("cpu").detach()
    X_train_tensor = X_train_tensor.unsqueeze(1)
    X_test_tensor = X_test_tensor.unsqueeze(1)

    # Create DataLoader for training and testing sets
    # train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
    # train_loader = DataLoader(
    #     train_dataset,
    #     batch_size=int(len(train_dataset) // AMOUNT_OF_BATCHES),
    #     shuffle=True,
    #     generator=torch.Generator(device=device),
    #     num_workers=NUM_WORKERS,
    # )
    # test_dataset = TensorDataset(X_test_tensor, y_test_tensor)
    # test_loader = DataLoader(
    #     test_dataset,
    #     batch_size=int(len(train_dataset) // AMOUNT_OF_BATCHES),
    #     shuffle=True,
    #     generator=torch.Generator(device=device),
    #     num_workers=NUM_WORKERS,
    # )

    # Define hyperparameters
    INPUT_SIZE = X_train.shape[1]
    OUTPUT_SIZE = len(np.unique(y))

    # Initialize the model
    model = LSTM(
        INPUT_SIZE,
        HIDDEN_SIZE,
        NUM_LAYERS,
        OUTPUT_SIZE,
        start_time=start_time,
        device=device,
    ).to(device)

    # Define loss function and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
    gradscaler = GradScaler()

    train_pred, y_train = model.train_model(
        criterion=criterion,
        optimizer=optimizer,
        scaler=gradscaler,
        # loader=train_loader,
        x_tensor=X_train_tensor,
        y_tensor=y_train_tensor,
        num_epochs=NUM_EPOCHS,
        training_stats=TRAINING_STATS,
    )

    print(model.get_training_stats())

    # Test the model
    y_test_result, test_pred = model.eval_model(X_test_tensor, y_test_tensor)
    # Calculate metrics on training set
    print(f"{(time.time() - start_time):.5f}\t Train stats:")
    print_and_calc_metrics(y_train, train_pred, "train")

    # Calculate metrics on test set
    print(f"{(time.time() - start_time):.5f}\t Test stats:")
    print_and_calc_metrics(y_test_result, test_pred, "test")

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
        X_validate_tensor = X_validate_tensor.unsqueeze(1)

        # Test the model
        validate_pred, y_validate = model.eval_model(X_validate_tensor, y_validate_tensor)

        # Calculate metrics on validation set
        print(f"{(time.time() - start_time):.5f}\t Validation stats:")
        print_and_calc_metrics(y_validate, validate_pred, "validate")
