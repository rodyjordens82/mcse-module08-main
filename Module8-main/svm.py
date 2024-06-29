import time
import joblib
import torch
import torch.nn as nn
import torch.optim as optim
from torch.cuda.amp import GradScaler
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from helper_funcs import load_and_encode, print_and_calc_metrics
from own_models import SVM
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
    # AMOUNT_OF_BATCHES
)

if __name__ == "__main__":
    # Set default device/hyperparameters
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.set_default_device(device)

    # Variables
    NUM_EPOCHS = 10
    LEARNING_RATE = 0.0009668031124392923
    WEIGHT_DECAY = 0.05775496309321937

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
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

    print(f"{(time.time() - start_time):.5f}\t Performing StandardScaler")
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    print(f"{(time.time() - start_time):.5f}\t Setting up tensors")
    X_train_tensor = torch.tensor(X_train, dtype=torch.float32).to("cpu").detach()
    y_train_tensor = torch.tensor(y_train, dtype=torch.long).to("cpu").detach()
    X_test_tensor = torch.tensor(X_test, dtype=torch.float32).to("cpu").detach()
    y_test_tensor = torch.tensor(y_test, dtype=torch.long).to("cpu").detach()

    # criterion = nn.HingeEmbeddingLoss()
    criterion = nn.MSELoss()
    gradscaler = GradScaler()
    model = SVM(
        X_train.shape[1],
        start_time=start_time,
        device=device,
    )

    optimizer = optim.SGD(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)

    train_pred_binary, y_train = model.train_model(
        criterion=criterion,
        optimizer=optimizer,
        scaler=gradscaler,
        x_tensor=X_train_tensor,
        y_tensor=y_train_tensor,
        num_epochs=NUM_EPOCHS,
        training_stats=TRAINING_STATS,
    )
    print(model.get_training_stats())

    test_pred_binary, y_test = model.eval_model(X_test_tensor, y_test_tensor)

    print(f"{(time.time() - start_time):.5f}\t SVM Metrics:")
    print_and_calc_metrics(y_train, train_pred_binary, "train")
    print_and_calc_metrics(y_test, test_pred_binary, "test")

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

        X_validate = scaler.transform(X_validate)
        X_validate_tensor = torch.tensor(X_validate, dtype=torch.float32).to("cpu").detach()
        y_validate_tensor = torch.tensor(y_validate).to("cpu").detach()

        validate_pred_binary, y_validate = model.eval_model(
            x_tensor=X_validate_tensor,
            y_tensor=y_validate_tensor,
        )
        print(f"{(time.time() - start_time):.5f}\t Validation stats:")
        print_and_calc_metrics(y_validate, validate_pred_binary, "validate")
