import time
import numpy as np
import torch
import torch.nn as nn
from torch.cuda.amp import GradScaler
import joblib
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from own_models import ANN, SVM, LSTM, MetaLearner
from helper_funcs import print_and_calc_metrics, load_and_encode
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
    # BATCH_SIZE,
    # AMOUNT_OF_BATCHES,
    # NUM_WORKERS,
    CLEANED,
)

if __name__ == "__main__":
    # Set default device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.set_default_device(device)
    
    # Variables/hyperparameters
    HIDDEN_SIZE = 64
    NUM_LAYERS = 2
    LEARNING_RATE = 0.01
    NUM_EPOCHS = 5

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
    print(f"{(time.time() - start_time):.5f}\t Splitting Test and Train Data")
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

    # Scale the features
    print(f"{(time.time() - start_time):.5f}\t Performing StandardScaler")
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    # Creating tensors
    print(f"{(time.time() - start_time):.5f}\t Converting to Tensors")
    X_train_tensor = torch.tensor(X_train, dtype=torch.float32).to("cpu").detach()
    y_train_tensor = torch.tensor(y_train, dtype=torch.long).to("cpu").detach()
    X_test_tensor = torch.tensor(X_test, dtype=torch.float32).to("cpu").detach()
    y_test_tensor = torch.tensor(y_test, dtype=torch.long).to("cpu").detach()
    x_train_tensor_unsqueezed = X_train_tensor.unsqueeze(1)
    X_test_tensor_unsqueezed = X_test_tensor.unsqueeze(1)

    # Define hyperparameters
    INPUT_SIZE = X_train.shape[1]
    OUTPUT_SIZE = len(np.unique(y))

    lstm = LSTM(INPUT_SIZE, HIDDEN_SIZE, NUM_LAYERS, OUTPUT_SIZE, start_time, device)
    ann = ANN(INPUT_SIZE, HIDDEN_SIZE, OUTPUT_SIZE, start_time, device)
    svm = SVM(INPUT_SIZE, start_time, device)

    criterion = nn.CrossEntropyLoss()
    svm_criterion = nn.HingeEmbeddingLoss()
    lstm_optimizer = torch.optim.Adam(lstm.parameters(), lr=LEARNING_RATE)
    ann_optimizer = torch.optim.Adam(ann.parameters(), lr=LEARNING_RATE)
    svm_optimizer = torch.optim.SGD(svm.parameters(), lr=LEARNING_RATE)
    gradscaler = GradScaler()

    # Training models
    print(f"{(time.time() - start_time):.5f}\t Training models")
    lstm_train_pred, lstm_train_label = lstm.train_model(
        criterion=criterion,
        optimizer=lstm_optimizer,
        x_tensor=x_train_tensor_unsqueezed,
        y_tensor=y_train_tensor,
        num_epochs=NUM_EPOCHS,
        training_stats=TRAINING_STATS,
    )
    ann_train_pred, ann_train_label = ann.train_model(
        criterion=criterion,
        optimizer=ann_optimizer,
        x_tensor=X_train_tensor,
        y_tensor=y_train_tensor,
        num_epochs=NUM_EPOCHS,
        dataloader_trainer=True,
        training_stats=TRAINING_STATS,
    )
    svm_train_pred, svm_train_label = svm.train_model(
        criterion=svm_criterion,
        optimizer=svm_optimizer,
        x_tensor=X_train_tensor,
        y_tensor=y_train_tensor,
        num_epochs=NUM_EPOCHS,
        training_stats=TRAINING_STATS,
    )

    # Testing models
    print(f"{(time.time() - start_time):.5f}\t Testing models")
    lstm_test_pred, lstm_test_labels = lstm.eval_model(
        X_test_tensor_unsqueezed, y_test_tensor
    )
    ann_test_pred, ann_test_labels = ann.eval_model(X_test_tensor, y_test_tensor)
    svm_test_pred, svm_test_labels = svm.eval_model(X_test_tensor, y_test_tensor)

    # Calculate metrics
    print(f"{(time.time() - start_time):.5f}\t Calculating metrics")
    print_and_calc_metrics(lstm_train_label, lstm_train_pred, "LSTM Train")
    print_and_calc_metrics(ann_train_label, ann_train_pred, "ANN Train")
    print_and_calc_metrics(svm_train_label, svm_train_pred, "SVM Train")
    print_and_calc_metrics(lstm_test_labels, lstm_test_pred, "LSTM Test")
    print_and_calc_metrics(ann_test_labels, ann_test_pred, "ANN Test")
    print_and_calc_metrics(svm_test_labels, svm_test_pred, "SVM Test")

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

        X_validate = scaler.transform(X_validate)
        X_validate_tensor = torch.tensor(X_validate, dtype=torch.float32).to("cpu").detach()
        y_validate_tensor = torch.tensor(y_validate, dtype=torch.long).to("cpu").detach()
        x_validate_tensor_unsqueezed = X_validate_tensor.unsqueeze(1)

        lstm_validate_pred, lstm_validate_labels = lstm.eval_model(
            x_validate_tensor_unsqueezed, y_validate_tensor
        )
        ann_validate_pred, ann_validate_labels = ann.eval_model(
            X_validate_tensor, y_validate_tensor
        )
        svm_validate_pred, svm_validate_labels = svm.eval_model(
            X_validate_tensor, y_validate_tensor
        )

        print(f"{(time.time() - start_time):.5f}\t Calculating validation metrics")
        print_and_calc_metrics(
            lstm_validate_labels, lstm_validate_pred, "LSTM Validate"
        )
        print_and_calc_metrics(ann_validate_labels, ann_validate_pred, "ANN Validate")
        print_and_calc_metrics(svm_validate_labels, svm_validate_pred, "SVM Validate")

    # Combine predictions
    combined_train_preds = np.column_stack(
        (
            lstm_train_pred,
            ann_train_pred,
            svm_train_pred,
        )
    )
    combined_test_preds = np.column_stack(
        (
            lstm_test_pred,
            ann_test_pred,
            svm_test_pred,
        )
    )

    # Create tensors for meta-learner
    combined_train_preds_tensor = torch.tensor(
        combined_train_preds, dtype=torch.float32
    )
    # y_train_tensor = torch.tensor(y_train, dtype=torch.long).to("cpu").detach()
    combined_test_preds_tensor = torch.tensor(
        combined_test_preds, dtype=torch.float32
    ).to("cpu").detach()
    # y_test_tensor = torch.tensor(y_test, dtype=torch.long).to("cpu").detach()

    # Define and train meta-learner
    metalearner = MetaLearner(
        input_dim=3,
        hidden_dim=50,
        output_dim=OUTPUT_SIZE,
        start_time=start_time,
        device=device,
    )
    meta_criterion = nn.CrossEntropyLoss()
    meta_optimizer = torch.optim.Adam(metalearner.parameters(), lr=LEARNING_RATE)

    print(f"{(time.time() - start_time):.5f}\t Training Meta-Learner")
    meta_train_pred, meta_train_label = metalearner.train_model(
        criterion=meta_criterion,
        optimizer=meta_optimizer,
        x_tensor=combined_train_preds_tensor,
        y_tensor=y_train_tensor,
        num_epochs=NUM_EPOCHS,
        training_stats=TRAINING_STATS,
    )

    print(f"{(time.time() - start_time):.5f}\t Testing MetaLeaner")
    meta_test_pred, meta_test_label = metalearner.eval_model(
        x_tensor=combined_test_preds_tensor, y_tensor=y_test_tensor
    )

    print(f"{(time.time() - start_time):.5f}\t Calculating metrics for Meta-Learner")
    print_and_calc_metrics(meta_train_label, meta_train_pred, "Meta-Learner Train")
    print_and_calc_metrics(meta_test_label, meta_test_pred, "Meta-Learner Test")

    if VALIDATE_SET:
        combined_validate_preds = np.column_stack(
            (
                lstm_validate_pred,
                ann_validate_pred,
                svm_validate_pred,
            )
        )
        combined_validate_preds_tensor = torch.tensor(
            combined_validate_preds, dtype=torch.float32
        ).to("cpu").detach()

        meta_validate_pred, meta_validate_label = metalearner.validate_model(
            combined_validate_preds_tensor, y_validate_tensor
        )

        print(
            f"{(time.time() - start_time):.5f}\t Calculating validation metrics for Meta-Learner"
        )
        print_and_calc_metrics(
            meta_validate_label, meta_validate_pred, "Meta-Learner Validate"
        )
