import time
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.utils
from torch import cuda
from torch.utils.data import DataLoader, Dataset, TensorDataset
from scipy.sparse import csr_matrix
from helper_funcs import print_and_calc_metrics
from config import AMOUNT_OF_BATCHES, NUM_WORKERS, REPORT_PER_BATCH, REPORT_PER_EPOCH
from optuna.exceptions import TrialPruned
from optuna import Trial

class Own_Models():
    def __init__(self, start_time: float, device: torch.device) -> None:
        self.start_time = start_time
        self.device = device
        self.training_stats = {
            "train": {
                "epoch": [],
                "accuracy": [],
                "precision": [],
                "recall": [],
                "f1": [],
                "roc_auc": [],
            },
            # "test": {
            #     "accuracy": [],
            #     "precision": [],
            #     "recall": [],
            #     "f1": [],
            #     "roc_auc": [],
            # },
        }
        
    def get_training_stats(self):
        if len(self.training_stats["train"]["accuracy"]) == 0:
            print("Training stats are empty. Please train the model first and enable traing_stats.")
        return self.training_stats

    def train_model(self):
        raise NotImplementedError("train_model method not implemented.")
    def eval_model(self):
        raise NotImplementedError("eval_model method not implemented.")
    def validate_model(self):
        raise NotImplementedError("validate_model method not implemented.")

# Define the ANN model
class ANN(nn.Module, Own_Models):
    def __init__(
        self, input_dim, hidden_dim, output_dim, start_time: float, device: torch.device
    ):
        nn.Module.__init__(self)
        Own_Models.__init__(self, start_time, device)
        # super(ANN, self).__init__(start_time, device)
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden_dim, output_dim)
        

    def forward(self, x):
        out = self.fc1(x)
        out = self.relu(out)
        out = self.fc2(out)
        return out

    def train_model(
        self,
        criterion: nn.CrossEntropyLoss,
        optimizer: torch.optim.Optimizer,
        scaler: torch.cuda.amp.GradScaler = None,
        loader: DataLoader = None,
        x_tensor: torch.Tensor = None,
        y_tensor: torch.Tensor = None,
        num_epochs: int = 10,
        accumulation_steps: int = 4,
        training_stats: bool = False,
        dataloader_trainer: bool = False,
        amount_of_batches: int = AMOUNT_OF_BATCHES,
        num_workers: int = NUM_WORKERS,
        do_not_eval: bool = False,
        trial: Trial = 0
    ) -> tuple[np.ndarray, np.ndarray]:

        print(f"{(time.time() - self.start_time):.5f}\t Training ANN")
        if (
            dataloader_trainer
        ):  # flip this to change between DataLoader and manual training, True is DataLoader.
            # Train the model
            dataset = TensorDataset(x_tensor, y_tensor)
            loader = DataLoader(
                dataset,
                batch_size=int(len(dataset) // amount_of_batches),
                shuffle=False,
                generator=torch.Generator(device=self.device),
                num_workers=num_workers,
            )
            for epoch in range(num_epochs):
                self.train()
                running_loss = 0.0
                for i, data in enumerate(loader, 0):
                    inputs, labels = data[0].to(self.device), data[1].to(self.device)
                    optimizer.zero_grad()

                    if scaler is not None:
                        with torch.cuda.amp.autocast():
                            outputs = self(inputs)
                            loss = criterion(outputs, labels)

                        scaler.scale(loss).backward()
                        if (i + 1) % accumulation_steps == 0:
                            scaler.step(optimizer)
                            scaler.update()
                            optimizer.zero_grad()
                    else:
                        outputs = self(inputs)
                        loss = criterion(outputs, labels)
                        loss.backward()
                        optimizer.step()
                        optimizer.zero_grad()
                    running_loss += loss.item()
                    if (i + 1) % REPORT_PER_BATCH == 0:  # print every 2000 mini-batches
                        print(
                            f"{(time.time() - self.start_time):.5f}\t [{epoch + 1}, {i + 1:5d}/{len(loader)}] loss: {running_loss / REPORT_PER_BATCH:.3f}"
                        )
                        running_loss = 0.0
                if (epoch + 1) % REPORT_PER_EPOCH == 0:
                    print(
                        f"{(time.time() - self.start_time):.5f}\t Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item():.5f}"
                    )
                    if training_stats:
                        predicted = torch.argmax(outputs, dim=1)
                        y = y_tensor.cpu().numpy()
                        pred = predicted.cpu().numpy()
                        accuracy, precision, recall, f1, uac, _ =  print_and_calc_metrics(y, pred, "train")
                        self.training_stats["train"]["epoch"].append(epoch+1)
                        self.training_stats["train"]["accuracy"].append(accuracy)
                        self.training_stats["train"]["precision"].append(precision)
                        self.training_stats["train"]["recall"].append(recall)
                        self.training_stats["train"]["f1"].append(f1)
                        self.training_stats["train"]["roc_auc"].append(uac)
                if trial != 0:
                    trial.report(loss.item(), epoch)
                    
                    if trial.should_prune():
                        
                        cuda.empty_cache()
                    raise TrialPruned()
            # Collect predictions for training data
            if do_not_eval:
                cuda.empty_cache()
                return np.array([0]), np.array([0])
            pred_list = []
            self.eval()
            with torch.no_grad():
                for inputs, labels in loader:
                    inputs = inputs.to(self.device)
                    outputs = self(inputs)
                    pred_list.extend(torch.argmax(outputs, dim=1).cpu().numpy())
            pred = np.array(pred_list)
        else:
            x_tensor = x_tensor.to(self.device)
            y_tensor = y_tensor.to(self.device)
            self.train()
            # Train the model
            for epoch in range(num_epochs):
                # Forward pass
                outputs = self(x_tensor)
                loss = criterion(outputs, y_tensor)

                # Backward and optimize
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                if (epoch + 1) % REPORT_PER_EPOCH == 0:
                    print(
                        f"{(time.time() - self.start_time):.5f}\t Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item():.7f}"
                    )
                    if training_stats:
                        predicted = torch.argmax(outputs, dim=1)
                        y = y_tensor.cpu().numpy()
                        pred = predicted.cpu().numpy()
                        accuracy, precision, recall, f1, uac, _ =  print_and_calc_metrics(y, pred, "train")
                        self.training_stats["train"]["epoch"].append(epoch+1)
                        self.training_stats["train"]["accuracy"].append(accuracy)
                        self.training_stats["train"]["precision"].append(precision)
                        self.training_stats["train"]["recall"].append(recall)
                        self.training_stats["train"]["f1"].append(f1)
                        self.training_stats["train"]["roc_auc"].append(uac)
                if trial != 0:
                    trial.report(loss.item(), epoch)
                    if trial.should_prune():
                        cuda.empty_cache()
                        raise TrialPruned()
            if do_not_eval:
                cuda.empty_cache()
                return np.array([0]), np.array([0])
            predicted = torch.argmax(outputs, dim=1)
            pred = predicted.cpu().numpy()
        y = y_tensor.cpu().numpy()
        cuda.empty_cache()
        return pred, y

    def eval_model(
        self,
        x_tensor: torch.Tensor,
        y_tensor: torch.Tensor,
    ) -> tuple[np.ndarray, np.ndarray]:
        print(f"{(time.time() - self.start_time):.5f}\t Testing ANN")
        self.eval()
        with torch.no_grad():
            outputs = self(x_tensor.to(self.device))

        predicted = torch.argmax(outputs, dim=1)
        y = y_tensor.cpu().numpy()
        pred = predicted.cpu().numpy()
        return pred, y

    def validate_model(
        self,
        criterion: nn.CrossEntropyLoss,
        x_tensor: torch.Tensor,
        y_tensor: torch.Tensor,
    ) -> tuple[np.ndarray, np.ndarray, float]:
        print(f"{(time.time() - self.start_time):.5f}\t Evaluating ANN")
        x_tensor = x_tensor.to(self.device)
        y_tensor = y_tensor.to(self.device)
        self.eval()
        with torch.no_grad():
            outputs = self(x_tensor)
            # Calculate the loss
            loss = criterion(outputs, y_tensor)
        predicted = torch.argmax(outputs, dim=1)
        y = y_tensor.cpu().numpy()
        return predicted, y, loss.item()


# Define the SVM model
class SVM(nn.Module, Own_Models):
    def __init__(self, input_dim, start_time: float, device: torch.device):
        nn.Module.__init__(self)
        Own_Models.__init__(self, start_time, device)
        # super(SVM, self).__init__(start_time, device)
        self.linear = nn.Linear(input_dim, 1)

    def forward(self, x):
        return self.linear(x)

    def train_model(
        self,
        criterion: nn.CrossEntropyLoss,
        optimizer: torch.optim.Optimizer,
        x_tensor: torch.Tensor,
        y_tensor: torch.Tensor,
        num_epochs: int = 10,
        scaler: torch.cuda.amp.GradScaler = None,
        accumulation_steps: int = 4,
        training_stats: bool = False,
        amount_of_batches: int = AMOUNT_OF_BATCHES,
        num_workers: int = NUM_WORKERS,
        do_not_eval: bool = False,
        use_data_loader: bool = True,
        trial: Trial = 0
    ) -> tuple[np.ndarray, np.ndarray]:

        if use_data_loader:
            dataset = TensorDataset(x_tensor, y_tensor)
            loader = DataLoader(
                dataset,
                batch_size=int(len(dataset) // amount_of_batches),
                shuffle=False,
                generator=torch.Generator(device=self.device),
                num_workers=num_workers,
            )
            print(f"{(time.time() - self.start_time):.5f}\t Training SVM")
            for epoch in range(num_epochs):
                self.train()
                running_loss = 0.0
                for i, data in enumerate(loader, 0):
                    inputs, labels = data[0].to(self.device), data[1].to(self.device)
                    optimizer.zero_grad()
                    if scaler is not None:
                        with torch.cuda.amp.autocast():
                            outputs = self(inputs)
                            loss = criterion(outputs.squeeze(), labels.float())

                        scaler.scale(loss).backward()
                        if (i + 1) % accumulation_steps == 0:
                            scaler.step(optimizer)
                            scaler.update()
                            optimizer.zero_grad()
                    else:
                        outputs = self(inputs)
                        loss = criterion(outputs.squeeze(), labels.float())
                        loss.backward()
                        optimizer.step()
                        optimizer.zero_grad()
                    running_loss += loss.item()
                    if (i + 1) % REPORT_PER_BATCH == 0:  # print every 2000 mini-batches
                        print(
                            f"{(time.time() - self.start_time):.5f}\t [{epoch + 1}, {i + 1:5d}/{len(loader)}] loss: {running_loss / REPORT_PER_BATCH:.3f}"
                        )
                        running_loss = 0.0
                if (epoch + 1) % REPORT_PER_EPOCH == 0:
                    print(
                        f"{(time.time() - self.start_time):.5f}\t Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item():.5f}"
                    )
                    if training_stats:
                        pred_list = []
                        self.eval()
                        with torch.no_grad():
                            for inputs, labels in loader:
                                inputs = inputs.to(self.device)
                                outputs = self(inputs)
                                pred_list.extend(torch.argmax(outputs, dim=1).cpu().numpy())
                        pred = np.array(pred_list)
                        y = y_tensor.cpu().numpy()
                        accuracy, precision, recall, f1, uac, _ =  print_and_calc_metrics(y, pred, "train")
                        self.training_stats["train"]["epoch"].append(epoch+1)
                        self.training_stats["train"]["accuracy"].append(accuracy)
                        self.training_stats["train"]["precision"].append(precision)
                        self.training_stats["train"]["recall"].append(recall)
                        self.training_stats["train"]["f1"].append(f1)
                        self.training_stats["train"]["roc_auc"].append(uac)
                if trial != 0:
                    trial.report(loss.item(), epoch)
                    
                    if trial.should_prune():
                        
                        raise TrialPruned()
        else:
            x_tensor = x_tensor.to(self.device)
            y_tensor = y_tensor.to(self.device)
            for epoch in range(num_epochs):
                self.train()
                optimizer.zero_grad()
                if scaler is not None:
                    with torch.cuda.amp.autocast():
                        outputs = self(x_tensor)
                        loss = criterion(outputs.squeeze(), y_tensor.float())
                        scaler.scale(loss).backward()
                        if (i + 1) % accumulation_steps == 0:
                            scaler.step(optimizer)
                            scaler.update()
                            optimizer.zero_grad()
                else:
                    outputs = self(x_tensor)
                    loss = criterion(outputs.squeeze(), y_tensor.float())
                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()
                    
                if (epoch + 1) % REPORT_PER_EPOCH == 0:
                    print(
                        f"{(time.time() - self.start_time):.5f}\t Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item():.5f}"
                    )
                    if training_stats:
                        pred_list = []
                        self.eval()
                        with torch.no_grad():
                            outputs = self(x_tensor)
                            pred_list.extend(torch.argmax(outputs, dim=1).cpu().numpy())
                        pred = np.array(pred_list)
                        y = y_tensor.cpu().numpy()
                        accuracy, precision, recall, f1, uac, _ =  print_and_calc_metrics(y, pred, "train")
                        self.training_stats["train"]["epoch"].append(epoch+1)
                        self.training_stats["train"]["accuracy"].append(accuracy)
                        self.training_stats["train"]["precision"].append(precision)
                        self.training_stats["train"]["recall"].append(recall)
                        self.training_stats["train"]["f1"].append(f1)
                        self.training_stats["train"]["roc_auc"].append(uac)
                if trial != 0:
                    trial.report(loss.item(), epoch)
                    if trial.should_prune():
                        raise TrialPruned()
        pred_list = []
        if do_not_eval:
            return np.array([0]), np.array([0])
        else:
            self.eval()
            with torch.no_grad():
                for inputs, labels in loader:
                    inputs = inputs.to(self.device)
                    outputs = self(inputs)
                    pred_list.extend(torch.argmax(outputs, dim=1).cpu().numpy())
            pred = np.array(pred_list)
            y = y_tensor.cpu().numpy()
            return pred, y

    def eval_model(
        self,
        x_tensor: torch.Tensor,
        y_tensor: torch.Tensor,
    ) -> tuple[np.ndarray, np.ndarray]:
        print(f"{(time.time() - self.start_time):.5f}\t Testing SVM")
        pred = self(x_tensor.to(self.device)).squeeze().detach().cpu().numpy()
        pred_binary = (pred > 0).astype(int)
        y = y_tensor.cpu().numpy()
        return pred_binary, y

    def validate_model(
        self,
        criterion: nn.CrossEntropyLoss,
        x_tensor: torch.Tensor,
        y_tensor: torch.Tensor,
    ) -> tuple[np.ndarray, np.ndarray, float]:
        print(f"{(time.time() - self.start_time):.5f}\t Validating SVM")
        x_tensor = x_tensor.to(self.device)
        y_tensor = y_tensor.to(self.device)
        with torch.no_grad():
            outputs = self(x_tensor)
            loss = criterion(outputs.squeeze(), y_tensor.float())
            pred = outputs.squeeze().detach().cpu().numpy()
            pred_binary = (pred > 0).astype(int)
            y = y_tensor.cpu().numpy()
        return pred_binary, y, loss.item()

# Define the LSTM model
class LSTM(nn.Module, Own_Models):
    def __init__(
        self,
        input_size,
        hidden_size,
        num_layers,
        output_size,
        start_time: float,
        device: torch.device,
    ):
        nn.Module.__init__(self)
        Own_Models.__init__(self, start_time, device)
        # super(LSTM, self).__init__(start_time=start_time, device=device)
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        # Initialize the initial hidden state and cell state with zeros
        batch_size = x.size(0)
        h0 = torch.zeros(self.num_layers, batch_size, self.hidden_size)
        c0 = torch.zeros(self.num_layers, batch_size, self.hidden_size)
        out, _ = self.lstm(x, (h0, c0))
        out = self.fc(out[:, -1, :])
        return out

    def train_model(
        self,
        criterion: nn.CrossEntropyLoss,
        optimizer: torch.optim.Optimizer,
        y_tensor: torch.Tensor,
        x_tensor: torch.Tensor,
        scaler: torch.cuda.amp.GradScaler = None,
        accumulation_steps: int = 4,
        num_epochs: int = 10,
        amount_of_batches: int = AMOUNT_OF_BATCHES,
        batch_size: int = 0,
        num_workers: int = NUM_WORKERS,
        training_stats=False,
        do_not_eval: bool = False,
        trial: Trial = 0
    ) -> tuple[np.ndarray, np.ndarray]:

        print(f"{(time.time() - self.start_time):.5f}\t Training LSTM")
        dataset = TensorDataset(x_tensor, y_tensor)
        if batch_size != 0:
            loader = DataLoader(
                dataset,
                batch_size=batch_size,
                shuffle=True,
                generator=torch.Generator(device=self.device),
                num_workers=num_workers,
            )
        else:
            loader = DataLoader(
                dataset,
                batch_size=int(len(dataset) // amount_of_batches),
                shuffle=True,
                generator=torch.Generator(device=self.device),
                num_workers=num_workers,
            )
        # Train the model
        for epoch in range(num_epochs):
            self.train()
            running_loss = 0.0
            for i, data in enumerate(loader, 0):
                inputs, labels = data[0].to(self.device), data[1].to(self.device)
                optimizer.zero_grad()

                if scaler is not None:
                    with torch.cuda.amp.autocast():
                        outputs = self(inputs)
                        loss = criterion(outputs, labels)

                    scaler.scale(loss).backward()
                    if (i + 1) % accumulation_steps == 0:
                        scaler.step(optimizer)
                        scaler.update()
                        optimizer.zero_grad()
                else:
                    outputs = self(inputs)
                    loss = criterion(outputs, labels)
                    loss.backward()
                    optimizer.step()
                    optimizer.zero_grad()
                running_loss += loss.item()
                if (i + 1) % REPORT_PER_BATCH == 0:  # print every 2000 mini-batches
                    print(
                        f"{(time.time() - self.start_time):.5f}\t [{epoch + 1}, {i + 1:5d}/{len(loader)}] loss: {running_loss / REPORT_PER_BATCH:.3f}"
                    )
                    running_loss = 0.0
            if (epoch + 1) % REPORT_PER_EPOCH == 0:
                print(
                    f"{(time.time() - self.start_time):.5f}\t Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item():.5f}"
                )
                if training_stats:
                    pred_list = []
                    self.eval()
                    with torch.no_grad():
                        for inputs, labels in loader:
                            inputs = inputs.to(self.device)
                            outputs = self(inputs)
                            pred_list.extend(torch.argmax(outputs, dim=1).cpu().numpy())
                    pred = np.array(pred_list)
                    y = y_tensor.cpu().numpy()
                    accuracy, precision, recall, f1, uac, _ =  print_and_calc_metrics(y, pred, "train")
                    self.training_stats["train"]["epoch"].append(epoch+1)
                    self.training_stats["train"]["accuracy"].append(accuracy)
                    self.training_stats["train"]["precision"].append(precision)
                    self.training_stats["train"]["recall"].append(recall)
                    self.training_stats["train"]["f1"].append(f1)
                    self.training_stats["train"]["roc_auc"].append(uac)
            if trial != 0:
                trial.report(loss.item(), epoch)
                if trial.should_prune():
                    raise TrialPruned()
        # Collect predictions for training data
        if do_not_eval:
            return np.array([0]), np.array([0])
        pred_list = []
        self.eval()
        with torch.no_grad():
            for inputs, labels in loader:
                inputs = inputs.to(self.device)
                outputs = self(inputs)
                pred_list.extend(torch.argmax(outputs, dim=1).cpu().numpy())
        pred = np.array(pred_list)
        y = y_tensor.cpu().numpy()
        return pred, y

    def eval_model(
        self,
        x_tensor: torch.Tensor,
        y_tensor: torch.Tensor,
        amount_of_batches: int = AMOUNT_OF_BATCHES,
        num_workers: int = NUM_WORKERS,
        use_data_loader: bool = True,
    ) -> tuple[np.ndarray, np.ndarray]:
        print(f"{(time.time() - self.start_time):.5f}\t Testing LSTM")

        if use_data_loader:
            dataset = TensorDataset(x_tensor, y_tensor)
            loader = DataLoader(
                dataset,
                batch_size=int(len(dataset) // amount_of_batches),
                shuffle=False,
                generator=torch.Generator(device=self.device),
                num_workers=num_workers,
            )
            pred_list = []
            self.eval()
            with torch.no_grad():
                for inputs, labels in loader:
                    inputs, labels = inputs.to(self.device), labels.to(self.device)
                    outputs = self(inputs)
                    _, predicted = torch.max(outputs.data, 1)
                    pred_list.extend(predicted.cpu().numpy())

            pred = np.array(pred_list)
        else:
            self.eval()
            with torch.no_grad():
                outputs = self(x_tensor)
            predicted = torch.argmax(outputs, dim=1)
            pred = predicted.cpu().numpy()
        y = y_tensor.cpu().numpy()
        return pred, y

    def validate_model(
        self,
        criterion: nn.CrossEntropyLoss,
        x_tensor: torch.Tensor,
        y_tensor: torch.Tensor,
        num_workers: int = NUM_WORKERS,
        amount_of_batches: int = AMOUNT_OF_BATCHES,
    ) -> tuple[np.ndarray, np.ndarray, float]:
        print(f"{(time.time() - self.start_time):.5f}\t Evaluating LSTM")
        x_tensor = x_tensor.to(self.device)
        y_tensor = y_tensor.to(self.device)
        
        dataset = TensorDataset(x_tensor, y_tensor)
        loader = DataLoader(
            dataset,
            batch_size=int(len(dataset) // amount_of_batches),
            shuffle=True,
            generator=torch.Generator(device=self.device),
            num_workers=num_workers,
        )
        pred_list = []
        total_loss = 0
        self.eval()
        with torch.no_grad():
            for inputs, labels in loader:
                inputs, labels = inputs.to(self.device), labels.to(self.device)
                outputs = self(inputs)
                loss = criterion(outputs, labels)
                total_loss += loss.item()
                _, predicted = torch.max(outputs.data, 1)
                pred_list.extend(predicted.cpu().numpy())

        pred = np.array(pred_list)
        y = y_tensor.cpu().numpy()
        avg_loss = total_loss / len(loader)
        return pred, y, avg_loss


class MetaLearner(nn.Module, Own_Models):
    def __init__(
        self, input_dim, hidden_dim, output_dim, start_time: float, device: torch.device
    ):
        nn.Module.__init__(self)
        Own_Models.__init__(self, start_time, device)
        # super(MetaLearner, self).__init__(start_time, device)
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        out = self.fc1(x)
        out = self.relu(out)
        out = self.fc2(out)
        return out

    def train_model(
        self,
        criterion: nn.CrossEntropyLoss,
        optimizer: torch.optim.Optimizer,
        x_tensor: torch.Tensor = None,
        y_tensor: torch.Tensor = None,
        num_epochs: int = 10,
        training_stats: bool = False,
        do_not_eval: bool = False,
        trial: Trial = 0
    ) -> tuple[np.ndarray, np.ndarray]:
        print(f"{(time.time() - self.start_time):.5f}\t Training MetaLearner")
        self.train()
        x_tensor = x_tensor.to(self.device)
        y_tensor = y_tensor.to(self.device)
        for epoch in range(num_epochs):
            optimizer.zero_grad()
            outputs = self(x_tensor)
            loss = criterion(outputs, y_tensor)
            loss.backward()
            optimizer.step()

            if (epoch + 1) % REPORT_PER_EPOCH == 0:
                print(
                    f"{(time.time() - self.start_time):.5f}\t Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item():.7f}"
                )
                if training_stats:
                    predicted = torch.argmax(outputs, dim=1)
                    y = y_tensor.cpu().numpy()
                    pred = predicted.cpu().numpy()
                    accuracy, precision, recall, f1, uac, _ =  print_and_calc_metrics(y, pred, "train")
                    self.training_stats["train"]["epoch"].append(epoch+1)
                    self.training_stats["train"]["accuracy"].append(accuracy)
                    self.training_stats["train"]["precision"].append(precision)
                    self.training_stats["train"]["recall"].append(recall)
                    self.training_stats["train"]["f1"].append(f1)
                    self.training_stats["train"]["roc_auc"].append(uac)
            if trial != 0:
                trial.report(loss.item(), epoch)
                if trial.should_prune():
                    raise TrialPruned()
            
        if do_not_eval:
            return np.array([0]), np.array([0])
        predicted = torch.argmax(outputs, dim=1)
        pred = predicted.cpu().numpy()
        y = y_tensor.cpu().numpy()
        return pred, y

    def eval_model(
        self,
        x_tensor: torch.Tensor,
        y_tensor: torch.Tensor,
    ) -> tuple[np.ndarray, np.ndarray]:

        x_tensor = x_tensor.to(self.device)
        y_tensor = y_tensor.to(self.device)
        print(f"{(time.time() - self.start_time):.5f}\t Testing MetaLeaner")
        self.eval()
        with torch.no_grad():
            outputs = self(x_tensor)

        predicted = torch.argmax(outputs, dim=1)
        y = y_tensor.cpu().numpy()
        pred = predicted.cpu().numpy()
        return pred, y

    def validate_model(
        self,
        criterion: nn.CrossEntropyLoss,
        x_tensor: torch.Tensor,
        y_tensor: torch.Tensor,
    ) -> tuple[np.ndarray, np.ndarray, float]:

        x_tensor = x_tensor.to(self.device)
        y_tensor = y_tensor.to(self.device)
        print(f"{(time.time() - self.start_time):.5f}\t Evaluating MetaLeaner")
        self.eval()
        with torch.no_grad():
            outputs = self(x_tensor)
            # Calculate the loss
            loss = criterion(outputs, y_tensor)

        predicted = torch.argmax(outputs, dim=1)
        y = y_tensor.cpu().numpy()
        pred = predicted.cpu().numpy()
        return pred, y, loss.item()


# Custom Dataset class to handle sparse matrices
class SparseDataset(Dataset):
    def __init__(self, X, y):
        self.X = csr_matrix(X)
        self.y = y

    def __len__(self):
        return self.X.shape[0]

    def __getitem__(self, idx):
        X = torch.tensor(self.X[idx].toarray(), dtype=torch.float32).squeeze(0)
        y = torch.tensor(self.y[idx], dtype=torch.long)
        return X, y


# Ensure the ANN, SVM, and LSTM models are compatible with scikit-learn
class SklearnWrapperForTorch:
    def __init__(self, model_class, input_dim, **kwargs):
        self.model_class = model_class
        self.input_dim = input_dim
        self.kwargs = kwargs
        self.model = model_class(input_dim, **kwargs)
        self.is_fitted = False

    def fit(self, X, y):
        optimizer = optim.Adam(self.model.parameters(), lr=0.001)
        criterion = nn.CrossEntropyLoss()

        # Convert sparse matrix to dense
        if isinstance(X, csr_matrix):
            X = X.toarray()

        X_tensor = torch.tensor(X, dtype=torch.float32)
        y_tensor = torch.tensor(y, dtype=torch.long)

        if self.model.__class__.__name__ == "LSTM":
            X_tensor = X_tensor.unsqueeze(1)

        dataset = TensorDataset(X_tensor, y_tensor)
        loader = DataLoader(
            dataset,
            batch_size=1000000,
            generator=torch.Generator(
                device="cuda" if torch.cuda.is_available() else "cpu"
            ),
        )
        print(len(loader))
        print(loader.batch_size)

        print(f"Training {self.model.__class__.__name__}...")
        print(f"{len(loader)=} : {len(dataset)=}")
        for epoch in range(2):
            self.model.train()
            running_loss = 0.0
            for i, data in enumerate(loader, 0):
                inputs, labels = data[0], data[1]
                optimizer.zero_grad()

                outputs = self.model(inputs)

                if self.model.__class__.__name__ == "SVM":
                    loss = criterion(outputs.squeeze(), labels.float())
                else:
                    loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()
                running_loss += loss.item()
                if (i + 1) % REPORT_PER_BATCH == 0:  # print every 2000 mini-batches
                    print(
                        f"[{epoch + 1}, {i + 1:5d}/{len(loader)}] loss: {running_loss / REPORT_PER_BATCH:.3f}"
                    )
                    running_loss = 0.0
            print(f"Epoch [{epoch+1}/2], Loss: {loss.item():.5f}")

        self.is_fitted = True
        print(f"Training of {self.model.__class__.__name__} completed.")

    def predict(self, X):
        if not self.is_fitted:
            raise RuntimeError("You must train the model before predicting!")

        # Convert sparse matrix to dense
        if isinstance(X, csr_matrix):
            X = X.toarray()

        self.model.eval()
        X_tensor = torch.tensor(X, dtype=torch.float32)
        with torch.no_grad():
            outputs = self.model(X_tensor)
        return torch.argmax(outputs, dim=1).cpu().numpy()

    def get_params(self, deep=True):
        return {
            "model_class": self.model_class,
            "input_dim": self.input_dim,
            **self.kwargs,
        }

    def set_params(self, **parameters):
        for parameter, value in parameters.items():
            setattr(self, parameter, value)
        return self
