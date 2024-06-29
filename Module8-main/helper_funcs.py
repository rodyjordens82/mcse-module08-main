import time
import ipaddress
import joblib
import pandas as pd
import sklearn.metrics
from sklearn.preprocessing import LabelEncoder, OrdinalEncoder
from config import SEPERATOR, NAN_REPLACE_VAL


def attempt_convert_string_float(x: str) -> float:
    if type(x) == float:
        return x
    try:
        if x == "-":
            return int(NAN_REPLACE_VAL)
        else:
            return float(x)
    except ValueError:
        return int(NAN_REPLACE_VAL)


def attempt_convert_string_int(x: str) -> int:
    if type(x) == int:
        return x
    try:
        if x == "-":
            return int(NAN_REPLACE_VAL)
        else:
            return int(x)
    except ValueError:
        return int(NAN_REPLACE_VAL)


def cast_t_f_to_bool(x: str) -> int:
    if type(x) == int:
        return x
    if x == "T":
        return 1
    elif x == "F":
        return 0
    else:
        return NAN_REPLACE_VAL


def cast_ip_to_int(x: str) -> int:
    if type(x) == int:
        return x
    elif type(x) == float:
        return int(x)
    elif "." in x:
        return int(ipaddress.IPv4Address(x))
    elif ":" in x:
        return int(ipaddress.IPv6Address(x))
    else:
        print(f"Neither IPv4 nor IPv6 address found in {x}")
        return NAN_REPLACE_VAL


def load_and_encode(
    file_path: str,
    encoded_file_path: str,
    start_time: float,
    label_encoders: tuple[list[OrdinalEncoder], LabelEncoder] = None,
    sep=SEPERATOR,
    cleaned=True,
) -> tuple[pd.DataFrame, pd.DataFrame, OrdinalEncoder, LabelEncoder]:
    print(f"{(time.time() - start_time):.5f}\t Loading Dataset")

    # Load the dataset
    dataset = pd.read_csv(file_path, sep=sep, comment="#", low_memory=False)

    # Cleaning Data
    print(f"{(time.time() - start_time):.5f}\t Cleaning Dataset")
    dataset = dataset.drop(columns=["uid"]) if "uid" in dataset.columns else dataset
    # dataset = dataset.fillna(NAN_REPLACE_VAL)
    if not cleaned:
        dataset["ts"] = dataset["ts"].apply(attempt_convert_string_float)
        dataset["id.orig_h"] = dataset["id.orig_h"].apply(cast_ip_to_int)
        dataset["id.orig_p"] = dataset["id.orig_p"].apply(attempt_convert_string_int)
        dataset["id.resp_h"] = dataset["id.resp_h"].apply(cast_ip_to_int)
        dataset["id.resp_p"] = dataset["id.resp_p"].apply(attempt_convert_string_int)
        dataset["proto"] = dataset["proto"].apply(lambda x: str(x).lower())
        # dataset["service"] = dataset["service"].apply(lambda x: str(x).replace("-1", "unknown"))
        dataset["service"] = dataset["service"].apply(lambda x: str(x).lower())
        dataset["duration"] = dataset["duration"].apply(attempt_convert_string_float)
        dataset["orig_bytes"] = dataset["orig_bytes"].apply(attempt_convert_string_int)
        dataset["resp_bytes"] = dataset["resp_bytes"].apply(attempt_convert_string_int)
        dataset["conn_state"] = dataset["conn_state"].apply(lambda x: str(x).lower())
        dataset["local_orig"] = dataset["local_orig"].apply(cast_t_f_to_bool)
        dataset["local_resp"] = dataset["local_resp"].apply(cast_t_f_to_bool)
        dataset["missed_bytes"] = dataset["missed_bytes"].apply(attempt_convert_string_int)
        # dataset["history"] = dataset["history"].apply(lambda x: str(x).replace("-1", "unknown"))
        dataset["orig_pkts"] = dataset["orig_pkts"].apply(attempt_convert_string_int)
        dataset["orig_ip_bytes"] = dataset["orig_ip_bytes"].apply(
            attempt_convert_string_int
        )
        dataset["resp_pkts"] = dataset["resp_pkts"].apply(attempt_convert_string_int)
        dataset["resp_ip_bytes"] = dataset["resp_ip_bytes"].apply(
            attempt_convert_string_int
        )
        # dataset["tunnel_parents"] = dataset["tunnel_parents"].apply(lambda x: x.lower())
        dataset["label"] = dataset["label"].apply(lambda x: str(x).lower())

        dataset = dataset.astype(
            {
                "ts": "Float64",
                "id.orig_h": "float64",
                "id.orig_p": "Int64",
                "id.resp_h": "float64",
                "id.resp_p": "Int64",
                "proto": "string",
                "service": "string",
                "duration": "Float64",
                "orig_bytes": "Int64",
                "resp_bytes": "Int64",
                "conn_state": "string",
                "local_orig": "Int64",
                "local_resp": "Int64",
                "missed_bytes": "Int64",
                "orig_pkts": "Float64",
                "orig_ip_bytes": "Int64",
                "resp_pkts": "Int64",
                "resp_ip_bytes": "Int64",
                "label": "string",
            }
        )
    # Separate features and labels
    X = dataset.drop(columns=["label", "detailed-label"])
    y = dataset["label"]

    print(f"{(time.time() - start_time):.5f}\t Encoding Dataset")
    # Encode non-numeric columns
    non_numeric_columns = X.select_dtypes(include=["object", "string"]).columns
    if label_encoders is None:
        label_encoders = {}
        for col in non_numeric_columns:
            label_encoders[col] = OrdinalEncoder(
                unknown_value=NAN_REPLACE_VAL, handle_unknown="use_encoded_value"
            )
            X[[col]] = label_encoders[col].fit_transform(X[[col]])

        print(f"{(time.time() - start_time):.5f}\t Encoding labels")
        # Convert labels to numeric type
        label_encoder_y = LabelEncoder()
        y = label_encoder_y.fit_transform(y)
    else:
        label_encoders, label_encoder_y = label_encoders
        for col in non_numeric_columns:
            if col not in label_encoders:
                label_encoders[col] = OrdinalEncoder()
            X[[col]] = label_encoders[col].transform(X[[col]])
        y = label_encoder_y.transform(y)

    print(f"{(time.time() - start_time):.5f}\t Saving encoded dataset to {encoded_file_path}")
    # Save encoded dataset
    joblib.dump((X, y, label_encoders, label_encoder_y), encoded_file_path)
    return X, y, label_encoders, label_encoder_y


def calc_metrics(y_train, x_pred):
    try:
        x_pred = x_pred.cpu().numpy()
    except AttributeError:
        pass
        # print("Could not convert x_pred to cpu-numpy")
    accuracy = sklearn.metrics.accuracy_score(y_train, x_pred)
    precision = sklearn.metrics.precision_score(y_train, x_pred, average="macro")
    recall = sklearn.metrics.recall_score(y_train, x_pred, average="macro")
    f1 = sklearn.metrics.f1_score(y_train, x_pred, average="macro")
    if len(set(y_train)) > 1:
        uac = sklearn.metrics.roc_auc_score(y_train, x_pred, average="macro")
    else:
        uac = None
    matrix = sklearn.metrics.confusion_matrix(y_train, x_pred)
    return accuracy, precision, recall, f1, uac, matrix


def print_metrics(
    accuracy, precision, recall, f1, uac, matrix, text_add="", print_out=True
):
    a = f"Accuracy_{text_add}:\t{accuracy:.6f}\t"
    a += f"Precision_{text_add}: \t{precision:.6f}\n"
    a += f"Recall_{text_add}: \t{recall:.6f}\t"
    a += f"F1_{text_add}: \t\t{f1:.6f}\n"
    if uac is not None:
        a += f"AUC_{text_add}: \t{uac:.6f}\n"
    else:
        a += f"AUC_{text_add}: \tN/A\n"
    a += f"Matrix_{text_add}: \n{matrix}"
    if print_out:
        print(a)
    return a

def print_and_calc_metrics(y_train, x_pred, text_add=""):
    accuracy, precision, recall, f1, uac, matrix = calc_metrics(y_train, x_pred)
    print_metrics(accuracy, precision, recall, f1, uac, matrix, text_add)
    return accuracy, precision, recall, f1, uac, matrix