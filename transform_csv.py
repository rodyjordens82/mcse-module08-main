import os
import pandas as pd
from helper_funcs import (
    attempt_convert_string_float,
    attempt_convert_string_int,
    cast_t_f_to_bool,
    cast_ip_to_int,
)

""" transform CSVs """


def modify_file(
    file_path, new_file, x_value, x_replace, y_value, y_replace, z_value, z_replace
):
    # Read the contents of the file
    with open(file_path, "r") as file:
        file_contents = file.read()

    # Replace X with the provided value
    modified_contents = file_contents.replace(x_value, x_replace)

    # Remove Z from the contents
    modified_contents = modified_contents.replace(y_value, y_replace)
    modified_contents = modified_contents.replace(z_value, z_replace)
    # Remove lines starting with #
    modified_contents = "\n".join(
        line for line in modified_contents.split("\n") if not line.startswith("#")
    )
    # Write the modified contents back to the file
    with open(new_file, "w") as file:
        file.write(modified_contents)


def process_directory(
    directory,
    x_value,
    x_replace,
    y_value,
    y_replace,
    z_value,
    z_replace,
    clean_data=False,
):
    print(directory)
    # Get the list of files in the directory
    files = os.listdir(directory)

    # Iterate over the files
    for file in files:
        # Check if the file ends with .labeled
        if file.endswith(".labeled"):
            # Construct the file path
            file_path = os.path.join(directory, file)

            # Construct the new file path
            new_file = f"{file_path}.csv"

            # Call the modify_file function
            modify_file(
                file_path,
                new_file,
                x_value,
                x_replace,
                y_value,
                y_replace,
                z_value,
                z_replace,
            )
            if clean_data:
                dataset = pd.read_csv(new_file, sep="\t", low_memory=False)
                # Cleaning Data
                print("Cleaning Dataset")
                # dataset = dataset.drop(columns=["uid"])
                # dataset = dataset.fillna(NAN_REPLACE_VAL)
                dataset["ts"] = dataset["ts"].apply(attempt_convert_string_float)
                dataset["id.orig_h"] = dataset["id.orig_h"].apply(cast_ip_to_int)
                dataset["id.orig_p"] = dataset["id.orig_p"].apply(
                    attempt_convert_string_int
                )
                dataset["id.resp_h"] = dataset["id.resp_h"].apply(cast_ip_to_int)
                dataset["id.resp_p"] = dataset["id.resp_p"].apply(
                    attempt_convert_string_int
                )
                dataset["proto"] = dataset["proto"].apply(lambda x: str(x).lower())
                # dataset["service"] = dataset["service"].apply(lambda x: str(x).replace("-1", "unknown"))
                dataset["service"] = dataset["service"].apply(lambda x: str(x).lower())
                dataset["duration"] = dataset["duration"].apply(
                    attempt_convert_string_float
                )
                dataset["orig_bytes"] = dataset["orig_bytes"].apply(
                    attempt_convert_string_int
                )
                dataset["resp_bytes"] = dataset["resp_bytes"].apply(
                    attempt_convert_string_int
                )
                dataset["conn_state"] = dataset["conn_state"].apply(
                    lambda x: str(x).lower()
                )
                dataset["local_orig"] = dataset["local_orig"].apply(cast_t_f_to_bool)
                dataset["local_resp"] = dataset["local_resp"].apply(cast_t_f_to_bool)
                dataset["missed_bytes"] = dataset["missed_bytes"].apply(
                    attempt_convert_string_int
                )
                # dataset["history"] = dataset["history"].apply(lambda x: str(x).replace("-1", "unknown"))
                dataset["orig_pkts"] = dataset["orig_pkts"].apply(
                    attempt_convert_string_int
                )
                dataset["orig_ip_bytes"] = dataset["orig_ip_bytes"].apply(
                    attempt_convert_string_int
                )
                dataset["resp_pkts"] = dataset["resp_pkts"].apply(
                    attempt_convert_string_int
                )
                dataset["resp_ip_bytes"] = dataset["resp_ip_bytes"].apply(
                    attempt_convert_string_int
                )
                # dataset["tunnel_parents"] = dataset["tunnel_parents"].apply(lambda x: x.lower())
                dataset["label"] = dataset["label"].apply(lambda x: str(x).lower())

                dataset = dataset.astype(
                    {
                        "uid": "string",
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
                filename = file_path + ".cleaned.csv"
                print(f"Saving to {filename}")
                dataset.to_csv(filename, index=False)

        # Check if the file is a directory
        if os.path.isdir(os.path.join(directory, file)):
            # Recursively call process_directory for the subdirectory
            process_directory(
                os.path.join(directory, file),
                x_value,
                x_replace,
                y_value,
                y_replace,
                z_value,
                z_replace,
                clean_data,
            )


STATIC_X_VALUE = "   "
STATIC_X_REPLACE = "\t"
STATIC_Y_VALUE = "#fields\t"
STATIC_Y_REPLACE = ""
STATIC_Z_VALUE = ""
STATIC_Z_REPLACE = ""
DIRECTORY = r"C:\Users\rodyj\Documents\data\IoTScenarios"

process_directory(
    DIRECTORY,
    STATIC_X_VALUE,
    STATIC_X_REPLACE,
    STATIC_Y_VALUE,
    STATIC_Y_REPLACE,
    STATIC_Z_VALUE,
    STATIC_Z_REPLACE,
    True,
)
