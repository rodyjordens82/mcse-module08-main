"""Script to combine all the data from the datasets into a single CSV file."""

import os
import pandas as pd
from config import SEPERATOR

#BASE_DIRECTORY = r"B:\Datasets\IoTScenarios"
BASE_DIRECTORY = r"C:\Users\rodyj\Documents\data\IoTScenarios"
MAX_SIZE = 1e9  # 1 GB in bytes
#SEPERATOR = "\t"
ONLY_CLEANED = True

stock = None

for root, dirs, files in os.walk(BASE_DIRECTORY):
    for file in files:
        if file.endswith(".csv"):
            file_path = os.path.join(root, file)
            if ONLY_CLEANED and "cleaned" not in file_path:
                continue
            elif not ONLY_CLEANED and "cleaned" in file_path:
                continue
            file_size = os.path.getsize(file_path)

            if file_size < MAX_SIZE:
                print(f"Reading {file_path}")
                df = pd.read_csv(file_path, sep=SEPERATOR, comment="#", dtype=str, low_memory=False)
                if stock is None:
                    stock = df
                else: # Combine the data
                    stock = pd.concat([stock, df], ignore_index=True)
                    print(len(stock))
                #print(f"Columns in {file}: {df.columns}")

stock['label'] = stock['label'].apply(lambda x: x.lower())
stock.to_csv("stock.csv", index=False, sep=SEPERATOR)
print(stock['label'].value_counts())
print(stock['duration'].value_counts())
