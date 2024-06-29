LABEL_COLUMN = "label"
BATCH_SIZE = 128
AMOUNT_OF_BATCHES = 20000
NUM_WORKERS = 1
CLEANED=True
TRAINING_STATS = True

REPORT_PER_EPOCH = 1
REPORT_PER_BATCH = 10000

# Training/Test set
USE_SAVED_DATA = True
SEPERATOR = ","
# CSV_FILE = r"C:\Users\alper\Desktop\dataset\conn.log.labeled.cleaned.csv"
# CSV_FILE = r"C:\Users\alper\Desktop\dataset\conn.log.labeled.cleaned.csv"
CSV_FILE = r"C:\Users\alper\Desktop\dataset\stock.csv"
# ENCODED_FILE = r"C:\Users\alper\Desktop\dataset\encoded_dataset.pkl"
ENCODED_FILE = r"C:\Users\alper\Desktop\dataset\encoded_dataset.pkl"

# Validation set
VALIDATE_SET = True
VALIDATE_SEPERATOR = ","
VALIDATE_USE_SAVED_DATA = True
# VALIDATE_CSV = (
#    r"C:\Users\rodyj\Documents\data\IoTScenarios\CTU-IoT-Malware-Capture-7-1\conn.log.labeled.cleaned.csv"
# )
VALIDATE_CSV = (
   r"B:\Datasets\IoTScenarios\CTU-IoT-Malware-Capture-7-1\conn.log.labeled.cleaned.csv"
)

# VALIDATE_ENCODED_FILE = r"C:\Users\rodyj\Documents\Module8\encoded_dataset2.pkl"
VALIDATE_ENCODED_FILE = r".\encoded_dataset2.pkl"

# Cleaning options
NAN_REPLACE_VAL = -1
STRING_REPLACE_VAL = "UNKNOWN"
