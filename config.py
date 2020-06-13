import torch


# Hyperparameters
EPOCHS = 30
BATCH_SIZE = 64
LEARNING_RATE = 0.01
MOMENTUM = 0.9
PATIENCE = 3
EMBEDDING_SIZE = 200
BIDIRECTIONAL = 2  # set to 1 (not bidirectional) or 2 (bidirectional)
WORD_HIDDEN_SIZE = 50
SENT_HIDDEN_SIZE = 50
PADDING = 80  # percentage of the documents to cover with the padding

# Others
DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
TQDM = True  # Display a progress bar in training and test
DATASET_DIR = "data"
EMBEDDING_DIR = "embedding"
MODEL_DIR = "models"
LOG_DIR = "runs"


class Yelp:
    TRAIN_DATASET = f"{DATASET_DIR}/yelp_train.csv"
    VAL_DATASET = f"{DATASET_DIR}/yelp_val.csv"
    TEST_DATASET = f"{DATASET_DIR}/yelp_test.csv"
    EMBEDDING_FILE = f"{EMBEDDING_DIR}/yelp.wv"

    SENT_PER_DOC = {80: 12, 90: 17, 95: 22, 100: 151}
    WORDS_PER_SENT = {80: 26, 90: 34, 95: 43, 100: 996}
    WORDS_PER_DOC = {80: 230, 90: 325, 95: 426, 100: 1221}


class Yahoo:
    TRAIN_DATASET = f"{DATASET_DIR}/yahoo_train.csv"
    VAL_DATASET = f"{DATASET_DIR}/yahoo_val.csv"
    TEST_DATASET = f"{DATASET_DIR}/yahoo_test.csv"
    EMBEDDING_FILE = f"{EMBEDDING_DIR}/yahoo.wv"

    SENT_PER_DOC = {80: 8, 90: 11, 95: 16, 100: 514}
    WORDS_PER_SENT = {80: 28, 90: 37, 95: 48, 100: 3997}
    WORDS_PER_DOC = {80: 157, 90: 234, 95: 320, 100: 4001}


class Amazon:
    TRAIN_DATASET = f"{DATASET_DIR}/amazon_train.csv"
    VAL_DATASET = f"{DATASET_DIR}/amazon_val.csv"
    TEST_DATASET = f"{DATASET_DIR}/amazon_test.csv"
    EMBEDDING_FILE = f"{EMBEDDING_DIR}/amazon.wv"

    SENT_PER_DOC = {80: 10, 90: 16, 95: 23, 100: 660}
    WORDS_PER_SENT = {80: 28, 90: 35, 95: 43, 100: 1981}
    WORDS_PER_DOC = {80: 201, 90: 346, 95: 506, 100: 7485}


class Synthetic:
    TRAIN_DATASET = f"{DATASET_DIR}/synthetic_train.csv"
    VAL_DATASET = f"{DATASET_DIR}/synthetic_val.csv"
    TEST_DATASET = f"{DATASET_DIR}/synthetic_test.csv"
    EMBEDDING_FILE = f"{EMBEDDING_DIR}/synthetic.wv"

    SENT_PER_DOC = {80: 9, 90: 10, 95: 12, 100: 17}
    WORDS_PER_SENT = {80: 19, 90: 22, 95: 24, 100: 29}
    WORDS_PER_DOC = {80: 123, 90: 144, 95: 163, 100: 270}
