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

    SENT_PER_DOC_80 = 12
    SENT_PER_DOC_90 = 17
    SENT_PER_DOC_95 = 22
    SENT_PER_DOC_100 = 151

    WORDS_PER_SENT_80 = 23
    WORDS_PER_SENT_90 = 30
    WORDS_PER_SENT_95 = 36
    WORDS_PER_SENT_100 = 846

    WORDS_PER_DOC_80 = 208
    WORDS_PER_DOC_90 = 294
    WORDS_PER_DOC_95 = 389
    WORDS_PER_DOC_100 = 1234


class Yahoo:
    TRAIN_DATASET = f"{DATASET_DIR}/yahoo_train.csv"
    VAL_DATASET = f"{DATASET_DIR}/yahoo_val.csv"
    TEST_DATASET = f"{DATASET_DIR}/yahoo_test.csv"
    EMBEDDING_FILE = f"{EMBEDDING_DIR}/yahoo.wv"

    SENT_PER_DOC_80 = 8
    SENT_PER_DOC_90 = 11
    SENT_PER_DOC_95 = 16
    SENT_PER_DOC_100 = 514

    WORDS_PER_SENT_80 = 28
    WORDS_PER_SENT_90 = 37
    WORDS_PER_SENT_95 = 48
    WORDS_PER_SENT_100 = 3977

    WORDS_PER_DOC_80 = 157
    WORDS_PER_DOC_90 = 234
    WORDS_PER_DOC_95 = 320
    WORDS_PER_DOC_100 = 4001


class Amazon:
    TRAIN_DATASET = f"{DATASET_DIR}/amazon_train.csv"
    VAL_DATASET = f"{DATASET_DIR}/amazon_val.csv"
    TEST_DATASET = f"{DATASET_DIR}/amazon_test.csv"
    EMBEDDING_FILE = f"{EMBEDDING_DIR}/amazon.wv"

    SENT_PER_DOC_80 = 10
    SENT_PER_DOC_90 = 16
    SENT_PER_DOC_95 = 23
    SENT_PER_DOC_100 = 660

    WORDS_PER_SENT_80 = 28
    WORDS_PER_SENT_90 = 35
    WORDS_PER_SENT_95 = 43
    WORDS_PER_SENT_100 = 1981

    WORDS_PER_DOC_80 = 201
    WORDS_PER_DOC_90 = 346
    WORDS_PER_DOC_95 = 506
    WORDS_PER_DOC_100 = 7485


class Synthetic:
    TRAIN_DATASET = f"{DATASET_DIR}/synthetic_train.csv"
    VAL_DATASET = f"{DATASET_DIR}/synthetic_val.csv"
    TEST_DATASET = f"{DATASET_DIR}/synthetic_test.csv"
    EMBEDDING_FILE = f"{EMBEDDING_DIR}/synthetic.wv"

    SENT_PER_DOC_80 = 9
    SENT_PER_DOC_90 = 10
    SENT_PER_DOC_95 = 12
    SENT_PER_DOC_100 = 17

    WORDS_PER_SENT_80 = 19
    WORDS_PER_SENT_90 = 22
    WORDS_PER_SENT_95 = 24
    WORDS_PER_SENT_100 = 29

    WORDS_PER_DOC_80 = 123
    WORDS_PER_DOC_90 = 144
    WORDS_PER_DOC_95 = 163
    WORDS_PER_DOC_100 = 270
