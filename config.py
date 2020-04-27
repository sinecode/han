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
# DEVICE = "cpu"
TQDM = True  # Display a progress bar in training and test
DATASET_DIR = "data"
EMBEDDING_DIR = "embedding"
MODEL_DIR = "models"
LOG_DIR = "runs"


class Yelp:
    FULL_DATASET = f"{DATASET_DIR}/yelp.csv"
    TRAIN_DATASET = f"{DATASET_DIR}/yelp_train.csv"
    VAL_DATASET = f"{DATASET_DIR}/yelp_val.csv"
    TRAINVAL_DATASET = f"{DATASET_DIR}/yelp_trainval.csv"
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


class YelpSample:
    "10% of the original Yelp dataset"
    FULL_DATASET = f"{DATASET_DIR}/yelp_sample.csv"
    TRAIN_DATASET = f"{DATASET_DIR}/yelp_train_sample.csv"
    VAL_DATASET = f"{DATASET_DIR}/yelp_val_sample.csv"
    TEST_DATASET = f"{DATASET_DIR}/yelp_test.csv"
    EMBEDDING_FILE = Yelp.EMBEDDING_FILE

    SENT_PER_DOC_80 = Yelp.SENT_PER_DOC_80
    SENT_PER_DOC_90 = Yelp.SENT_PER_DOC_90
    SENT_PER_DOC_95 = Yelp.SENT_PER_DOC_95
    SENT_PER_DOC_100 = Yelp.SENT_PER_DOC_100

    WORDS_PER_SENT_80 = Yelp.WORDS_PER_SENT_80
    WORDS_PER_SENT_90 = Yelp.WORDS_PER_SENT_90
    WORDS_PER_SENT_95 = Yelp.WORDS_PER_SENT_95
    WORDS_PER_SENT_100 = Yelp.WORDS_PER_SENT_100

    WORDS_PER_DOC_80 = Yelp.WORDS_PER_DOC_80
    WORDS_PER_DOC_90 = Yelp.WORDS_PER_DOC_90
    WORDS_PER_DOC_95 = Yelp.WORDS_PER_DOC_95
    WORDS_PER_DOC_100 = Yelp.WORDS_PER_DOC_100


class Yahoo:
    FULL_DATASET = f"{DATASET_DIR}/yahoo.csv"
    TRAIN_DATASET = f"{DATASET_DIR}/yahoo_train.csv"
    VAL_DATASET = f"{DATASET_DIR}/yahoo_val.csv"
    TRAINVAL_DATASET = f"{DATASET_DIR}/yahoo_trainval.csv"
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
    FULL_DATASET = f"{DATASET_DIR}/amazon.csv"
    TRAIN_DATASET = f"{DATASET_DIR}/amazon_train.csv"
    VAL_DATASET = f"{DATASET_DIR}/amazon_val.csv"
    TRAINVAL_DATASET = f"{DATASET_DIR}/amazon_trainval.csv"
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
