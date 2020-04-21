import torch


# Hyperparameters
EPOCHS = 2
BATCH_SIZE = 64
LEARNING_RATE = 0.01
MOMENTUM = 0.9
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


class Yelp:
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
    TRAIN_DATASET = f"{DATASET_DIR}/yahoo_train.csv"
    VAL_DATASET = f"{DATASET_DIR}/yahoo_val.csv"
    TRAINVAL_DATASET = f"{DATASET_DIR}/yahoo_trainval.csv"
    TEST_DATASET = f"{DATASET_DIR}/yahoo_test.csv"
    EMBEDDING_FILE = f"{EMBEDDING_DIR}/yahoo.wv"

    SENT_PER_DOC_80 = 12  # TODO update
    SENT_PER_DOC_90 = 17  # TODO update
    SENT_PER_DOC_95 = 22  # TODO update
    SENT_PER_DOC_100 = 151  # TODO update

    WORDS_PER_SENT_80 = 23  # TODO update
    WORDS_PER_SENT_90 = 30  # TODO update
    WORDS_PER_SENT_95 = 36  # TODO update
    WORDS_PER_SENT_100 = 846  # TODO update

    WORDS_PER_DOC_80 = 208  # TODO update
    WORDS_PER_DOC_90 = 294  # TODO update
    WORDS_PER_DOC_95 = 389  # TODO update
    WORDS_PER_DOC_100 = 1234  # TODO update


class Amazon:
    TRAIN_DATASET = f"{DATASET_DIR}/amazon_train.csv"
    VAL_DATASET = f"{DATASET_DIR}/amazon_val.csv"
    TRAINVAL_DATASET = f"{DATASET_DIR}/amazon_trainval.csv"
    TEST_DATASET = f"{DATASET_DIR}/amazon_test.csv"
    EMBEDDING_FILE = f"{EMBEDDING_DIR}/amazon.wv"

    SENT_PER_DOC_80 = 12  # TODO update
    SENT_PER_DOC_90 = 17  # TODO update
    SENT_PER_DOC_95 = 22  # TODO update
    SENT_PER_DOC_100 = 151  # TODO update

    WORDS_PER_SENT_80 = 23  # TODO update
    WORDS_PER_SENT_90 = 30  # TODO update
    WORDS_PER_SENT_95 = 36  # TODO update
    WORDS_PER_SENT_100 = 846  # TODO update

    WORDS_PER_DOC_80 = 208  # TODO update
    WORDS_PER_DOC_90 = 294  # TODO update
    WORDS_PER_DOC_95 = 389  # TODO update
    WORDS_PER_DOC_100 = 1234  # TODO update
