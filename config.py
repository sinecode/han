# Hyperparameters
EPOCHS = 1
BATCH_SIZE = 3
LEARNING_RATE = 1
MOMENTUM = 0.9

# Padding (Yelp)
SENT_PER_DOC = 12  # 80%
WORDS_PER_SENT = 23  # 80%
# SENT_PER_DOC = 17  # 90%
# WORDS_PER_SENT = 30  # 90%

# Padding (Yahoo)
# SENT_PER_DOC = 8
# WORDS_PER_SENT = 28

# Others
# DEVICE = "cuda:0"
DEVICE = "cpu"
TQDM = False
