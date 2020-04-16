# Hyperparameters
EPOCHS = 50
BATCH_SIZE = 64
LEARNING_RATE = 1
MOMENTUM = 0.9

# Padding (Yelp)
# MAX_SENT = 2
# MAX_WORDS = 46
MAX_SENT = 12  # 80%
MAX_WORDS = 23  # 80%
# MAX_SENT = 17  # 90%
# MAX_WORDS = 30  # 90%

# Padding (Yahoo)
# MAX_SENT = 8
# MAX_WORDS = 28

# Others
DEVICE = "cuda:0"
# DEVICE = "cpu"
TQDM = True
