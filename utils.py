import pickle


def save_obj_as_pickle(obj, filename):
    with open(filename, "wb") as f:
        pickle.dump(obj, f)


def load_pickled_obj(filename):
    with open(filename, "rb") as f:
        obj = pickle.load(f)
    return obj
