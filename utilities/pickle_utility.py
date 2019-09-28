import cPickle as pickle


def read_pickle(file_path, type):
    with open(file_path, type) as f:
        obj = pickle.load(f)
        f.close()
    return obj