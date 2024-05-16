import pickle


def save_model(model, path: str):
    """Saves model as pickle file"""
    with open(path, 'wb') as file:
        pickle.dump(model, file)


def load_model(path: str):
    """Loads model from pickle file"""
    with open(path, 'rb') as file:
        return pickle.load(file)
