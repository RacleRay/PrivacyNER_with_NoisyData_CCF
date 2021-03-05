import numpy as np
from utils.tools import console


class EarlyStopping:
    """Early stops the training if validation loss doesn't improve after a given patience."""
    def __init__(self,
                 model,
                 path,
                 mode='min',
                 compare=None,
                 patience=7,
                 verbose=True,
                 delta=0):
        """
        Args:
            patience (int): How long to wait after last time validation loss improved.
                            Default: 7
            verbose (bool): If True, prints a message for each validation loss improvement.
                            Default: False
            delta (float): Minimum change in the monitored quantity to qualify as an improvement.
                            Default: 0
        """
        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        self.best_score = 0.0
        self.early_stop = False
        self.val_loss_min = np.Inf
        self.delta = delta

        self.model = model
        self.path = path
        self.mode = mode
        if compare is None:
            if mode == 'min':
                self.compare = lambda a, b: a < b
            elif mode == 'max':
                self.compare = lambda a, b: a > b
            else:
                assert 0
        else:
            self.compare = compare

    def __call__(self, score):
        score = float(score)
        if self.best_score == 0 or self.compare(score,
                                                self.best_score + self.delta):
            if self.verbose:
                console.log("Earlystop metric: {:.6} improved to {:.6}".format(
                            self.best_score, score))
            self.best_score = score
            self.counter = 0
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True
        return self.early_stop, self.best_score