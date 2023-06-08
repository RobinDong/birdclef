import numpy as np


class EarlyStopper:
    def __init__(self, patience=1):
        self.patience = patience
        self.counter = 0
        self.min_validation_accuracy = -np.inf

    def early_stop(self, validation_accuracy):
        if validation_accuracy > self.min_validation_accuracy:
            self.min_validation_accuracy = validation_accuracy
            self.counter = 0
        elif validation_accuracy < self.min_validation_accuracy:
            self.counter += 1
            if self.counter >= self.patience:
                return True
        return False
