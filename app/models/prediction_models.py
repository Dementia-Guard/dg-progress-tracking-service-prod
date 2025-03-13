import numpy as np


class DementiaProgressionModel:
    """
    Wrapper class for the dementia progression LSTM models
    """

    def __init__(self, mmse_model, cdr_model):
        self.mmse_model = mmse_model
        self.cdr_model = cdr_model

    def predict_mmse_change(self, sequence):
        """
        Predict MMSE score change

        Args:
            sequence: numpy array of shape (sequence_length, 3)
                     with MMSE, CDR, and Age values

        Returns:
            float: Predicted change in MMSE score
        """
        return self.mmse_model.predict(np.expand_dims(sequence, axis=0), verbose=0)[0][0]

    def predict_cdr_change(self, sequence):
        """
        Predict CDR score change

        Args:
            sequence: numpy array of shape (sequence_length, 3)
                     with MMSE, CDR, and Age values

        Returns:
            float: Predicted change in CDR score
        """
        return self.cdr_model.predict(np.expand_dims(sequence, axis=0), verbose=0)[0][0]

    def predict_next_values(self, sequence):
        """
        Predict next MMSE and CDR values

        Args:
            sequence: numpy array of shape (sequence_length, 3)
                     with MMSE, CDR, and Age values

        Returns:
            tuple: (next_mmse, next_cdr) predicted values
        """
        mmse_change = self.predict_mmse_change(sequence)
        cdr_change = self.predict_cdr_change(sequence)

        current_mmse = sequence[-1, 0]
        current_cdr = sequence[-1, 1]

        next_mmse = max(0, min(30, current_mmse + mmse_change))
        next_cdr = max(0, min(3, current_cdr + cdr_change))

        return next_mmse, next_cdr
