

#
# Interfaces that a Model must implement
#
class ModelInterface:

    def __init__(self):
        return

    def predict_classes(
            self,
            # ndarray type of >= 2 dimensions
            x):
        return

    def predict_class(
            self,
            # ndarray type of >= 2 dimensions, single point/row array
            x
    ):
        return

    def train(self):
        return

    def load_model_parameters(self):
        return