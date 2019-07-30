

#
# Interfaces that a Model must implement
#
class ModelInterface:

    # Terms for dataframe, etc.
    TERM_CLASS    = 'class'
    TERM_SCORE    = 'score'
    TERM_DIST     = 'dist'
    TERM_DISTNORM = 'distnorm'
    TERM_RADIUS   = 'radius'

    # Matching
    MATCH_TOP = 10

    def __init__(self):
        return

    def get_model_features(
            self
    ):
        return None

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

    def train(
            self
    ):
        return

    def load_model_parameters(
            self
    ):
        return

    def is_model_ready(
            self
    ):
        return True