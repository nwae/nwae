import numpy as np
from nwae.utils.Log import Log
from inspect import getframeinfo, currentframe


"""
Перекрестная Энтропия
Это математичесое ожидание количества информационных битов, требующих представить
предсказание моделя, которая не отражает истинные вероятности.
Это измерение достигает минимум когда предсказание моделя равно истинной вероятности,
поэтому соотвествует как измерение потери.
"""
class CategoricalCrossEntropy:

    # This number is arbitrary
    SMALL_NUMBER = 10**(-12)

    def __init__(
            self,
            # Истинное распределение
            p_real_prob_labels,
            # Заданное распределение
            q_given_probs
    ):
        self.p_real_prob_labels = p_real_prob_labels
        self.q_given_probs = q_given_probs
        assert len(self.p_real_prob_labels) == len(self.q_given_probs)

        self.N = len(self.p_real_prob_labels)
        return

    def calculate(self):
        losses = []
        # The losses of each class has already been conveniently broken up by the categorical format
        for real_prob, given_probs in zip(self.p_real_prob_labels, self.q_given_probs):
            # Just to be sure in case numbers don't sum up to 1 for probabilities
            given_probs_normalized = given_probs / given_probs.sum(axis=-1, keepdims=True)
            assert abs(np.sum(given_probs_normalized) - 1.0) < CategoricalCrossEntropy.SMALL_NUMBER
            Log.debugdebug('Given Probs: ' + str(given_probs_normalized))

            # Calculate the number of bits required to represent this information
            info_bits = -np.log(np.maximum(CategoricalCrossEntropy.SMALL_NUMBER, given_probs_normalized))
            Log.debugdebug('Information Bits: ' + str(info_bits))

            # If the label is categorical, the loss is only the loss of the single non-zero category usually
            loss = np.sum(
                real_prob * info_bits,
                axis     = -1,
                keepdims = False
            )
            losses.append(loss)
        Log.debugdebug('Losses: ' + str(losses))
        # We can actually ignore the constant N term if we wish
        return np.sum(losses) * (1 / self.N)


if __name__ == '__main__':
    Log.LOGLEVEL = Log.LOG_LEVEL_DEBUG_2

    # Labels are the actual probabilities
    real_prob_labels = np.array([
        # We demo that when the output is exactly the same, the loss is zero for [1.0, 0.0, 0.0]
        [1, 0, 0], [1, 0, 0],
        [0, 1, 0],
        [0, 0, 1],
    ])
    # Usually softmax last layer output, thus should already sum up to 1.0
    given_probs = np.array([
        # We demo that when the output is exactly the same, the loss is zero for [1.0, 0.0, 0.0]
        [.90, .05, .05], [1.0, 0.0, 0.0],
        [.50, .89, .60],
        [.05, .01, .94],
    ])

    etp = CategoricalCrossEntropy(
        p_real_prob_labels = real_prob_labels,
        q_given_probs      = given_probs
    ).calculate()
    print(etp)
