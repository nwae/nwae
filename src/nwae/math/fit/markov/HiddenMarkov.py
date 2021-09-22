# --*-- coding: utf-8 --*--

import numpy as np
from nwae.utils.Log import Log
from inspect import getframeinfo, currentframe
from nwae.utils.UnitTest import ResultObj, UnitTest


"""
Модель скрытых состояний

Конечное количество k состояний
{s0, s1, s2, …, sk-1}
Конечное число l наблюдаемых
{o0, o1, o2, …, ol-1}

Скрытый слой (обозначение “y” в нейронной сети)
h = [h0, h1, ... hn-1]
Наблюдаемый (обозначение “x” в нейронной сети)
y = [y0, y1, ... yn-1]

Длина n как переменными

Параметры для оптимизации
Версия оригинальная
Начальные вероятности: p_st_s0, p_st_s1, … p_st_sk-1
Вероятность перехода состояния
p_tr_s0_s0, p_tr_s1_s0, p_tr_s2_s0, …
p_tr_s0_s1, p_tr_s1_s1, p_tr_s2_s1, …
p_tr_s0_s2, p_tr_s1_s2, p_tr_s2_s2, …
…
p_tr_s0_xn, p_tr_s1_sn, p_tr_s2_sn, …
Вероятность выпуска наблюдаемого
p_em_o0_s0, p_em_o1_s0, p_em_o2_s0, …
p_em_o0_s1, p_em_o1_s1, p_em_o2_s1, …
p_em_o0_s2, p_em_o1_s2, p_em_o2_s2, …
…
p_em_o0_sn, p_em_o1_sn, p_em_o2_sn, …

Функция потерь
L(h,o) = СУММА ВСЕХ ПРЕДЛОЖЕНИЙ[
              f(p_st_h0 ) + f(p_em_o0_h0)
              + f(p_tr_h1_h0)) + f(p_em_o1_h1))
              + f(p_tr_h2_h1)) + f(p_em_o2_h2))
              + …
              + {до длины предложения}
         ]
и f(z) = -log(z) чтобы уменьшается с повышением z
Нужно в каждом шаге обновить параметры

p - delta * dL/p

где delta является скоростью обучения
"""
class HiddenMarkov:

    def __init__(
            self,
            unique_states,
            unique_observables,
    ):
        assert type(unique_states) in [list, tuple]
        assert type(unique_observables) in [list, tuple]

        # Обязательно такие числа 0, 1, 2, ...
        assert sorted(unique_states) == list(range(len(unique_states)))
        assert sorted(unique_observables) == list(range(len(unique_observables)))

        self.unique_states = unique_states
        # add one extra state for start
        self.state_none = max(self.unique_states)+1
        self.unique_states = [self.state_none] + self.unique_states
        self.n_h = len(self.unique_states)
        Log.info(
            str(self.__class__) + ' ' + str(getframeinfo(currentframe()).lineno)
            + ': Unique states: ' + str(self.unique_states)
        )

        # add one extra observable for start
        self.unique_observables = unique_observables
        self.observable_none = max(self.unique_observables)+1
        self.unique_observables = [self.observable_none] + self.unique_observables
        self.n_o = len(self.unique_observables)
        Log.info(
            str(self.__class__) + ' ' + str(getframeinfo(currentframe()).lineno)
            + ': Unique observables: ' + str(self.unique_observables)
        )
        return

    """
    Loss as maximum likelihood
    """
    def __loss(
            self,
            # скрытые значения
            h,
            # наблюдаемые
            o,
            h_trns_prob_matrix,
            o_emis_prob_matrix,
            # для отладки
            info_i = -1,
            info_j = -1,
    ):
        assert len(h) == len(o)
        assert len(h) > 0

        ml = 0
        for i in range(1, len(h), 1):
            ml_part = - np.log(h_trns_prob_matrix[h[i-1], h[i]]) - np.log(o_emis_prob_matrix[h[i], o[i]])
            if (h[i], h[i-1]) == (info_j,info_i):
                Log.debugdebug(
                    str(self.__class__) + ' ' + str(getframeinfo(currentframe()).lineno)
                    + '; p(' + str(info_i) + ',' + str(info_j) + ') = ' + str(h_trns_prob_matrix[h[i-1], h[i]])
                )
            ml += ml_part

        return ml

    def gradient_descent_state_transition_matrix(
            self,
            h_transform,
            o_transform,
            h_trns_prob_matrix,
            o_emis_prob_matrix,
            which_matrix,
            x_delta,
            learn_rate,
    ):
        matrix_move = h_trns_prob_matrix
        if which_matrix == 'emission':
            matrix_move = o_emis_prob_matrix
        for i in range(matrix_move.shape[0]):
            for j in range(matrix_move.shape[1]):
                matrix_move_copy = matrix_move.copy()
                matrix_move_copy[i, j] += x_delta

                if which_matrix == 'state transition':
                    loss_param_plus = self.__loss(
                        h = h_transform,
                        o = o_transform,
                        h_trns_prob_matrix = matrix_move_copy,
                        o_emis_prob_matrix = o_emis_prob_matrix,
                    )
                    loss_param = self.__loss(
                        h = h_transform,
                        o = o_transform,
                        h_trns_prob_matrix = matrix_move,
                        o_emis_prob_matrix = o_emis_prob_matrix,
                    )
                    dloss_param = loss_param_plus - loss_param
                else:
                    loss_param_plus = self.__loss(
                        h = h_transform,
                        o = o_transform,
                        h_trns_prob_matrix = h_trns_prob_matrix,
                        o_emis_prob_matrix = matrix_move_copy,
                    )
                    loss_param = self.__loss(
                        h = h_transform,
                        o = o_transform,
                        h_trns_prob_matrix = h_trns_prob_matrix,
                        o_emis_prob_matrix = matrix_move,
                    )
                    dloss_param = loss_param_plus - loss_param

                dloss_param = dloss_param / x_delta
                old_value = matrix_move[i, j]
                # sign_dloss_dparam = 1 * (dloss_param > 0)
                # if sign_dloss_dparam == 0:
                #     sign_dloss_dparam = -1
                movement = - dloss_param * matrix_move[i, j] * learn_rate
                # movement = sign_dloss_dparam * learn_rate
                if movement < 0:
                    movement = - min(matrix_move[i, j] * 0.9, -movement)
                else:
                    movement = min(1.0 - matrix_move[i, j], movement)
                matrix_move[i, j] += movement
                matrix_move[i] = matrix_move[i] / np.sum(matrix_move[i])
                assert (matrix_move[i, j] >= 0) & (matrix_move[i, j] <= 1)
        return matrix_move

    def fit(
            self,
            # Hidden states
            h,
            # Observables
            o,
            iterations = 100,
            x_delta = 0.000001,
            learn_rate = 0.01,
            loss_limit = 0.001,
    ):
        assert type(h) in [list, tuple]
        assert type(o) in [list, tuple]
        # потянуть в одную строчку
        h_transform = []
        for s_h in h:
            h_transform.append(self.state_none)
            h_transform += s_h
        o_transform = []
        for s_o in o:
            o_transform.append(self.observable_none)
            o_transform += s_o

        # Probability of current state, given current value, previous value, & previous state
        # P(x[t] | y[t] & x[t-1]) = P(x[t] and x[t-1] and y[t]) / P()
        # Начинать с равномерным распределением, с одиконокой вероятностью
        h_trns_prob_matrix = np.ones(shape=(self.n_h, self.n_h)) / self.n_h
        o_emis_prob_matrix = np.ones(shape=(self.n_h, self.n_o)) / self.n_o

        loss_iter = self.__loss(
            h = h_transform,
            o = o_transform,
            h_trns_prob_matrix = h_trns_prob_matrix,
            o_emis_prob_matrix = o_emis_prob_matrix,
        )
        for it in range(iterations):
            # new parameters
            h_trns_prob_matrix = self.gradient_descent_state_transition_matrix(
                h_transform = h_transform,
                o_transform = o_transform,
                h_trns_prob_matrix = h_trns_prob_matrix,
                o_emis_prob_matrix = o_emis_prob_matrix,
                which_matrix = 'state transition',
                x_delta = x_delta,
                learn_rate = learn_rate,
            )
            o_emis_prob_matrix = self.gradient_descent_state_transition_matrix(
                h_transform = h_transform,
                o_transform = o_transform,
                h_trns_prob_matrix = h_trns_prob_matrix,
                o_emis_prob_matrix = o_emis_prob_matrix,
                which_matrix = 'emission',
                x_delta = x_delta,
                learn_rate = learn_rate,
            )
            loss_new = self.__loss(
                h = h_transform,
                o = o_transform,
                h_trns_prob_matrix = h_trns_prob_matrix,
                o_emis_prob_matrix = o_emis_prob_matrix,
            )
            diff_loss = loss_iter - loss_new
            if diff_loss <= loss_limit:
                Log.debugdebug(
                    str(self.__class__) + ' ' + str(getframeinfo(currentframe()).lineno)
                    + ': Finished at iter # ' + str(it) + ' Loss diff = ' + str(diff_loss)
                )
                break
            Log.debugdebug(
                str(self.__class__) + ' ' + str(getframeinfo(currentframe()).lineno)
                + ': Iter # ' + str(it) + ' Loss diff = ' + str(diff_loss)
                + ', from ' + str(loss_iter) + ' to ' + str(loss_new)
                # + '. State transition matrix: ' + str(h_trns_prob_matrix)
            )
            loss_iter = loss_new

        for name, mtx in [('transition matrix', h_trns_prob_matrix), ('emission matrix', o_emis_prob_matrix)]:
            check_sum_to_1 = np.sum(mtx, axis=1)
            for i in range(len(check_sum_to_1)):
                value = check_sum_to_1[i]
                UnitTest.assert_true(
                    observed = round(value, 5),
                    expected = 1.00000,
                    test_comment = 'Check ' + str(name) + ' probability sum to one for line ' + str(i) + ', value ' + str(value),
                )
        Log.info(
            str(self.__class__) + ' ' + str(getframeinfo(currentframe()).lineno)
            + ': Final state transition matrix: ' + str(h_trns_prob_matrix)
        )
        Log.info(
            str(self.__class__) + ' ' + str(getframeinfo(currentframe()).lineno)
            + ': Final emission matrix: ' + str(o_emis_prob_matrix)
        )

        return h_trns_prob_matrix, o_emis_prob_matrix

    def predict(
            self,
            o,
            h_trns_prob_matrix,
            o_emis_prob_matrix,
    ):
        # TODO Выбирать из несколько вариантов, а не только одного
        h_best = None
        h_candidate = [self.state_none]
        likelihood = 0
        for word in o:
            h_prev = h_candidate[-1]
            likelihood_vec = []
            for h_tmp in range(len(self.unique_states)):
                if h_tmp == self.state_none:
                    continue
                val_tmp = - np.log(h_trns_prob_matrix[h_prev, h_tmp]) - np.log(o_emis_prob_matrix[h_tmp, word])
                likelihood_vec.append(val_tmp)
            # Наименьшее значение
            likelihood_vec_sorted_desc = np.argsort(likelihood_vec)
            h_optimal = likelihood_vec_sorted_desc[0]
            h_candidate.append(h_optimal)
            likelihood += likelihood_vec[h_optimal]
        return h_candidate


class HiddenMarkovUnitTest:
    def __init__(self, ut_params = None):
        self.ut_params = ut_params

    def run_unit_test(self):
        res = ResultObj()

        sample_data = [
            # [('I', 'PRP'), ('run', 'VBP'), ('the', 'DT'), ('code', 'NN')]
            ['i run the code', 'noun verb determiner noun'],
            # [('The', 'DT'), ('code', 'NN'), ('run', 'NN'), ('is', 'VBZ'), ('successful', 'JJ')]
            ['the code run is successful', 'determiner noun noun verb adjective'],
            # [('run', 'VBP')]
            ['run', 'verb'],
        ]
        # индексы 0, 1, 2,... обязательно по порядке для совпадения позиций в матрицах
        map_h = {'noun': 0, 'verb': 1, 'determiner': 2, 'adjective': 3} # Another None state will be added as #4
        map_o = {'i': 0, 'run': 1, 'the': 2, 'code': 3, 'is': 4, 'successful': 5} # Another None output will be added as #6

        h = [[map_h[word] for word in t[1].split(' ')] for t in sample_data]
        o = [[map_o[word] for word in t[0].split(' ')] for t in sample_data]

        res.update_bool(UnitTest.assert_true(
            observed = h,
            expected = [[0, 1, 2, 0], [2, 0, 0, 1, 3], [1]],
            test_comment = 'Check hidden states input'
        ))
        res.update_bool(UnitTest.assert_true(
            observed = o,
            expected = [[0, 1, 2, 3], [2, 3, 1, 4, 5], [1]],
            test_comment = 'Check observables input'
        ))

        hmm = HiddenMarkov(
            unique_states = list(map_h.values()),
            unique_observables = list(map_o.values()),
        )
        h_trns_prob_matrix, o_emis_prob_matrix = hmm.fit(
            h = h,
            o = o,
        )

        map_h[None] = len(map_h)
        map_o[None] = len(map_o)
        reverse_map_h = { v:k for k,v in map_h.items() }
        reverse_map_o = { v:k for k,v in map_o.items() }

        expected_h_trans = {
            # maximum probability of transition by order
            'noun':       ['verb', None, 'noun'],
            'verb':       ['adjective', 'determiner', None],
            'determiner': ['noun'],
            'adjective':  [None],
            None:         ['determiner', 'verb', 'noun']
        }
        for k in expected_h_trans.keys():
            i = map_h[k]
            high_orders_expected = [map_h[x] for x in expected_h_trans[k]]
            observed_line = h_trns_prob_matrix[i]
            observed_order = np.flip(np.argsort(observed_line))
            for j in range(len(high_orders_expected)):
                UnitTest.assert_true(
                    observed = reverse_map_h[observed_order[j]],
                    expected = reverse_map_h[high_orders_expected[j]],
                    test_comment = 'Test "' + str(k) + '" with highest probabilities of transition to '
                                   + str(expected_h_trans[k]) + ' for ' + str(high_orders_expected)
                )
        expected_emission = {
            # maximum probability of emission by order
            'noun':       ['code', 'run', 'i'],
            'verb':       ['run'],
            'determiner': ['the'],
            'adjective':  ['successful'],
            None:         [None]
        }
        for k in expected_emission.keys():
            i = map_h[k]
            high_orders_expected = [map_o[x] for x in expected_emission[k]]
            observed_line = o_emis_prob_matrix[i]
            observed_order = np.flip(np.argsort(observed_line))
            for j in range(len(high_orders_expected)):
                UnitTest.assert_true(
                    observed = reverse_map_o[observed_order[j]],
                    expected = reverse_map_o[high_orders_expected[j]],
                    test_comment = 'Test "' + str(k) + '" with highest probabilities of emission to '
                                   + str(expected_emission[k]) + ' for ' + str(high_orders_expected)
                )

        # Try to predict states
        for o_line in o:
            Log.debug('Predict ' + str([reverse_map_o[w] for w in o_line]))
            h_pred = hmm.predict(
                o = o_line,
                h_trns_prob_matrix = h_trns_prob_matrix,
                o_emis_prob_matrix = o_emis_prob_matrix,
            )
            Log.debug([reverse_map_h[w] for w in h_pred])
        return res


if __name__ == '__main__':
    Log.LOGLEVEL = Log.LOG_LEVEL_DEBUG_1
    Log.DEBUG_PRINT_ALL_TO_SCREEN = True
    res = HiddenMarkovUnitTest().run_unit_test()
    exit(res.count_fail)
