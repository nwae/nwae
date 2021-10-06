# -*- coding: utf-8 -*-

from nwae.utils.Log import Log
from inspect import getframeinfo, currentframe
import numpy as np
import pandas as pd
from nwae.utils.data.DataPreprcscr import DataPreprocessor
from nwae.utils.UnitTest import UnitTest, ResultObj
from nwae.math.suggest.SuggestDataProfile import SuggestDataProfile
from nwae.utils.Profiling import Profiling, ProfilingHelper


class SuggestMetric:

    # Быстро как в нейронных сетях
    METRIC_COSINE = 'cosine'
    # Медленно
    METRIC_EUCLIDEAN = 'euclidean'

    BIG_NUMBER_NON_EXISTENT_PRD_INDEX = 2**31

    def __init__(
            self,
    ):
        self.profiler_normalize_euclidean = ProfilingHelper(profiler_name = 'normalize euclidean')
        self.profiler_recommend = ProfilingHelper(profiler_name = 'recommend')

        pd.set_option('display.max_rows', 500)
        pd.set_option('display.max_columns', 500)
        pd.set_option('display.width', 1000)
        return

    def extract_attributes_list(
            self,
            df,
            unique_name_colums_list,
    ):
        cols_all = list(df.columns)
        attributes_list = cols_all.copy()
        for col in unique_name_colums_list:
            attributes_list.remove(col)
        return attributes_list

     #
    # TODO
    #   Два подходы отсюда
    #     1. Мы рассчитать или выводить отображения клиент --> (п1, п2, ...)
    #        который является форматом для методов МО.
    #        В этом случае нет такого "ДНК", а только параметры нейронных сетей,
    #        "xg boosting", и тд
    #     2. Мы сразу выводить "ДНК" продуктов через простую статистику,
    #        и алгоритм персонализации не будет AI, а класическая математика
    #
    def encode_product_attributes(
            self,
            df_human_profile,
            # Столцы которые определяют уникальных клиентов
            unique_human_key_columns,
            df_object,
            unique_df_object_human_key_columns,
            unique_df_object_object_key_columns,
            unique_df_object_value_column,
            unique_df_object_human_attribute_columns,
            apply_object_value_as_weight,
            # 'none', 'unit' (единичный вектор) or 'prob' (сумма атрибутов = 1)
            normalize_method,
    ):
        colkeep = unique_df_object_human_key_columns \
                  + unique_df_object_object_key_columns \
                  + [unique_df_object_value_column]
        df_object = df_object[colkeep]
        # Merge
        df_object_human_attributes = df_object.merge(
            df_human_profile,
            left_on  = unique_df_object_human_key_columns,
            right_on = unique_human_key_columns,
            # TODO
            #    Во время разработки мы упрощаем задачу с "inner" чтобы не столкнемся с значениями NaN
            #    Но в настоящей запуске программы должна быть "left" и нам следует обрабатывать те NaN значения
            how      = 'inner',
        )
        # Очистить числа
        df_object_human_attributes[unique_df_object_value_column] = \
            df_object_human_attributes[unique_df_object_value_column].apply(DataPreprocessor.filter_number)
        Log.info(
            str(self.__class__) + ' ' + str(getframeinfo(currentframe()).lineno)
            + ': Object human attributes (first 20 lines): ' + str(df_object_human_attributes[0:20])
        )
        # df_object_human_attributes.to_csv('object_human.csv')

        # Больше не нужны клиенты
        colkeep = unique_df_object_object_key_columns \
                  + [unique_df_object_value_column] \
                  + unique_df_object_human_attribute_columns
        df_object_attributes = df_object_human_attributes[colkeep]

        return self.__encode_product(
            df_object_attributes                = df_object_attributes,
            unique_df_object_object_key_columns = unique_df_object_object_key_columns,
            unique_df_object_value_column       = unique_df_object_value_column,
            unique_attribute_columns            = unique_df_object_human_attribute_columns,
            apply_object_value_as_weight        = apply_object_value_as_weight,
            normalize_method                    = normalize_method,
        )

    """
    С таких данных
            client        product  quantity  bonaqua  borjomi  illy  karspatskaya  lavazza
                  a       borjomi       1.0      0.0      1.0   0.0           1.0      0.0
                  a  karspatskaya       1.0      0.0      1.0   0.0           1.0      0.0
                  b       borjomi       2.0      0.0      2.0   0.0           1.0      0.0
                  b  karspatskaya       1.0      0.0      2.0   0.0           1.0      0.0
                  c          illy       1.0      1.0      0.0   1.0           0.0      0.0
                  c       bonaqua       1.0      1.0      0.0   1.0           0.0      0.0
                  d          illy       2.0      0.0      0.0   2.0           0.0      1.0
                  d       lavazza       1.0      0.0      0.0   2.0           0.0      1.0
                  e       bonaqua       2.0      2.0      0.0   0.0           0.0      1.0
                  e       lavazza       1.0      2.0      0.0   0.0           0.0      1.0
                  f       lavazza       2.0      0.0      0.0   0.0           0.0      2.0
                 n1       borjomi       1.0      0.0      1.0   0.0           0.0      0.0
                 n2          illy       1.0      0.0      0.0   1.0           0.0      0.0
                 n3       bonaqua       1.0      1.0      0.0   0.0           0.0      0.0
    в такие
       в случае "normalize_method=prob" (сумма каждой строки равно 1)
                    product   bonaqua   borjomi      illy  karspatskaya   lavazza
            0       bonaqua  0.666667  0.000000  0.166667      0.000000  0.166667
            1       borjomi  0.000000  0.666667  0.000000      0.333333  0.000000
            2          illy  0.166667  0.000000  0.666667      0.000000  0.166667
            3  karspatskaya  0.000000  0.600000  0.000000      0.400000  0.000000
            4       lavazza  0.250000  0.000000  0.250000      0.000000  0.500000
    то есть продукты спрофированы с атрибутами как самими продуктами
    ** Байесовская вероятность
    Математически по методу "transform_method=prob", эквивалентно Байесовской вероятности, то есть
    если значение в строке i и столбце j = v(i,j), то P(купит продукт j | куплен продукт i) = v(i,j)
    """
    def __encode_product(
            self,
            # Датафрейм который уже соединенный с аттрибутами человека
            df_object_attributes,
            unique_df_object_object_key_columns,
            unique_df_object_value_column,
            # Атрибуты из человека или сами продукты
            unique_attribute_columns,
            apply_object_value_as_weight,
            # 'none', 'unit' (единичный вектор) or 'prob' (сумма атрибутов = 1)
            normalize_method,
    ):
        assert len(unique_df_object_object_key_columns) == 1, 'Multi-column product names not supported yet'
        # TODO
        #    Сейчас только самый простой метод вычислить "аттрибуты" объектов
        Log.important(
            str(self.__class__) + ' ' + str(getframeinfo(currentframe()).lineno)
            + ': Start encoding product.. '
        )
        colkeep = unique_df_object_object_key_columns + [unique_df_object_value_column]
        df_agg_value = df_object_attributes[colkeep].groupby(
            by       = unique_df_object_object_key_columns,
            as_index = False,
        ).sum()
        df_agg_value.columns = unique_df_object_object_key_columns + ['__total_value']
        df_object_attributes = df_object_attributes.merge(
            df_agg_value,
            on  = unique_df_object_object_key_columns,
            how = 'left'
        )
        Log.info(
            str(self.__class__) + ' ' + str(getframeinfo(currentframe()).lineno)
            + ': Object attributes (first 20 lines): ' + str(df_object_attributes[0:20])
        )
        # Взвешенные аттрибуты
        if apply_object_value_as_weight:
            for col in unique_attribute_columns:
                df_object_attributes[col] = df_object_attributes[col] * \
                                            df_object_attributes[unique_df_object_value_column] \
                                            / df_object_attributes['__total_value']
        # df_object_attributes.to_csv('object_attr.csv')
        df_object_attributes_summarized = df_object_attributes.groupby(
            by       = unique_df_object_object_key_columns,
            as_index = False,
        ).sum()
        colkeep = unique_df_object_object_key_columns + unique_attribute_columns
        df_object_attributes_summarized = df_object_attributes_summarized[colkeep]
        # df_object_attributes_summarized.to_csv('object_attr_summary.csv')

        original_cols = list(df_object_attributes_summarized.columns)
        name_col = original_cols[0]
        attr_cols = original_cols.copy()
        attr_cols.remove(name_col)

        df_object_attributes_summarized_normalized = SuggestDataProfile.normalize(
            df                = df_object_attributes_summarized,
            name_columns      = [name_col],
            attribute_columns = attr_cols,
            normalize_method  = normalize_method,
        )

        set_non_zero_products = set(
            df_object_attributes_summarized_normalized[unique_df_object_object_key_columns].to_numpy().squeeze()
        )
        Log.important(
            str(self.__class__) + ' ' + str(getframeinfo(currentframe()).lineno)
            + ': Non-zero products from encoding: ' + str(set_non_zero_products)
        )
        zero_product_names_list = list( set(unique_attribute_columns).difference(set_non_zero_products) )
        Log.important(
            str(self.__class__) + ' ' + str(getframeinfo(currentframe()).lineno)
            + ': Zero products from encoding: ' + str(zero_product_names_list)
        )
        # Add to product encoding as 0 rows
        max_index = max(df_object_attributes_summarized_normalized.index)
        print(max_index)
        for zero_prd in zero_product_names_list:
            d = {k:np.inf for k in unique_attribute_columns}
            d[unique_df_object_object_key_columns[0]] = zero_prd
            df_row = pd.DataFrame.from_records(data=[d])
            df_object_attributes_summarized_normalized = df_object_attributes_summarized_normalized.append(df_row)
            df_object_attributes_summarized_normalized = df_object_attributes_summarized_normalized.reset_index(drop=True)
            Log.important(
                str(self.__class__) + ' ' + str(getframeinfo(currentframe()).lineno)
                + ': Appended zero product "' + str(zero_prd)
                + '" to product encoding successfully, assinged as: ' + str(d)
            )
        return df_object_attributes_summarized_normalized

    def get_object_distance(
            self,
            # Single tensor (np array)
            x_reference,
            # Array of other tensors (np array)
            y,
    ):
        return

    def recommend_products(
            self,
            # Any object with standard DNA (e.g. client, product, payment method)
            # np.array type. Одномерные, форма (1, n) чтобы упростить проблему
            # Например [[1 3 3]]
            obj_ref_dna,
            # np.array type. Многомерные, форма (m, n)
            # Например
            #   [
            #     [1.0 2.0 2.0],
            #     [2.5 2.0 2.5],
            #     [1.0 2.0 2.5],
            #     [1.0 2.0 2.0],
            #   ]
            df_product_dna,
            # List type, e.g. ['league']
            unique_prdname_cols,
            metric,
            force_normalization = False,
            how_many = 10,
            # Проблемой с include_purchased_product=False вклячается в том, что длины предложений может быть разными
            include_purchased_product = True,
            replace_purchased_product_with_nan = False,
    ):
        assert len(unique_prdname_cols) == 1, 'Multi-column product names not supported yet'
        obj_ref_dna = obj_ref_dna.astype(float)
        if len(obj_ref_dna.shape) == 1:
            # From [1,2,3] to [[1,2,3]]
            obj_ref_dna = np.reshape(obj_ref_dna, newshape=(1, obj_ref_dna.shape[0]))

        # нельзя
        if replace_purchased_product_with_nan:
            assert include_purchased_product == True

        # Get nan_product index
        condition = df_product_dna[unique_prdname_cols[0]] == SuggestDataProfile.NAN_PRODUCT
        df_row_nan_product = df_product_dna[condition]
        if len(df_row_nan_product) == 1:
            index_nan_product = df_row_nan_product.index[0]
        else:
            index_nan_product = None

        start_time = Profiling.start()
        attributes_list = self.extract_attributes_list(
            df = df_product_dna,
            unique_name_colums_list = unique_prdname_cols,
        )
        # Collapse to 1-dimensional vector
        np_product_names = df_product_dna[unique_prdname_cols].to_numpy().squeeze()
        Log.info(
            str(self.__class__) + ' ' + str(getframeinfo(currentframe()).lineno)
            + ': Extracted attributes list from product dna: ' + str(attributes_list)
            + ', product list: ' + str(np_product_names)
        )
        tensor_cmp = df_product_dna[attributes_list].values

        closest = self.find_closest(
            obj_ref_dna = obj_ref_dna,
            tensor_cmp  = tensor_cmp,
            how_many    = how_many,
            metric      = metric,
            force_normalization = force_normalization,
        )
        recommendations = np_product_names[closest]

        # если список продуктов было раньше сокращен, то продукты которые убраны не будут смены
        if replace_purchased_product_with_nan:
            if obj_ref_dna.shape[0] > 1:
                for i in range(len(recommendations)):
                    purchased_before = np.array(attributes_list)[obj_ref_dna[i] > 0]
                    replace_x = [(r in purchased_before) for r in recommendations[i]]
                    recommendations[i][replace_x] = SuggestDataProfile.NAN_PRODUCT
            else:
                purchased_before = np.array(attributes_list)[obj_ref_dna[0] > 0]
                replace_x = [(r in purchased_before) for r in recommendations]
                recommendations[replace_x] = SuggestDataProfile.NAN_PRODUCT

        self.profiler_recommend.profile_time(start_time=start_time)
        return recommendations.tolist()

    #
    # Given any object in standard DNA (tensor form or np array),
    # returns objects whose DNA is of close distance (any mathematical metric) to it.
    #
    def find_closest(
            self,
            # Any object with standard DNA (e.g. client, product, payment method)
            # np.array type. Одномерные, форма (1, n) чтобы упростить проблему
            # Например [[1 3 3]] или [[1 3 3], [5,1,2], [2,6,7], [9,3,4]]
            obj_ref_dna,
            # np.array type. Многомерные, форма (m, n)
            # Например
            #   [
            #     [1.0 2.0 2.0],
            #     [2.5 2.0 2.5],
            #     [1.0 2.0 2.5],
            #     [1.0 2.0 2.0],
            #   ]
            tensor_cmp,
            metric,
            # Для большей матрицы, вычисление нармализации очень медленно
            force_normalization,
            how_many = 0,
    ):
        obj_ref_dna = obj_ref_dna.astype(float)
        if len(obj_ref_dna.shape) == 1:
            # From [1,2,3] to [[1,2,3]]
            obj_ref_dna = np.reshape(obj_ref_dna, newshape=(1, obj_ref_dna.shape[0]))
        """
        Если вычислит предложения более одним клиентам (например один клиент [[1.0, 2.0, 2.0]]),
        нужно изменить из такого
            [
                [1.0 2.0 2.0],
                [2.5 2.0 2.5],
                [1.0 2.0 2.5],
                [1.0 2.0 2.0],
            ]
        размером (shape) (4,3)
        в такой
            [
                [ [1.0 2.0 2.0] ],
                [ [2.5 2.0 2.5] ],
                [ [1.0 2.0 2.5] ],
                [ [1.0 2.0 2.0] ],
            ]
        размером (shape) (4,1,3)
        """
        multi_client = obj_ref_dna.shape[0] > 1
        client_count = obj_ref_dna.shape[0]
        attribute_len = obj_ref_dna.shape[1]
        if multi_client:
            new_shape = (client_count, 1, attribute_len)
            Log.important(
                str(self.__class__) + ' ' + str(getframeinfo(currentframe()).lineno)
                + ': Reshaping from ' + str(obj_ref_dna.shape) + ' to ' + str(new_shape)
            )
            obj_ref_dna = np.reshape(obj_ref_dna, newshape=new_shape)
        tensor_cmp = tensor_cmp.astype(float)
        # print('*** ref dna   : ' + str(obj_ref_dna))
        # print('*** tensor cmp: ' + str(tensor_cmp))
        indxs_dist_sort = self.calculate_metric(
            x         = obj_ref_dna,
            prd_attrs = tensor_cmp,
            metric    = metric,
            force_normalization = force_normalization,
        )
        # print(indxs_dist_sort)

        if how_many > 0:
            if multi_client:
                indxs_dist_sort_truncate = indxs_dist_sort[:, 0:min(how_many, attribute_len)]
            else:
                indxs_dist_sort_truncate = indxs_dist_sort[0:min(how_many, attribute_len)]
        else:
            indxs_dist_sort_truncate = indxs_dist_sort

        return indxs_dist_sort_truncate

    def calculate_metric(
            self,
            x,
            prd_attrs,
            # Для большей матрицы, вычисление нармализации очень медленно
            force_normalization,
            metric,
    ):
        if force_normalization:
            # Для большей матрицы, это вычисление очень медленно
            x_new = self.normalize_euclidean(x=x)
            prd_attrs_new = self.normalize_euclidean(x=prd_attrs)
            Log.debug(
                str(self.__class__) + ' ' + str(getframeinfo(currentframe()).lineno)
                + ': x normalized: ' + str(x_new) + '\n\rp normalized: ' + str(prd_attrs_new)
            )
        else:
            x_new = x
            prd_attrs_new = prd_attrs

        """
        Суммирование по последней оси
        """
        # sum_axis = 1 + 1 * (ref_dna.shape[0] > 1)
        sum_axis = len(x_new.shape) - 1
        if metric == self.METRIC_COSINE:
            # Fast method just like NN layer
            distances = np.matmul(x_new, prd_attrs_new.transpose())
            # nan can occur for nan product with 0-vector
            condition_nan = np.isnan(distances)
            distances[condition_nan] = -1
            if sum_axis == 1:
                distances = np.reshape(distances, newshape=(prd_attrs_new.shape[0]))
                indxs_dist_sort = np.flip(np.argsort(distances), axis=0)
            else:
                distances = np.reshape(distances, newshape=(x_new.shape[0], prd_attrs_new.shape[0]))
                indxs_dist_sort = np.flip(np.argsort(distances), axis=1)
        elif metric == self.METRIC_EUCLIDEAN:
            # Slow, but more accurate for certain situations
            diff = x_new - prd_attrs_new
            distances = np.sqrt(np.sum((diff) ** 2, axis=sum_axis))
            # nan can occur for nan product with 0-vector
            condition_nan = np.isnan(distances)
            distances[condition_nan] = np.inf
            indxs_dist_sort = np.argsort(distances)
        else:
            raise Exception(
                str(self.__class__) + ' ' + str(getframeinfo(currentframe()).lineno)
                + ': No such metric "' + str(metric) + '" supported'
            )
        Log.debug(
            str(self.__class__) + ' ' + str(getframeinfo(currentframe()).lineno)
            + ': Distances: ' + str(distances) + ' indexes sorted: ' + str(indxs_dist_sort)
        )
        # Return the filtered data frame
        return indxs_dist_sort

    def find_furthest(
            self,
            obj_ref_dna,
            tensor_cmp,
            metric,
            force_normalization,
            how_many = 0,
    ):
        close_indexes = self.find_closest(
            obj_ref_dna = obj_ref_dna,
            tensor_cmp  = tensor_cmp,
            how_many    = how_many,
            metric      = metric,
            force_normalization = force_normalization,
        )
        return np.flip(close_indexes)

    def normalize_euclidean(
            self,
            x,
    ):
        start_time = Profiling.start()

        if len(x.shape) == 2:
            axis_sum = 1
        elif len(x.shape) == 3:
            axis_sum = 2
        else:
            raise Exception('Unexpected shape ' + str(x.shape))

        mags = np.sqrt((x**2).sum(axis=axis_sum))
        x_normalized = np.zeros(shape=x.shape)

        # TODO How to do without looping?
        for row in range(x.shape[0]):
            x_row = x[row]
            if axis_sum == 2:
                x_row = x[row][0]
            x_normalized[row] = x_row / mags[row]

        # Double check
        #mags_check = np.sqrt((x_normalized**2).sum(axis=axis_sum))
        #tmp_squares = (mags_check - np.ones(shape=mags_check.shape))**2
        #assert np.sum(tmp_squares) < 10**(-12), 'Check sum squares ' + str(np.sum(tmp_squares))

        self.profiler_normalize_euclidean.profile_time(
            start_time = start_time,
            additional_info = str(x.shape)
        )

        return x_normalized


class SuggestMetricUnitTest:
    def __init__(self, ut_params=None):
        self.ut_params = ut_params
        self.res_final = ResultObj(count_ok=0, count_fail=0)
        self.recommend_data_profile = SuggestDataProfile()
        self.recommend_metric = SuggestMetric()
        return

    def run_unit_test(self):
        self.__test_text()
        self.__test_water()
        return self.res_final

    def __test_text(self):
        def get_product_feature_vect(
                feature_template,
                prd_sentence,
                col_product_name,
        ):
            prd_dict = feature_template.copy()
            if col_product_name in prd_dict.keys():
                prd_dict[col_product_name] = prd
            words_list = prd_sentence.split(' ')
            for w in words_list:
                prd_dict[w] += 1
            return prd_dict

        equivalent_products = {
            'dep1': 'how crypto deposit', 'dep2': 'deposit method', 'dep3': 'how long deposit',
            'wid1': 'withdraw how', 'wid2': 'how long withdraw', 'wid3': 'withdraw method',
            'mat1': 'crap', 'mat2': 'slow like crap', 'mat3': 'crap site',
        }
        product_and_attributes_list = ['__product']
        attributes_list = []
        for sent in equivalent_products.values():
            [product_and_attributes_list.append(w) for w in sent.split(' ') if w not in product_and_attributes_list]
            [attributes_list.append(w) for w in sent.split(' ') if w not in attributes_list]

        feature_template_include_product = {w:0 for w in product_and_attributes_list}
        feature_template = {w:0 for w in attributes_list}
        prd_features = {}
        for prd in equivalent_products:
            prd_features[prd] = get_product_feature_vect(
                feature_template = feature_template_include_product,
                prd_sentence     = equivalent_products[prd],
                col_product_name = '__product',
            )

        df_product = pd.DataFrame.from_records(data=list(prd_features.values()))
        Log.debug('Product attributes: ' + str(attributes_list))
        Log.debug('Product features: ' + str(prd_features))
        Log.debug('Product profiles')
        Log.debug(df_product)

        metric_sent_expected = [
            [SuggestMetric.METRIC_EUCLIDEAN, 'dep1', ['dep1', 'dep3', 'dep2', 'wid1']],
            [SuggestMetric.METRIC_EUCLIDEAN, 'wid1', ['wid1', 'wid2', 'wid3', 'dep1']],
            [SuggestMetric.METRIC_EUCLIDEAN, 'mat1', ['mat1', 'mat3', 'mat2', 'dep2']],
            [SuggestMetric.METRIC_COSINE, 'dep1', ['dep1', 'dep3', 'wid1', 'dep2']],
            [SuggestMetric.METRIC_COSINE, 'wid1', ['wid1', 'wid2', 'wid3', 'dep3']],
            [SuggestMetric.METRIC_COSINE, 'mat1', ['mat1', 'mat3', 'mat2', 'wid3']],
        ]

        for m_v_e in metric_sent_expected:
            metric, sent, expected_recommendations = m_v_e
            ref_dna = get_product_feature_vect(
                feature_template = feature_template,
                prd_sentence     = equivalent_products[sent],
                col_product_name = None,
            )
            ref_dna = np.array(list(ref_dna.values()))
            recommendations = self.recommend_metric.recommend_products(
                obj_ref_dna    = ref_dna,
                df_product_dna = df_product,
                unique_prdname_cols = ['__product'],
                how_many       = 4,
                metric         = metric,
                force_normalization = (metric == SuggestMetric.METRIC_COSINE),
                include_purchased_product = True,
                replace_purchased_product_with_nan = False,
            )
            self.res_final.update_bool(res_bool=UnitTest.assert_true(
                observed     = recommendations,
                expected     = expected_recommendations,
                test_comment = 'Recomendations metric "' + str(metric) + '" for "' + str(sent)
                               + '" ' + str(recommendations) + ' expect ' + str(expected_recommendations)
            ))

        return

    def __test_water(self):
        """До того что сможет кодировать атрибуты, необходимо превращать форматы"""
        df_pokupki = pd.DataFrame({
            'client': ['a', 'a', 'b', 'b', 'c', 'c', 'd', 'd', 'e', 'e', 'f'],
            'product': ['borjomi', 'karspatskaya', 'borjomi', 'karspatskaya', 'illy', 'bonaqua', 'illy', 'lavazza', 'bonaqua', 'lavazza', 'lavazza'],
            'quantity': [1, 1, 2, 1, 1, 1, 2, 1, 2, 1, 2]
        })
        df_client_profiles, product_attributes_list = self.recommend_data_profile.convert_product_to_attributes(
            df_product                  = df_pokupki,
            unique_human_key_columns    = ['client'],
            unique_product_key_column   = 'product',
            unique_product_value_column = 'quantity',
            max_attribute_columns       = 0,
            transform_prd_values_method = SuggestDataProfile.TRANSFORM_PRD_VALUES_METHOD_NONE,
            add_unknown_product         = True,
        )
        Log.debug('Client profiles')
        Log.debug(df_client_profiles)
        Log.debug('Product as attributes')
        Log.debug(product_attributes_list)
        df_mapped_product = self.recommend_metric.encode_product_attributes(
            df_human_profile         = df_client_profiles,
            df_object                = df_pokupki,
            unique_human_key_columns = ['client'],
            unique_df_object_human_key_columns  = ['client'],
            unique_df_object_object_key_columns = ['product'],
            unique_df_object_value_column       = 'quantity',
            unique_df_object_human_attribute_columns = product_attributes_list,
            apply_object_value_as_weight        = False,
            # В реальном применении, нужно нормализирован через NORMALIZE_METHOD_UNIT чтобы стали единичними векторами
            normalize_method                    = SuggestDataProfile.NORMALIZE_METHOD_PROB,
        )
        Log.debug('Product profiles')
        Log.debug(df_mapped_product)

        self.res_final.update_bool(res_bool=UnitTest.assert_true(
            observed = product_attributes_list,
            expected = ['bonaqua', 'borjomi', 'illy', 'karspatskaya', 'lavazza', '__NAN_PRODUCT'],
            test_comment = 'attribute list ' + str(product_attributes_list)
        ))

        x_vec = np.array([1, 0, 0, 0, 0, 0])
        y_vec = np.array([0, 1, 0, 0, 0, 0])
        z_vec = np.array([0, 0, 1, 0, 0, 0])
        nanprd = SuggestDataProfile.NAN_PRODUCT
        x_expected_rec_euclidean = ['bonaqua', 'lavazza', 'illy', 'borjomi', 'karspatskaya', nanprd]
        x_expected_rec_cosine    = ['bonaqua', 'lavazza', 'illy', 'karspatskaya', 'borjomi', nanprd]
        y_expected_rec_euclidean = ['borjomi', 'karspatskaya', 'lavazza', 'bonaqua', 'illy', nanprd]
        y_expected_rec_cosine    = ['karspatskaya', 'borjomi', 'lavazza', 'illy', 'bonaqua', nanprd]
        z_expected_rec_euclidean = ['illy', 'lavazza', 'bonaqua', 'borjomi', 'karspatskaya', nanprd]
        z_expected_rec_cosine    = ['illy', 'lavazza', 'bonaqua', 'karspatskaya', 'borjomi', nanprd]

        vecs_all = np.array([x_vec, y_vec, z_vec])
        expected_cosine_all = [x_expected_rec_cosine, y_expected_rec_cosine, z_expected_rec_cosine]
        expected_euclidean_all = [x_expected_rec_euclidean, y_expected_rec_euclidean, z_expected_rec_euclidean]

        metric_vec_expected = [
            [SuggestMetric.METRIC_EUCLIDEAN, x_vec, x_expected_rec_euclidean],
            [SuggestMetric.METRIC_EUCLIDEAN, y_vec, y_expected_rec_euclidean],
            [SuggestMetric.METRIC_EUCLIDEAN, z_vec, z_expected_rec_euclidean],
            [SuggestMetric.METRIC_COSINE, x_vec, x_expected_rec_cosine],
            [SuggestMetric.METRIC_COSINE, y_vec, y_expected_rec_cosine],
            [SuggestMetric.METRIC_COSINE, z_vec, z_expected_rec_cosine],
            [SuggestMetric.METRIC_EUCLIDEAN, vecs_all, expected_euclidean_all],
            [SuggestMetric.METRIC_COSINE, vecs_all, expected_cosine_all],
        ]
        for m_v_e in metric_vec_expected:
            metric, vec, expected_recommendations = m_v_e
            recommendations = self.recommend_metric.recommend_products(
                obj_ref_dna    = vec,
                df_product_dna = df_mapped_product,
                unique_prdname_cols = ['product'],
                metric = metric,
                force_normalization = (metric == SuggestMetric.METRIC_COSINE),
                include_purchased_product = True,
                replace_purchased_product_with_nan = False,
            )
            self.res_final.update_bool(res_bool=UnitTest.assert_true(
                observed     = recommendations,
                expected     = expected_recommendations,
                test_comment = 'Inclusive recomendations metric "' + str(metric) + '" for "' + str(vec)
                               + '" ' + str(recommendations) + ' expect ' + str(expected_recommendations)
            ))

            # TEST INCLUSIVE WITH REPLACE WITH NAN_PRODUCT
            recommendations = self.recommend_metric.recommend_products(
                obj_ref_dna    = vec,
                df_product_dna = df_mapped_product,
                unique_prdname_cols = ['product'],
                metric = metric,
                force_normalization = (metric == SuggestMetric.METRIC_COSINE),
                include_purchased_product = True,
                replace_purchased_product_with_nan = True,
            )
            if vec.shape != vecs_all.shape:
                purchased_before = np.array(product_attributes_list)[vec > 0]
                np_exp_recmd = np.array(expected_recommendations)
                replace_x = [(r in purchased_before) for r in np_exp_recmd]
                np_exp_recmd[replace_x] = SuggestDataProfile.NAN_PRODUCT
                expected_recommendations = list(np_exp_recmd)
            else:
                np_exp_recmd = np.array(expected_recommendations)
                for i in range(len(np_exp_recmd)):
                    purchased_before = np.array(product_attributes_list)[vecs_all[i] > 0]
                    replace_x = [(r in purchased_before) for r in np_exp_recmd[i]]
                    np_exp_recmd[i][replace_x] = SuggestDataProfile.NAN_PRODUCT
                expected_recommendations = [list(row) for row in list(np_exp_recmd)]

            self.res_final.update_bool(res_bool=UnitTest.assert_true(
                observed     = recommendations,
                expected     = expected_recommendations,
                test_comment = 'Inclusive (replaced) recomendations metric "' + str(metric) + '" for "' + str(vec)
                               + '" ' + str(recommendations) + ' expect ' + str(expected_recommendations)
            ))



if __name__ == '__main__':
    Log.DEBUG_PRINT_ALL_TO_SCREEN = 1
    Log.LOGLEVEL = Log.LOG_LEVEL_DEBUG_1

    res = SuggestMetricUnitTest().run_unit_test()
    print('PASSED ' + str(res.count_ok) + ', FAILED ' + str(res.count_fail))
    exit(res.count_fail)
