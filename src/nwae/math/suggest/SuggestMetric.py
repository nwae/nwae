# -*- coding: utf-8 -*-

from nwae.utils.Log import Log
from inspect import getframeinfo, currentframe
import numpy as np
import pandas as pd
from nwae.utils.data.DataPreprcscr import DataPreprocessor
from nwae.utils.UnitTest import UnitTest, ResultObj
from nwae.math.suggest.SuggestDataProfile import SuggestDataProfile

pd.set_option('display.max_rows', 500)
pd.set_option('display.max_columns', 500)
pd.set_option('display.width', 1000)


class SuggestMetric:

    # Быстро как в нейронных сетях
    METRIC_COSINE = 'cosine'
    # Медленно
    METRIC_EUCLIDEAN = 'euclidean'

    def __init__(self):
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
        # TODO
        #    Сейчас только самый простой метод вычислить "аттрибуты" объектов
        Log.important(
            str(self.__class__) + ' ' + str(getframeinfo(currentframe()).lineno)
            + ': Start encoding product.. '
        )
        colkeep = unique_df_object_object_key_columns + [unique_df_object_value_column]
        df_agg_value = df_object_attributes[colkeep].groupby(
            by=unique_df_object_object_key_columns,
            as_index=False,
        ).sum()
        df_agg_value.columns = unique_df_object_object_key_columns + ['__total_value']
        df_object_attributes = df_object_attributes.merge(
            df_agg_value,
            on=unique_df_object_object_key_columns,
            how='left'
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
            by=unique_df_object_object_key_columns,
            as_index=False,
        ).sum()
        colkeep = unique_df_object_object_key_columns + unique_attribute_columns
        df_object_attributes_summarized = df_object_attributes_summarized[colkeep]
        # df_object_attributes_summarized.to_csv('object_attr_summary.csv')

        original_cols = list(df_object_attributes_summarized.columns)
        name_col = original_cols[0]
        attr_cols = original_cols.copy()
        attr_cols.remove(name_col)

        return SuggestDataProfile.normalize(
            df                = df_object_attributes_summarized,
            name_columns      = [name_col],
            attribute_columns = attr_cols,
            normalize_method  = normalize_method,
        )

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
            how_many = 10,
            include_purchased_product = True,
    ):
        attributes_list = self.extract_attributes_list(
            df=df_product_dna,
            unique_name_colums_list=unique_prdname_cols,
        )
        # Collapse to 1-dimensional vector
        np_product_names = df_product_dna[unique_prdname_cols].to_numpy().squeeze()
        Log.debugdebug(
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
        )
        recommendations = np_product_names[closest]
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
            how_many = 0,
            # TODO
            include_purchased_product=True,
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
        )
        if how_many > 0:
            if multi_client:
                return indxs_dist_sort[:, 0:min(how_many, attribute_len)]
            else:
                return indxs_dist_sort[0:min(how_many, attribute_len)]
        else:
            return indxs_dist_sort

    def calculate_metric(
            self,
            x,
            prd_attrs,
            metric = METRIC_EUCLIDEAN
    ):
        if metric == self.METRIC_COSINE:
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
            if sum_axis == 1:
                distances = np.reshape(distances, newshape=(prd_attrs_new.shape[0]))
            else:
                distances = np.reshape(distances, newshape=(x_new.shape[0], prd_attrs_new.shape[0]))
            indxs_dist_sort = np.flip(np.argsort(distances))
        elif metric == self.METRIC_EUCLIDEAN:
            # Slow, but more accurate for certain situations
            diff = x_new - prd_attrs_new
            distances = np.sqrt(np.sum((diff) ** 2, axis=sum_axis))
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
            how_many=0,
    ):
        close_indexes = self.find_closest(
            obj_ref_dna = obj_ref_dna,
            tensor_cmp  = tensor_cmp,
            how_many    = how_many
        )
        return np.flip(close_indexes)

    def normalize_euclidean(
            self,
            x,
    ):
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
        mags_check = np.sqrt((x_normalized**2).sum(axis=axis_sum))
        tmp_squares = (mags_check - np.ones(shape=mags_check.shape))**2
        assert np.sum(tmp_squares) < 10**(-12), 'Check sum squares ' + str(np.sum(tmp_squares))

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
        print('Product attributes: ' + str(attributes_list))
        print('Product features: ' + str(prd_features))
        print('****** To mapped product words')
        print(df_product)

        for human_recommendations in [
            ['dep1', ['dep1', 'dep3', 'dep2', 'wid1']],
            ['wid1', ['wid1', 'wid2', 'wid3', 'dep1']],
            ['mat1', ['mat1', 'mat3', 'mat2', 'dep2']],
        ]:
            human = human_recommendations[0]
            expected_recommendations = human_recommendations[1]
            ref_dna = get_product_feature_vect(
                feature_template = feature_template,
                prd_sentence     = equivalent_products[human],
                col_product_name = None,
            )
            ref_dna = np.array(list(ref_dna.values()))
            print(ref_dna)
            recommendations = self.recommend_metric.recommend_products(
                obj_ref_dna    = ref_dna,
                df_product_dna = df_product,
                unique_prdname_cols = ['__product'],
                how_many       = 4,
                metric         = SuggestMetric.METRIC_EUCLIDEAN,
            )
            self.res_final.update_bool(res_bool=UnitTest.assert_true(
                observed     = recommendations,
                expected     = expected_recommendations,
                test_comment = 'Recomendations for "' + str(human) + '" ' + str(recommendations) + ' expect ' + str(
                    expected_recommendations)
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
        )
        print('Client profiles')
        print(df_client_profiles)
        print('Product as attributes')
        print(product_attributes_list)
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
        print('Product profiles')
        print(df_mapped_product)

        # For phone
        x_vec = np.array([1, 0, 0, 0, 0])
        y_vec = np.array([0, 1, 0, 0, 0])
        z_vec = np.array([0, 0, 1, 0, 0])
        x_expected_rec = ['bonaqua', 'lavazza', 'illy', 'borjomi', 'karspatskaya']
        y_expected_rec = ['borjomi', 'karspatskaya', 'lavazza', 'bonaqua', 'illy']
        z_expected_rec = ['illy', 'lavazza', 'bonaqua', 'borjomi', 'karspatskaya']
        for human_recommendations in [
            [x_vec, x_expected_rec],
            [y_vec, y_expected_rec],
            [z_vec, z_expected_rec],
        ]:
            vec = human_recommendations[0]
            expected_recommendations = human_recommendations[1]
            recommendations = self.recommend_metric.recommend_products(
                obj_ref_dna    = vec,
                df_product_dna = df_mapped_product,
                unique_prdname_cols = ['product'],
                metric = SuggestMetric.METRIC_EUCLIDEAN,
            )
            self.res_final.update_bool(res_bool=UnitTest.assert_true(
                observed     = recommendations,
                expected     = expected_recommendations,
                test_comment = 'Recomendations for "' + str(vec) + '" ' + str(recommendations) + ' expect ' + str(
                    expected_recommendations)
            ))

        # Сразу вычислить для всех клинтов
        vecs = np.array([x_vec, y_vec, z_vec])
        print('vecs: ' + str(vecs))
        recommendations = self.recommend_metric.recommend_products(
            obj_ref_dna    = vecs,
            df_product_dna = df_mapped_product,
            unique_prdname_cols = ['product'],
            metric         = SuggestMetric.METRIC_EUCLIDEAN,
        )
        expected_recommendations = [x_expected_rec, y_expected_rec, z_expected_rec]
        self.res_final.update_bool(res_bool=UnitTest.assert_true(
            observed     = recommendations,
            expected     = expected_recommendations,
            test_comment = 'Recomendations for all: ' + str(recommendations) + ' expect ' + str(
                expected_recommendations)
        ))


if __name__ == '__main__':
    Log.DEBUG_PRINT_ALL_TO_SCREEN = 1
    Log.LOGLEVEL = Log.LOG_LEVEL_DEBUG_1

    res = SuggestMetricUnitTest().run_unit_test()
    print('PASSED ' + str(res.count_ok) + ', FAILED ' + str(res.count_fail))
    exit(res.count_fail)
