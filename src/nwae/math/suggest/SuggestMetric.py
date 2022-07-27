# -*- coding: utf-8 -*-

from nwae.utils.Log import Log
from inspect import getframeinfo, currentframe
import numpy as np
import pandas as pd
from nwae.utils.data.DataPreprcscr import DataPreprocessor
from nwae.math.suggest.SuggestDataProfile import SuggestDataProfile
from nwae.utils.Profiling import Profiling, ProfilingHelper


class SuggestMetric:

    # Быстро как в нейронных сетях
    METRIC_COSINE = 'cosine'
    # Медленно
    METRIC_EUCLIDEAN = 'euclidean'

    ALGO_BY_CLOSEST_PRODUCT = 'byclosestproduct'
    ALGO_BY_CLOSEST_CLIENT  = 'byclosestclient'

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
            # Обязательно только уникальные покупатели
            df_human_profile,
            # Столцы которые определяют уникальных клиентов
            unique_human_key_columns,
            # Обязательно только уникальные пари (покупатель, товар)
            df_object,
            unique_df_object_human_key_columns,
            unique_df_object_object_key_columns,
            unique_df_object_value_column,
            unique_df_object_human_attribute_columns,
            apply_object_value_as_weight,
            # 'none', 'unit' (единичный вектор) or 'prob' (сумма атрибутов = 1)
            normalize_method,
    ):
        assert len(unique_human_key_columns) == 1, \
            'Multiple human key columns not supported ' + str(unique_human_key_columns)
        assert len(unique_df_object_object_key_columns) == 1, \
            'Multiple product columns not supported ' + str(unique_df_object_object_key_columns)

        colkeep = unique_df_object_human_key_columns \
                  + unique_df_object_object_key_columns \
                  + [unique_df_object_value_column]
        df_object_clean = df_object[colkeep]

        # # Обязательно только уникальные пари (покупатель, товар)
        column_human_prd = [unique_human_key_columns[0], unique_df_object_object_key_columns[0]]
        df_object_agg = df_object_clean.groupby(
            column_human_prd,
            as_index = False,
        ).sum().reset_index(drop=True)

        # # Обязательно только уникальные пари (покупатель, товар)
        # df_test_unique_pairs = df_object_agg.copy()
        # df_test_unique_pairs['__count'] = 1
        # df_test_unique_pairs_agg = df_test_unique_pairs.groupby(
        #     column_human_prd,
        #     as_index=False,
        # ).sum()
        # condition_problem = df_test_unique_pairs_agg['__count'] > 1
        # df_test_problem_rows = df_test_unique_pairs_agg[condition_problem]
        # assert len(df_test_problem_rows) == 0, \
        #     'Dataframe for human/product must only have unique client/product pairs: ' + str(df_test_problem_rows)

        # Обязательно только уникальные покупатели
        clients_list = np.unique(df_human_profile[unique_human_key_columns[0]])
        assert len(clients_list) == len(df_human_profile), \
            'Dataframe for human profile must only have unique clients: ' + str(clients_list)

        # Merge
        df_object_human_attributes = df_object_agg.merge(
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
        # Log.info(
        #     str(self.__class__) + ' ' + str(getframeinfo(currentframe()).lineno)
        #     + ': Object human attributes (first 20 lines): ' + str(df_object_human_attributes[0:20])
        # )
        # df_object_human_attributes.to_csv('object_human.csv')

        # Больше не нужны клиенты
        colkeep = unique_df_object_object_key_columns \
                  + [unique_df_object_value_column] \
                  + unique_df_object_human_attribute_columns
        df_object_attributes = df_object_human_attributes[colkeep]

        df_product_encoding = self.__encode_product(
            df_object_attributes                = df_object_attributes,
            unique_df_object_object_key_columns = unique_df_object_object_key_columns,
            unique_df_object_value_column       = unique_df_object_value_column,
            unique_attribute_columns            = unique_df_object_human_attribute_columns,
            apply_object_value_as_weight        = apply_object_value_as_weight,
            normalize_method                    = normalize_method,
        )
        # if include_filtered_out_products:
        #     return df_product_encoding
        # else:
        #     condition = df_product_encoding[unique_df_object_object_key_columns[0]].isin(unique_df_object_human_attribute_columns)
        #     df_product_encoding_truncate = df_product_encoding[condition]
        #     return df_product_encoding_truncate
        return df_product_encoding

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
        for zero_prd in zero_product_names_list:
            d = {k:np.inf for k in unique_attribute_columns}
            d[unique_df_object_object_key_columns[0]] = zero_prd
            df_row = pd.DataFrame.from_records(data=[d])
            # TODO FutureWarning: The frame.append method is deprecated and will be removed from pandas in a future version. Use pandas.concat instead.
            # df_object_attributes_summarized_normalized = df_object_attributes_summarized_normalized.append(df_row)
            df_object_attributes_summarized_normalized = pd.concat(
                [df_object_attributes_summarized_normalized, df_row],
                axis = 0,
                join = 'outer'
            )
            df_object_attributes_summarized_normalized = df_object_attributes_summarized_normalized.reset_index(drop=True)
            Log.important(
                str(self.__class__) + ' ' + str(getframeinfo(currentframe()).lineno)
                + ': Appended zero product "' + str(zero_prd)
                + '" to product encoding successfully, assigned as: ' + str(d)
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

    """
    Вроде предложению "покупатиели купившие этот продукт, тоже купили эти продукты"
    """
    def recommend_products_by_product_names_only(
            self,
            product_names_list,
            df_product_dna,
            # List type, e.g. ['league']
            unique_prdname_cols,
            replace_purchased_product_with_nan = True,
    ):
        assert len(unique_prdname_cols) == 1, 'Multi-column product names not supported yet'

        attributes_list = self.extract_attributes_list(
            df = df_product_dna,
            unique_name_colums_list = unique_prdname_cols,
        )
        np_attributes_list = np.array(attributes_list)
        # # Collapse to 1-dimensional vector
        # np_product_names = df_product_dna[unique_prdname_cols].to_numpy().squeeze()
        Log.info(
            str(self.__class__) + ' ' + str(getframeinfo(currentframe()).lineno)
            + ': Extracted attributes list from product dna: ' + str(np_attributes_list)
            # + ', product list: ' + str(np_product_names)
        )

        condition = False
        for prd in product_names_list:
            condition = condition | (df_product_dna[unique_prdname_cols[0]] == prd)
        df_prd_only = df_product_dna[condition]

        np_probs = df_prd_only[attributes_list].values
        # Sort by highest probability to lowest
        indxs_dist_sort = np.flip(np.argsort(np_probs), axis=1)

        np_recommendations = np_attributes_list[indxs_dist_sort]
        # если список продуктов было раньше сокращен, то продукты которые убраны не будут смены
        if replace_purchased_product_with_nan:
            for i in range(len(np_recommendations)):
                condition = np_recommendations[i] == product_names_list[i]
                purchased_before = np_recommendations[i][condition]
                replace_x = [(r in purchased_before) for r in np_recommendations[i]]
                np_recommendations[i][replace_x] = SuggestDataProfile.NAN_PRODUCT

        return np_recommendations.tolist()

    def convert_x_to_desired_shape(
            self,
            x,
    ):
        if x is None:
            return None
        x = x.astype(float)
        if len(x.shape) == 1:
            # From [1,2,3] to [[1,2,3]]
            x = np.reshape(x, newshape=(1, x.shape[0]))
        return x

    """
    алгоритм "byclosestproduct"
      1. ищет самые близкие товары в df_product_dna
      3. предложить товары упорядочены выше
    
    алгоритм "byclosestclient"
      1. ищет самых близких клиентов
      2. из этих клиентов, собирать все товары, купивщие ими
      3. предложить товары упорядочены по значению
    """
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
            replace_purchased_product_with_nan = False,
            include_products_not_in_attributes = True,
            # ALGO_BY_CLOSEST_PRODUCT, ALGO_BY_CLOSEST_CLIENT
            algorithm = ALGO_BY_CLOSEST_PRODUCT,
            # If algorithm==ALGO_BY_CLOSEST_CLIENT, need these
            how_many_humans   = 10,
            unique_human_cols = None,
            df_human_dna = None
    ):
        assert len(unique_prdname_cols) == 1, 'Multi-column product names not supported yet ' + str(unique_prdname_cols)
        assert algorithm in [self.ALGO_BY_CLOSEST_PRODUCT, self.ALGO_BY_CLOSEST_CLIENT]

        shape_prd_dna_ori = df_product_dna.shape
        obj_ref_dna = self.convert_x_to_desired_shape(x=obj_ref_dna)

        start_time = Profiling.start()
        attributes_list = self.extract_attributes_list(
            df = df_product_dna,
            unique_name_colums_list = unique_prdname_cols,
        )

        if not include_products_not_in_attributes:
            condition = df_product_dna[unique_prdname_cols[0]].isin(attributes_list)
            df_product_dna_modified = df_product_dna[condition].reset_index(drop=True)
        else:
            df_product_dna_modified = df_product_dna

        # Collapse to 1-dimensional vector
        np_product_names = df_product_dna_modified[unique_prdname_cols].to_numpy().squeeze()
        Log.info(
            str(self.__class__) + ' ' + str(getframeinfo(currentframe()).lineno)
            + ': Extracted attributes list from product dna: ' + str(attributes_list)
            + ', product list: ' + str(np_product_names)
        )

        # if is None, means get default recommendation
        if obj_ref_dna is None:
            attributes_len = len(attributes_list)
            return [ attributes_list[0:min(how_many, attributes_len)] ]

        if algorithm == self.ALGO_BY_CLOSEST_PRODUCT:
            tensor_cmp_product = df_product_dna_modified[attributes_list].values
            closest_products = self.find_closest(
                obj_ref_dna = obj_ref_dna,
                tensor_cmp  = tensor_cmp_product,
                how_many    = how_many,
                metric      = metric,
                force_normalization = force_normalization,
            )
            recommendations = np_product_names[closest_products]
            if len(recommendations.shape) == 1:
                # From [1,2,3] to [[1,2,3]]
                recommendations = np.reshape(recommendations, newshape=(1, recommendations.shape[0]))
        elif algorithm == self.ALGO_BY_CLOSEST_CLIENT:
            assert len(obj_ref_dna) == 1,\
                'Multi client recommendation for algorithm "' + str(algorithm) + '" not supported'
            tensor_cmp_human = df_human_dna[attributes_list].values
            closest_humans = self.find_closest(
                obj_ref_dna = obj_ref_dna,
                tensor_cmp  = tensor_cmp_human,
                how_many    = how_many_humans,
                metric      = metric,
                force_normalization = force_normalization,
            )
            # Collapse to 1-dimensional vector
            np_human_names = df_human_dna[unique_human_cols].to_numpy().squeeze()
            recommendations_human = np_human_names[closest_humans]
            Log.debug(
                str(self.__class__) + ' ' + str(getframeinfo(currentframe()).lineno)
                + ': Closest humans extracted: ' + str(recommendations_human)
            )
            # отфильтровать товары связанны с этими близкими клиентами
            idx = list(df_human_dna.index)
            assert idx == list(range(len(df_human_dna))),\
                'Index of human dna must be in order: ' + str(df_human_dna.index)
            df_associated_humans = df_human_dna.loc[closest_humans]
            df_associated_products = df_associated_humans[attributes_list].transpose()
            np_product_names_associated = np.array(df_associated_products.index, dtype=str)
            # Sum total of product purchase
            series_associated_products_sum = np.array(df_associated_products.sum(axis=1))
            Log.debug(
                str(self.__class__) + ' ' + str(getframeinfo(currentframe()).lineno)
                + ': Associated products: ' + str(df_associated_products)
            )
            # Order by highest total purchase value
            np_order = np.flip(np.argsort(series_associated_products_sum))
            recommendations = np_product_names_associated[np_order]
            recommendations = recommendations[0:min(how_many, len(recommendations))]
            Log.debug(
                str(self.__class__) + ' ' + str(getframeinfo(currentframe()).lineno)
                + ': Closest associated products: ' + str(recommendations)
            )
            if len(recommendations.shape) == 1:
                # From [1,2,3] to [[1,2,3]]
                recommendations = np.reshape(recommendations, newshape=(1, recommendations.shape[0]))
        else:
            raise Exception(
                str(self.__class__) + ' ' + str(getframeinfo(currentframe()).lineno)
                + ': Unknown algorithm "' + str(algorithm) + '"'
            )

        # если список продуктов было раньше сокращен, то продукты которые убраны не будут смены
        if replace_purchased_product_with_nan:
            # TODO Как вычислить без цикла?
            for i in range(len(recommendations)):
                purchased_before = np.array(attributes_list)[obj_ref_dna[i] > 0]
                replace_x = [(r in purchased_before) for r in recommendations[i]]
                recommendations[i][replace_x] = SuggestDataProfile.NAN_PRODUCT

        self.profiler_recommend.profile_time(start_time=start_time)
        # Make sure we don't modify the dataframe passed in
        assert df_product_dna.shape == shape_prd_dna_ori, \
            'Shape of product dna ' + str(shape_prd_dna_ori) + ' must not change'
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
        obj_ref_dna = self.convert_x_to_desired_shape(x=obj_ref_dna)
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


if __name__ == '__main__':
    Log.DEBUG_PRINT_ALL_TO_SCREEN = 1
    Log.LOGLEVEL = Log.LOG_LEVEL_DEBUG_1

    from nwae.math.suggest.SuggestUnitTest import SuggestMetricUnitTest
    res = SuggestMetricUnitTest().run_unit_test()
    print('PASSED ' + str(res.count_ok) + ', FAILED ' + str(res.count_fail))
    exit(res.count_fail)
