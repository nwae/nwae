# -*- coding: utf-8 -*-

from nwae.utils.Log import Log
from inspect import getframeinfo, currentframe
import numpy as np
import pandas as pd
from nwae.utils.UnitTest import UnitTest, ResultObj


"""
Обрабатывает общие данные покупок-то в "профили". Например из данных
     client,      product, quantity
          a,      borjomi,       1
          a, karspatskaya,       1
          b,      borjomi,       2
          b, morshinskaya,       1
          c,      bonaqua,       1
          c, morshinskaya,       3
в такие (NORMALIZE_METHOD_NONE), где атрибумами являются продукты
      client  bonaqua  borjomi  karspatskaya  morshinskaya
    0      a      0.0      1.0           1.0           0.0
    1      b      0.0      2.0           0.0           1.0
    2      c      1.0      0.0           0.0           3.0
то есть атрибуты клиента становятся (bonaqua, borjomi, karspatskaya, morshinskaya)
"""
class SuggestDataProfile:

    # 'none', 'unit' (единичный вектор) or 'prob' (сумма атрибутов = 1)
    NORMALIZE_METHOD_NONE = 'none'
    NORMALIZE_METHOD_UNIT = 'unit'
    NORMALIZE_METHOD_PROB = 'prob'

    COLNAME_PRODUCTS_NOT_INCLUDED = '__others'

    TRANSFORM_PRD_VALUES_METHOD_NONE = 'none'
    # Все значения трансакций в 1
    TRANSFORM_PRD_VALUES_METHOD_UNITY = 'unity'
    # Все значения трансакций x как log(1+x)
    TRANSFORM_PRD_VALUES_METHOD_LOG = 'log'

    @staticmethod
    def normalize(
            df,
            name_columns,
            attribute_columns,
            normalize_method,
    ):
        Log.info(
            str(__name__) + ' ' + str(getframeinfo(currentframe()).lineno)
            + ': Start normalizing process by "' + str(normalize_method) + '"...'
        )
        if normalize_method == SuggestDataProfile.NORMALIZE_METHOD_PROB:
            df_attr = df[attribute_columns]
            df_attr = df_attr.apply(lambda x: x / sum(x), axis=1)
        elif normalize_method == SuggestDataProfile.NORMALIZE_METHOD_UNIT:
            df_attr = df[attribute_columns]
            df_attr = df_attr.apply(lambda x: x / sum(x ** 2) ** 0.5, axis=1)
        else:
            return df

        for col in name_columns:
            df_attr[col] = df[col]
        keep_cols = name_columns + attribute_columns
        df = df_attr[keep_cols].reset_index(drop=True)

        return df

    def __init__(self):
        pd.set_option('display.max_rows', 500)
        pd.set_option('display.max_columns', 500)
        pd.set_option('display.width', 1000)
        return

    def convert_product_to_attributes(
            self,
            # Датафрейм с покупателями и продуктами
            df_product,
            # Столцы которые определяют уникальных клиентов
            unique_human_key_columns,
            unique_product_key_column,
            # Либо цена продуктов или количество
            unique_product_value_column,
            normalize_method = NORMALIZE_METHOD_NONE,
            # TODO уменшить количество атрибутов
            max_attribute_columns = 0,
            # Before any processing
            transform_prd_values_method = TRANSFORM_PRD_VALUES_METHOD_NONE,
    ):
        df_prd_agg, unique_product_list, unique_human_list = self.aggregate_products(
            df_product                  = df_product,
            unique_human_key_columns    = unique_human_key_columns,
            unique_product_key_column   = unique_product_key_column,
            unique_product_value_column = unique_product_value_column,
            transform_prd_values_method = transform_prd_values_method
        )

        #
        # Этот шаг приведет к несколько проблемам, так как у некоторых клиентов будет превратить к вектору нулей
        #
        is_reduced = False
        unique_remaining_products = None
        np_remaining_attributes = None
        if max_attribute_columns > 0:
            if len(unique_product_list) > max_attribute_columns:
                unique_top_products_by_order, unique_remaining_products = self.find_top_products(
                    df_product                  = df_product,
                    unique_product_key_column   = unique_product_key_column,
                    unique_product_value_column = unique_product_value_column,
                    top_x                       = max_attribute_columns
                )
                Log.info(
                    str(self.__class__) + ' ' + str(getframeinfo(currentframe()).lineno)
                    + ': Remaining ' + str(len(unique_remaining_products)) + ' least products: ' + str(
                        unique_remaining_products)
                )

                # Change the names to one name
                def change_name(prdname):
                    if prdname in unique_remaining_products:
                        return self.COLNAME_PRODUCTS_NOT_INCLUDED
                    else:
                        return prdname

                df_prd_agg[unique_product_key_column] = df_prd_agg[unique_product_key_column].apply(func=change_name)
                unique_product_list = unique_top_products_by_order + [self.COLNAME_PRODUCTS_NOT_INCLUDED]
                Log.important(
                    str(self.__class__) + ' ' + str(getframeinfo(currentframe()).lineno)
                    + ': After truncation, total unique products as attributes = ' + str(len(unique_product_list))
                    + '. Products: ' + str(unique_product_list)
                )
                # Need to regroup again, since each pair member-COLNAME_PRODUCTS_NOT_INCLUDED will appear on multiple lines
                shape_ori = df_prd_agg.shape
                df_prd_agg = df_prd_agg.groupby(
                    by=unique_human_key_columns + [unique_product_key_column],
                    as_index=False,
                ).sum()
                Log.info(
                    str(self.__class__) + ' ' + str(getframeinfo(currentframe()).lineno)
                    + ': After second round grouping by human/product columns, from shape ' + str(shape_ori)
                    + ' to new shape ' + str(df_prd_agg.shape)
                )

        """Датафрейм лишь с столбцом(ами) покупателей"""
        df_converted = df_prd_agg[unique_human_key_columns]
        df_converted = df_converted.drop_duplicates()
        Log.debugdebug(df_converted)
        assert len(df_converted) == len(unique_human_list), \
            'Length of final dataframe ' + str(len(df_converted)) + ' must be equal ' + str(len(unique_human_list))

        columns_running = list(df_converted.columns)
        """Продукт за продуктом, создавать столбец продукта"""
        for prd in unique_product_list:
            condition_only_this_product = df_prd_agg[unique_product_key_column] == prd
            df_prd_agg_part = df_prd_agg[condition_only_this_product]
            columns_keep = unique_human_key_columns + [unique_product_value_column]
            df_prd_agg_part = df_prd_agg_part[columns_keep].reset_index(drop=True)
            Log.debugdebug(prd)
            Log.debugdebug(df_prd_agg_part)
            """Соединить цену/количество продукта с человеком"""
            df_converted = df_converted.merge(
                df_prd_agg_part,
                on=unique_human_key_columns,
                how='left'
            )
            assert len(df_converted) == len(unique_human_list), \
                'After merge column "' + str(prd) + '" Length of final dataframe ' + str(
                    len(df_converted)) + ' must be equal ' + str(len(unique_human_list))

            """Новые имена столбцев"""
            columns_running = columns_running + [prd]
            df_converted.columns = columns_running
            df_converted[prd] = df_converted[prd].fillna(0.0)
            Log.debugdebug(df_converted)

        assert len(df_converted) == len(unique_human_list), \
            'Length of final dataframe ' + str(len(df_converted)) + ' must be equal ' + str(len(unique_human_list))

        if is_reduced:
            df_converted[self.COLNAME_PRODUCTS_NOT_INCLUDED] = np_remaining_attributes

        Log.important(
            str(self.__class__) + ' ' + str(getframeinfo(currentframe()).lineno)
            + ': Final human-product attributes shape: ' + str(df_converted.shape)
        )
        Log.debugdebug(
            str(self.__class__) + ' ' + str(getframeinfo(currentframe()).lineno)
            + ': Final mapped human-product attributes: '
        )
        Log.debugdebug(df_converted)

        original_cols = list(df_converted.columns)
        attr_cols = original_cols.copy()
        for col in unique_human_key_columns:
            attr_cols.remove(col)
        df_converted = self.normalize(
            df=df_converted,
            name_columns=unique_human_key_columns,
            attribute_columns=attr_cols,
            normalize_method=normalize_method,
        )

        return df_converted, unique_product_list

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

    """
    Входные данные
        product,quantity
        a,1
        b,1
        a,2
        b,1
        c,1
        d,3
        ..
    """
    def find_top_products(
            self,
            df_product,
            unique_product_key_column,
            unique_product_value_column,
            top_x,
    ):
        cols_keep = [unique_product_key_column, unique_product_value_column]
        df_top_products_agg = df_product[cols_keep].groupby(
            by = [unique_product_key_column],
            as_index = False
        ).sum()
        df_top_products_agg = df_top_products_agg.sort_values(
            by = [unique_product_value_column],
            ascending = False
        ).reset_index(drop=True)
        Log.debugdebug(df_top_products_agg)
        unique_product_list = list(pd.unique(df_top_products_agg[unique_product_key_column]))

        top_n_final = min(len(unique_product_list), top_x)
        df_top_products = df_top_products_agg[0:top_n_final]

        unique_top_products_by_order = df_top_products[unique_product_key_column].to_list()
        unique_remaining_products = [p for p in unique_product_list if p not in unique_top_products_by_order]
        assert len(unique_top_products_by_order) + len(unique_remaining_products) == len(unique_product_list)
        Log.important(
            str(self.__class__) + ' ' + str(getframeinfo(currentframe()).lineno)
            + ': Keeping only top ' + str(top_x) + ' products (discard ' + str(len(unique_remaining_products))
            + ') as attributes: ' + str(unique_top_products_by_order)
            + ', remaining: ' + str(unique_remaining_products)
        )
        return unique_top_products_by_order, unique_remaining_products

    def aggregate_products(
            self,
            df_product,
            unique_human_key_columns,
            unique_product_key_column,
            unique_product_value_column,
            transform_prd_values_method,
    ):
        Log.debugdebug(df_product)
        columns_keep = unique_human_key_columns + [unique_product_key_column, unique_product_value_column]
        for c in columns_keep:
            assert c in df_product.columns, 'Column "' + str(c) + '" must be in ' + str(df_product.columns)
        Log.debug(
            str(self.__class__) + ' ' + str(getframeinfo(currentframe()).lineno)
            + ': Keeping columns: ' + str(columns_keep)
        )
        df_prd_agg = df_product[columns_keep]
        assert list(df_prd_agg.columns) == columns_keep
        Log.debugdebug(
            str(self.__class__) + ' ' + str(getframeinfo(currentframe()).lineno)
            + ': After filtering by columns: ' + str(columns_keep)
        )
        Log.debugdebug(df_prd_agg)

        Log.important(
            str(self.__class__) + ' ' + str(getframeinfo(currentframe()).lineno)
            + ': Transform product values by "' + str(transform_prd_values_method) + '"'
        )
        if transform_prd_values_method == self.TRANSFORM_PRD_VALUES_METHOD_UNITY:
            df_prd_agg[unique_product_value_column] = 1.0
        if transform_prd_values_method == self.TRANSFORM_PRD_VALUES_METHOD_LOG:
            df_prd_agg[unique_product_value_column] = np.log(1 + df_prd_agg[unique_product_value_column])

        """Суммировать продукты по покупателям, те. каждая пара (покупатель, продукт) только в одной строке"""
        df_prd_agg = df_prd_agg.groupby(
            by=unique_human_key_columns + [unique_product_key_column],
            as_index=False,
        ).sum()
        Log.debugdebug(
            str(self.__class__) + ' ' + str(getframeinfo(currentframe()).lineno)
            + ': After grouping by human/product columns: '
        )
        Log.debugdebug(df_prd_agg)

        """Уникальные продукты"""
        unique_product_list = list(np.unique(df_prd_agg[unique_product_key_column]))
        Log.important(
            str(self.__class__) + ' ' + str(getframeinfo(currentframe()).lineno)
            + ': Total unique products as attributes = ' + str(len(unique_product_list))
            + '. Products: ' + str(unique_product_list)
        )
        Log.debugdebug(unique_product_list)

        unique_human_list = list(np.unique(df_prd_agg[unique_human_key_columns]))
        Log.important(
            str(self.__class__) + ' ' + str(getframeinfo(currentframe()).lineno)
            + ': Total unique humans = ' + str(len(unique_human_list))
            + '. Humans: ' + str(unique_human_list)
        )

        return df_prd_agg, unique_product_list, unique_human_list


class SuggestDataProfileUnitTest:
    def __init__(self, ut_params=None):
        self.ut_params = ut_params
        self.res_final = ResultObj()
        self.recommend_data_profile = SuggestDataProfile()

    def run_unit_test(self):
        df_pokupki = pd.DataFrame({
            'client': ['a', 'a', 'b', 'b', 'c', 'c'],
            'product': ['borjomi', 'karspatskaya', 'borjomi', 'morshinskaya', 'bonaqua', 'morshinskaya'],
            'quantity': [1,1,2,1,1,3]
        })
        df_profiles_expected = pd.DataFrame({
            'client':       ['a', 'b', 'c'],
            'bonaqua':      [0., 0., 1.],
            'borjomi':      [1., 2., 0.],
            'karspatskaya': [1., 0., 0.],
            'morshinskaya': [0., 1., 3.],
        })
        df_client_profiles, product_attributes_list = self.recommend_data_profile.convert_product_to_attributes(
            df_product                  = df_pokupki,
            unique_human_key_columns    = ['client'],
            unique_product_key_column   = 'product',
            unique_product_value_column = 'quantity',
            max_attribute_columns       = 0,
            transform_prd_values_method = SuggestDataProfile.TRANSFORM_PRD_VALUES_METHOD_NONE,
        )
        Log.debug('Client profiles')
        Log.debug(df_client_profiles)
        Log.debug('Product as attributes list')
        Log.debug(product_attributes_list)

        self.res_final.update_bool(res_bool=UnitTest.assert_true(
            observed = str(list(df_client_profiles.columns)),
            expected = str(list(df_profiles_expected.columns)),
            test_comment = 'Check columns are correct'
        ))
        for col in df_profiles_expected.columns:
            obs = str(list(df_client_profiles[col]))
            exp = str(list(df_profiles_expected[col]))
            self.res_final.update_bool(res_bool=UnitTest.assert_true(
                observed = obs,
                expected = exp,
                test_comment = 'Check column "' + str(col) + '" values ' + str(obs)
            ))

        return self.res_final


if __name__ == '__main__':
    Log.LOGLEVEL = Log.LOG_LEVEL_DEBUG_1
    res = SuggestDataProfileUnitTest().run_unit_test()
    exit(res.count_fail)
