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

    # предназначен для замены продуктов, которые отфильтрованы из топ-продуктов
    COLNAME_PRODUCTS_NOT_INCLUDED = '__others_filtered_out_products'
    # предназначен для замены продуктов, уже купивщих покупателя во время рекомендации
    NAN_PRODUCT                   = '***'
    # отфильтрованные товары все еще оставаются в данных, и во время предложения,
    # есть выбор чтобы заменить их с этим значением
    FILTERED_OUT_PRODUCT          = '__filtered_out_product'

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
            # уменшить количество атрибутов
            max_attribute_columns = 0,
            # по проценту, убирать продуктов меньше такого квартиля
            filter_out_quantile_byvalue = 0.0,
            filter_out_quantile_bycount = 0.0,
            # Before any processing
            transform_prd_values_method = TRANSFORM_PRD_VALUES_METHOD_NONE,
            transform_logbase           = 10.0,
            # осторожно здесь, этот неизвестный продукт будет присвоен 0-вектор (возможно),
            # поэтому если использовать метрику "euclidean", он будет "близок" другим векторам
            add_unknown_product = False,
    ):
        df_prd_agg, unique_product_list, unique_human_list = self.aggregate_products(
            df_product                  = df_product,
            unique_human_key_columns    = unique_human_key_columns,
            unique_product_key_column   = unique_product_key_column,
            unique_product_value_column = unique_product_value_column,
            transform_prd_values_method = transform_prd_values_method,
            transform_logbase           = transform_logbase,
        )

        #
        # Этот шаг приведет к несколько проблемам, так как у некоторых клиентов будет превратить к вектору нулей
        #
        is_reduced = False
        unique_remaining_products = None
        np_remaining_attributes = None
        if (max_attribute_columns > 0) | (filter_out_quantile_byvalue > 0.0) | (filter_out_quantile_bycount > 0.0):
            byval_unique_top_products_by_order, byval_unique_remaining_products = self.find_top_products(
                df_product                  = df_product,
                unique_product_key_column   = unique_product_key_column,
                unique_product_value_column = unique_product_value_column,
                # Do max filtering later
                top_x                       = 0,
                filter_out_quantile         = filter_out_quantile_byvalue,
                aggregate_method            = 'sum',
            )
            bycnt_unique_top_products_by_order, bycnt_unique_remaining_products = self.find_top_products(
                df_product                  = df_product,
                unique_product_key_column   = unique_product_key_column,
                unique_product_value_column = unique_product_value_column,
                # Do max filtering later
                top_x                       = 0,
                filter_out_quantile         = filter_out_quantile_bycount,
                aggregate_method            = 'count',
            )
            filtered_out_products_by_count_condition = [
                prd for prd in byval_unique_top_products_by_order
                if prd not in bycnt_unique_top_products_by_order
            ]
            Log.info(
                str(self.__class__) + ' ' + str(getframeinfo(currentframe()).lineno)
                + ': Products filtered out by 2nd count condition ' + str(filtered_out_products_by_count_condition)
            )
            # Убирать товары, не выполняется второе условие
            unique_top_products_by_order = [
                prd for prd in byval_unique_top_products_by_order
                if prd in bycnt_unique_top_products_by_order
            ]
            unique_remaining_products = filtered_out_products_by_count_condition + byval_unique_remaining_products
            assert len(unique_top_products_by_order) + len(unique_remaining_products) == len(unique_product_list), \
                'Length of unique top products and remaining products must equal length of product list'
            Log.info(
                str(self.__class__) + ' ' + str(getframeinfo(currentframe()).lineno)
                + ': Remaining ' + str(len(unique_remaining_products))
                + ' least products: ' + str(unique_remaining_products)
            )
            if max_attribute_columns > 0:
                max_final = min(len(unique_top_products_by_order), max_attribute_columns)
                if max_final < len(unique_top_products_by_order):
                    filtered_out_by_max_attribute_products = unique_top_products_by_order[max_final:]
                    unique_top_products_by_order = unique_top_products_by_order[0:max_final]
                    unique_remaining_products = filtered_out_by_max_attribute_products + unique_remaining_products
                    assert len(unique_top_products_by_order) + len(unique_remaining_products) == len(unique_product_list), \
                        'Length of unique top products and remaining products must equal length of product list'

            # Change the removed product names to one name
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

        # прибавить "неизвестный продукт"
        if add_unknown_product:
            if self.NAN_PRODUCT in unique_product_list:
                raise Exception(
                    str(self.__class__) + ' ' + str(getframeinfo(currentframe()).lineno)
                    + ': Name clash with nan product name "' + str(self.NAN_PRODUCT) + '"'
                )
            unique_product_list.append(self.NAN_PRODUCT)

        columns_running = list(df_converted.columns)
        """Продукт за продуктом, создавать столбец продукта"""
        for prd in unique_product_list:
            condition_only_this_product = df_prd_agg[unique_product_key_column] == prd
            df_prd_agg_part = df_prd_agg[condition_only_this_product]
            if len(df_prd_agg_part) == 0:
                Log.warning(
                    str(self.__class__) + ' ' + str(getframeinfo(currentframe()).lineno)
                    + ': For product "' + str(prd) + '", there are 0 sales, adding 0 column'
                )
                df_converted[prd] = 0.0
            else:
                columns_keep = unique_human_key_columns + [unique_product_value_column]
                df_prd_agg_part = df_prd_agg_part[columns_keep].reset_index(drop=True)
                Log.debugdebug(prd)
                Log.debugdebug(df_prd_agg_part)
                """Соединить цену/количество продукта с человеком"""
                df_converted = df_converted.merge(
                    df_prd_agg_part,
                    on  = unique_human_key_columns,
                    how = 'left'
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
            df                = df_converted,
            name_columns      = unique_human_key_columns,
            attribute_columns = attr_cols,
            normalize_method  = normalize_method,
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
            top_x = 0,
            # по проценту, убирать продуктов меньше такого квартиля
            filter_out_quantile = 0.0,
            aggregate_method = 'sum',
    ):
        # Агрегирование и сортировка
        cols_keep = [unique_product_key_column, unique_product_value_column]
        if aggregate_method == 'sum':
            df_top_products_agg = df_product[cols_keep].groupby(
                by = [unique_product_key_column],
                as_index = False
            ).sum()
            # Nothing to do. The aggregated column names will not change
        elif aggregate_method == 'count':
            df_top_products_agg = df_product[cols_keep].groupby(
                by = [unique_product_key_column],
                as_index = False
            ).size()
            # !! The aggregated column name will change
            df_top_products_agg.columns = cols_keep
        else:
            raise Exception(
                str(self.__class__) + ' ' + str(getframeinfo(currentframe()).lineno)
                + ': No such aggregate method "' + str(aggregate_method) + '"!'
            )

        df_top_products_agg = df_top_products_agg.sort_values(
            by = [unique_product_value_column],
            ascending = False
        ).reset_index(drop=True)
        Log.debugdebug(df_top_products_agg)
        unique_product_list = list(pd.unique(df_top_products_agg[unique_product_key_column]))

        # Квартили
        if filter_out_quantile > 0.0:
            assert filter_out_quantile <= 1.0
            q = [0.1, 0.25, 0.5, 0.75, 0.9, 0.95]
            quantile_values = np.quantile(df_top_products_agg[unique_product_value_column], q=q)
            quantile_filter_value = np.quantile(df_top_products_agg[unique_product_value_column], q=filter_out_quantile)
            Log.important(
                str(self.__class__) + ' ' + str(getframeinfo(currentframe()).lineno)
                + ': Aggregate by "' + str(aggregate_method) + '" quantile at ' + str(filter_out_quantile)
                + ' = ' + str(quantile_filter_value)
                + ' Other quantile values at ' + str(q) + ' ' + str(quantile_values)
            )
            condition_quantile = df_top_products_agg[unique_product_value_column] >= quantile_filter_value
            df_top_products_agg_quantile = df_top_products_agg[condition_quantile]
            Log.important(
                str(self.__class__) + ' ' + str(getframeinfo(currentframe()).lineno)
                + ': After filter out aggregate by "' + str(aggregate_method)
                + '" quantile at ' + str(filter_out_quantile) + ' = ' + str(quantile_filter_value)
                + ', remain ' + str(len(df_top_products_agg_quantile)) + ' from ' + str(len(df_top_products_agg))
                + ' products'
            )
        else:
            df_top_products_agg_quantile = df_top_products_agg

        if top_x > 0:
            top_n_final = min(len(df_top_products_agg_quantile), top_x)
            df_top_products = df_top_products_agg_quantile[0:top_n_final]
            Log.important(
                str(self.__class__) + ' ' + str(getframeinfo(currentframe()).lineno)
                + ': After getting final top ' + str(top_n_final) + ' aggregated by "' + str(aggregate_method)
                + '", quantile at ' + str(filter_out_quantile) + ', remain ' + str(len(df_top_products))
                + ' from ' + str(len(df_top_products_agg)) + ' products'
            )
        else:
            df_top_products = df_top_products_agg_quantile

        unique_top_products_by_order = df_top_products[unique_product_key_column].to_list()
        unique_remaining_products = [p for p in unique_product_list if p not in unique_top_products_by_order]
        assert len(unique_top_products_by_order) + len(unique_remaining_products) == len(unique_product_list)
        Log.important(
            str(self.__class__) + ' ' + str(getframeinfo(currentframe()).lineno)
            + ': Aggregated by "' + str(aggregate_method) + '", quantile at ' + str(filter_out_quantile)
            + ', keeping only top ' + str(top_x)
            + ' (if 0 means keep all) products (discard ' + str(len(unique_remaining_products))
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
            transform_logbase,
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
            df_prd_agg[unique_product_value_column] = np.log(1 + df_prd_agg[unique_product_value_column]) / np.log(transform_logbase)

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
            'client': ['a', 'a', 'b', 'b', 'c', 'c', 'c'],
            'product': ['borjomi', 'karspatskaya', 'borjomi', 'morshinskaya', 'bonaqua', 'morshinskaya', 'xxx'],
            'quantity': [1, 1, 2, 1, 1, 3, 0]
        })
        # в случае max_attribute_columns>0, будет упорядочить колонки от наибольшего к наименее популярному продукту
        df_profiles_expected = pd.DataFrame({
            'client':       ['a', 'b', 'c'],
            'morshinskaya': [0., 1., 3.],
            'borjomi':      [1., 2., 0.],
            'bonaqua':      [0., 0., 1.],
            'karspatskaya': [1., 0., 0.],
            SuggestDataProfile.COLNAME_PRODUCTS_NOT_INCLUDED: [0., 0., 0.],
        })
        df_client_profiles, product_attributes_list = self.recommend_data_profile.convert_product_to_attributes(
            df_product                  = df_pokupki,
            unique_human_key_columns    = ['client'],
            unique_product_key_column   = 'product',
            unique_product_value_column = 'quantity',
            max_attribute_columns       = 100,
            filter_out_quantile_byvalue = 0.1,
            filter_out_quantile_bycount = 0.1,
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
    print('PASS ' + str(res.count_ok) + ' FAIL ' + str(res.count_fail))
    exit(res.count_fail)
