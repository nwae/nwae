# -*- coding: utf-8 -*-

from nwae.utils.Log import Log
from inspect import getframeinfo, currentframe
import numpy as np
import pandas as pd
from nwae.utils.UnitTest import UnitTest, ResultObj
from nwae.math.suggest.SuggestDataProfile import SuggestDataProfile
from nwae.math.suggest.SuggestMetric import SuggestMetric


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
            if prd_sentence is None:
                return None
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
            # Не существующий код "***"
            [SuggestMetric.METRIC_EUCLIDEAN, '', ['how', 'crypto', 'deposit', 'method']],
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
                prd_sentence     = equivalent_products.get(sent),
                col_product_name = None,
            )
            if ref_dna is not None:
                ref_dna = np.array(list(ref_dna.values()))
            recommendations = self.recommend_metric.recommend_products(
                obj_ref_dna    = ref_dna,
                df_product_dna = df_product,
                unique_prdname_cols = ['__product'],
                how_many       = 4,
                metric         = metric,
                force_normalization = (metric == SuggestMetric.METRIC_COSINE),
                replace_purchased_product_with_nan = False,
            )
            # Обратный формат всегда 2-мерный [[1,2,3]], [[1,2,3],[4,5,6]], ..
            if ( ref_dna is None ) or ( len(ref_dna.shape) == 1 ):
                expected_recommendations_mod = [expected_recommendations]
            else:
                expected_recommendations_mod = expected_recommendations
            self.res_final.update_bool(res_bool=UnitTest.assert_true(
                observed     = recommendations,
                expected     = expected_recommendations_mod,
                test_comment = 'Recomendations metric "' + str(metric) + '" for "' + str(sent)
                               + '" ' + str(recommendations) + ' expect ' + str(expected_recommendations_mod)
            ))

        return

    def __test_water(self):
        """До того что сможет кодировать атрибуты, необходимо превращать форматы"""
        df_pokupki = pd.DataFrame({
            'client': ['a', 'a', 'b', 'b', 'c', 'c', 'd', 'd', 'e', 'e', 'f', 'f'],
            'product': ['borjomi', 'karspatskaya', 'borjomi', 'karspatskaya', 'illy', 'bonaqua', 'illy', 'lavazza', 'bonaqua', 'lavazza', 'lavazza', 'xxx'],
            'quantity': [1, 1, 2, 1, 1, 1, 2, 1, 2, 1, 2, 0]
        })
        df_client_profiles, product_attributes_list = self.recommend_data_profile.convert_product_to_attributes(
            df_product                  = df_pokupki,
            unique_human_key_columns    = ['client'],
            unique_product_key_column   = 'product',
            unique_product_value_column = 'quantity',
            max_attribute_columns       = 100,
            filter_out_quantile         = 0.01,
            transform_prd_values_method = SuggestDataProfile.TRANSFORM_PRD_VALUES_METHOD_NONE,
            add_unknown_product         = False,
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

        nanprd = SuggestDataProfile.NAN_PRODUCT
        othersprd = SuggestDataProfile.COLNAME_PRODUCTS_NOT_INCLUDED
        filteredprd = SuggestDataProfile.FILTERED_OUT_PRODUCT

        res_test = UnitTest.assert_true(
            observed = product_attributes_list,
            expected = ['lavazza', 'bonaqua', 'borjomi', 'illy', 'karspatskaya', othersprd],
            test_comment = 'attribute list ' + str(product_attributes_list)
        )
        self.res_final.update_bool(res_bool=res_test)
        if not res_test:
            raise Exception('wrong')

        for prds_expected in [
            [
                ['borjomi'],
                [[nanprd, 'karspatskaya', othersprd, 'illy', 'bonaqua', 'lavazza']],
            ],
            [
                ['borjomi', 'lavazza'],
                [
                    [nanprd, 'karspatskaya', othersprd, 'illy', 'bonaqua', 'lavazza'],
                    [nanprd, 'illy', 'bonaqua', othersprd, 'karspatskaya', 'borjomi']
                ],
            ],
        ]:
            prd_names_list = prds_expected[0]
            expected_recs_list = prds_expected[1]
            recs = self.recommend_metric.recommend_products_by_product_names_only(
                product_names_list = prd_names_list,
                df_product_dna = df_mapped_product,
                unique_prdname_cols = ['product'],
                replace_purchased_product_with_nan = True
            )
            res_test =  UnitTest.assert_true(
                observed = str(recs),
                expected = str(expected_recs_list),
                test_comment = 'test ' + str(prd_names_list)
            )
            self.res_final.update_bool(res_bool=res_test)
            if not res_test:
                raise Exception('wrong')

        none_vec = None
        x_vec = np.array([1, 0, 0, 0, 0, 0])
        y_vec = np.array([0, 1, 0, 0, 0, 0])
        z_vec = np.array([0, 0, 1, 0, 0, 0])
        none_vec_expected_rec    = ['lavazza', 'bonaqua', 'borjomi', 'illy', 'karspatskaya', othersprd]
        x_expected_rec_euclidean = ['lavazza', 'bonaqua', 'illy', 'borjomi', 'karspatskaya', othersprd]
        x_expected_rec_cosine    = ['lavazza', 'illy', 'bonaqua', 'karspatskaya', 'borjomi', othersprd]
        y_expected_rec_euclidean = ['bonaqua', 'lavazza', 'illy', 'borjomi', 'karspatskaya', othersprd]
        y_expected_rec_cosine    = ['bonaqua', 'lavazza', 'illy', 'karspatskaya', 'borjomi', othersprd]
        z_expected_rec_euclidean = ['borjomi', 'karspatskaya', 'lavazza', 'bonaqua', 'illy', othersprd]
        z_expected_rec_cosine    = ['karspatskaya', 'borjomi', 'lavazza', 'illy', 'bonaqua', othersprd]

        vecs_all = np.array([x_vec, y_vec, z_vec])
        expected_cosine_all = [x_expected_rec_cosine, y_expected_rec_cosine, z_expected_rec_cosine]
        expected_euclidean_all = [x_expected_rec_euclidean, y_expected_rec_euclidean, z_expected_rec_euclidean]

        metric_vec_expected = [
            [SuggestMetric.METRIC_EUCLIDEAN, none_vec, none_vec_expected_rec],
            [SuggestMetric.METRIC_EUCLIDEAN, x_vec, x_expected_rec_euclidean],
            [SuggestMetric.METRIC_EUCLIDEAN, y_vec, y_expected_rec_euclidean],
            [SuggestMetric.METRIC_EUCLIDEAN, z_vec, z_expected_rec_euclidean],
            [SuggestMetric.METRIC_COSINE, none_vec, none_vec_expected_rec],
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
                replace_purchased_product_with_nan = False,
                include_products_not_in_attributes = False,
            )
            # Обратный формат всегда 2-мерный [[1,2,3]], [[1,2,3],[4,5,6]], ..
            if ( vec is None ) or ( len(vec.shape) == 1 ):
                expected_recommendations_mod = [expected_recommendations]
            else:
                expected_recommendations_mod = expected_recommendations
            res_test = UnitTest.assert_true(
                observed     = recommendations,
                expected     = expected_recommendations_mod,
                test_comment = 'Inclusive recomendations metric "' + str(metric) + '" for "' + str(vec)
                               + '" ' + str(recommendations) + ' expect ' + str(expected_recommendations)
            )
            self.res_final.update_bool(res_bool=res_test)
            if not res_test:
                raise Exception('wrong')

            # TEST INCLUSIVE WITH REPLACE WITH NAN_PRODUCT
            recommendations = self.recommend_metric.recommend_products(
                obj_ref_dna    = vec,
                df_product_dna = df_mapped_product,
                unique_prdname_cols = ['product'],
                metric = metric,
                force_normalization = (metric == SuggestMetric.METRIC_COSINE),
                replace_purchased_product_with_nan = True,
                include_products_not_in_attributes = False,
            )
            if vec is None:
                pass
            elif vec.shape != vecs_all.shape:
                purchased_before = np.array(product_attributes_list)[vec > 0]
                np_exp_recmd = np.array(expected_recommendations)
                replace_x = [(r in purchased_before) for r in np_exp_recmd]
                np_exp_recmd[replace_x] = nanprd
                expected_recommendations = list(np_exp_recmd)
            else:
                np_exp_recmd = np.array(expected_recommendations)
                for i in range(len(np_exp_recmd)):
                    purchased_before = np.array(product_attributes_list)[vecs_all[i] > 0]
                    replace_x = [(r in purchased_before) for r in np_exp_recmd[i]]
                    np_exp_recmd[i][replace_x] = nanprd
                expected_recommendations = [list(row) for row in list(np_exp_recmd)]

            # Обратный формат всегда 2-мерный [[1,2,3]], [[1,2,3],[4,5,6]], ..
            if (vec is None) or ( len(vec.shape) == 1 ):
                expected_recommendations_mod = [expected_recommendations]
            else:
                expected_recommendations_mod = expected_recommendations
            res_test = UnitTest.assert_true(
                observed     = recommendations,
                expected     = expected_recommendations_mod,
                test_comment = 'Inclusive (replaced) recomendations metric "' + str(metric) + '" for "' + str(vec)
                               + '" ' + str(recommendations) + ' expect ' + str(expected_recommendations)
            )
            self.res_final.update_bool(res_bool=res_test)
            if not res_test:
                raise Exception('wrong')


if __name__ == '__main__':
    Log.DEBUG_PRINT_ALL_TO_SCREEN = 1
    Log.LOGLEVEL = Log.LOG_LEVEL_DEBUG_1

    res = SuggestMetricUnitTest().run_unit_test()
    print('PASSED ' + str(res.count_ok) + ', FAILED ' + str(res.count_fail))
    exit(res.count_fail)
