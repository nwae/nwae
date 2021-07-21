# -*- coding: utf-8 -*-

import pandas as pd
from nwae.lang.LangFeatures import LangFeatures
from nwae.lang.preprocessing.BasicPreprocessor import BasicPreprocessor


class SampleTextClassificationData:

    COL_CLASS_ID = 'classId'
    COL_CLASS = 'class'
    COL_CLASS_NAME = 'className'
    COL_TEXT = 'text'
    COL_TEXT_ID = 'textId'
    # Segmented text
    COL_TEXT_SEG = 'textSegmented'

    TYPE_LANG_MAIN = 'lang_main'
    TYPE_LANG_ADDITIONAL = 'lang_additional'
    TYPE_IO_IN = 'in'
    TYPE_IO_OUT = 'out'

    # The class text
    SAMPLE_TRAINING_DATA = [
        {
            TYPE_LANG_MAIN: LangFeatures.LANG_KO,
            TYPE_LANG_ADDITIONAL: (),
            # Class/Intent ID, Class Name/Intent Name, Text
            TYPE_IO_IN: (
                (1, '하나', '하나 두 두 셋 넷'),
                (1, '하나', '하나 하나 두 셋 셋 넷'),
                (1, '하나', '하나 두 셋 넷'),
                (2, '두', '두 셋 셋 넷'),
                (2, '두', '두 두 셋 셋 넷 넷'),
                (2, '두', '두 두 셋 넷 넷')
            ),
            # Class/Intent ID, Class Name/Intent Name, Text, Text Segmented
            TYPE_IO_OUT: (
                (1, '하나', '하나 두 두 셋 넷', '하나 두 두 셋 넷'),
                (1, '하나', '하나 하나 두 셋 셋 넷', '하나 하나 두 셋 셋 넷'),
                (1, '하나', '하나 두 셋 넷', '하나 두 셋 넷'),
                # Appended intent name from processing
                (1, '하나', '하나', '하나'),
                (2, '두', '두 셋 셋 넷', '두 셋 셋 넷'),
                (2, '두', '두 두 셋 셋 넷 넷', '두 두 셋 셋 넷 넷'),
                (2, '두', '두 두 셋 넷 넷', '두 두 셋 넷 넷'),
                # Appended intent name from processing
                (2, '두', '두', '두'),
            )
        },
        {
            TYPE_LANG_MAIN: LangFeatures.LANG_VI,
            TYPE_LANG_ADDITIONAL: (),
            # Class/Intent ID, Class Name/Intent Name, Text
            TYPE_IO_IN: (
                (1, 'rút tiền', 'giới hạn rút tiền'),
                (1, 'rút tiền', 'rút bao nhiêu'),
                (1, 'rút tiền', 'trạng thái lệnh rút tiền')
            ),
            # Class/Intent ID, Class Name/Intent Name, Text, Text Segmented
            TYPE_IO_OUT: (
                (1, 'rút tiền', 'giới hạn rút tiền', 'giới hạn--||--rút tiền'),
                (1, 'rút tiền', 'giới hạn rút tiền', 'gioi han--||--rut tien'), # Appended latin form
                (1, 'rút tiền', 'rút bao nhiêu', 'rút--||--bao nhiêu'),
                (1, 'rút tiền', 'rút bao nhiêu', 'rut--||--bao nhieu'), # Appended latin form
                (1, 'rút tiền', 'trạng thái lệnh rút tiền', 'trạng thái--||--lệnh--||--rút tiền'),
                (1, 'rút tiền', 'trạng thái lệnh rút tiền', 'trang thai--||--lenh--||--rut tien'),  # Appended latin form
                # Appended intent name from processing
                (1, 'rút tiền', 'rút tiền', 'rút tiền'),
                (1, 'rút tiền', 'rút tiền', 'rut tien'),    # Appended latin form
            )
        },
        {
            TYPE_LANG_MAIN: LangFeatures.LANG_TH,
            TYPE_LANG_ADDITIONAL: (),
            # Class/Intent ID, Class Name/Intent Name, Text
            TYPE_IO_IN: (
                (1, 'รัก', 'ทำไมน่ารักขนาดนี้'),
                (1, 'รัก', 'น่ารักกกกก'),
                (1, 'รัก', 'สวยจัง 10/10.'),
                (1, 'รัก', 'สวยที่สุด'),
                (1, 'รัก', 'ชมน้องจังเลย'),
                (2, 'บัญชี', 'เปลี่ยนเบอร์ทำไง'),
                (2, 'บัญชี', 'อัพเดตประวัติส่วนตัวยังไง?'),
                (2, 'บัญชี', 'เพิ่มข้อมูลส่วนตัวตรงไหน'),
            ),
            # Class/Intent ID, Class Name/Intent Name, Text, Text Segmented
            TYPE_IO_OUT: (
                (1, 'รัก', 'ทำไมน่ารักขนาดนี้', 'ทำไม น่า รัก ขนาด นี้'),
                (1, 'รัก', 'น่ารักกกกก', 'น่า รัก กก กก'),
                (1, 'รัก', 'สวยจัง 10/10', 'สวย จัง ' + BasicPreprocessor.W_NUM + ' / ' + BasicPreprocessor.W_NUM +  ' .'),
                (1, 'รัก', 'สวยที่สุด', 'สวย ที่ สุด'),
                (1, 'รัก', 'ชมน้องจังเลย', 'ชม น้อง จัง เลย'),
                # Appended intent name from processing
                (1, 'รัก', 'รัก', 'รัก'),
                (2, 'บัญชี', 'เปลี่ยนเบอร์ทำไง', 'เปลี่ยน เบอร์ ทำ ไง'),
                (2, 'บัญชี', 'อัพเดตประวัติส่วนตัวยังไง?', 'อัพเดต ประวัติ ส่วน ตัว ยังไง ?'),
                (2, 'บัญชี', 'เพิ่มข้อมูลส่วนตัวตรงไหน', 'เพิ่ม ข้อ มูล ส่วน ตัว ตรง ไหน'),
                # Appended intent name from processing
                (2, 'บัญชี', 'บัญชี', 'บัญชี'),
            )
        },
        {
            TYPE_LANG_MAIN: LangFeatures.LANG_ZH,
            TYPE_LANG_ADDITIONAL: (),
            # Class/Intent ID, Class Name/Intent Name, Text
            TYPE_IO_IN: (
                (1, '登录', '登录次数多了，怎么办'),
                (2, '忘记', '我忘记账号'),
                (2, '忘记', '我忘记我的名称了'),
                (2, '忘记', '我记不到密码了'),
            ),
            # Class/Intent ID, Class Name/Intent Name, Text, Text Segmented
            TYPE_IO_OUT: (
                (1, '登录', '登录次数多了，怎么办', '登录 次数 多 了 ， 怎么办'),
                # Appended intent name from processing
                (1, '登录', '登录', '登录'),
                (2, '忘记', '我忘记账号', '我 忘记 账号'),
                (2, '忘记', '我忘记我的名称了', '我 忘记 我 的 名称 了'),
                (2, '忘记', '我记不到密码了', '我 记 不到 密码 了'),
                # Appended intent name from processing
                (2, '忘记', '忘记', '忘记'),
            )
        },
        {
            TYPE_LANG_MAIN: LangFeatures.LANG_ZH,
            TYPE_LANG_ADDITIONAL: (LangFeatures.LANG_TH, LangFeatures.LANG_EN, LangFeatures.LANG_VI),
            # Class/Intent ID, Class Name/Intent Name, Text
            TYPE_IO_IN: (
                # Chinese
                (1, '登录', '登录次数多了，怎么办'),
                (2, '忘记', '我忘记账号'),
                (2, '忘记', '我忘记我的名称了'),
                (2, '忘记', '我记不到密码了'),
                # Thai
                (3, 'รัก', 'ทำไมน่ารักขนาดนี้'),
                (3, 'รัก', 'น่ารักกกกก'),
                (3, 'รัก', 'สวยจัง 10/10.'),
                (3, 'รัก', 'สวยที่สุด'),
                (3, 'รัก', 'ชมน้องจังเลย'),
                (4, 'บัญชี', 'เปลี่ยนเบอร์ทำไง'),
                (4, 'บัญชี', 'อัพเดตประวัติส่วนตัวยังไง?'),
                (4, 'บัญชี', 'เพิ่มข้อมูลส่วนตัวตรงไหน'),
                # Vietnamese
                (5, 'rút tiền', 'giới hạn rút tiền'),
                (5, 'rút tiền', 'rút bao nhiêu'),
                (5, 'rút tiền', 'trạng thái lệnh rút tiền')
            ),
            # Class/Intent ID, Class Name/Intent Name, Text, Text Segmented
            TYPE_IO_OUT: (
                (1, '登录', '登录次数多了，怎么办', '登录 次数 多 了 ， 怎么办'),
                # Appended intent name from processing
                (1, '登录', '登录', '登录'),
                (2, '忘记', '我忘记账号', '我 忘记 账号'),
                (2, '忘记', '我忘记我的名称了', '我 忘记 我 的 名称 了'),
                (2, '忘记', '我记不到密码了', '我 记 不到 密码 了'),
                # Appended intent name from processing
                (2, '忘记', '忘记', '忘记'),
                (3, 'รัก', 'ทำไมน่ารักขนาดนี้', 'ทำไม น่า รัก ขนาด นี้'),
                (3, 'รัก', 'น่ารักกกกก', 'น่า รัก กก กก'),
                (3, 'รัก', 'สวยจัง 10/10', 'สวย จัง ' + BasicPreprocessor.W_NUM + ' / ' + BasicPreprocessor.W_NUM + ' .'),
                (3, 'รัก', 'สวยที่สุด', 'สวย ที่ สุด'),
                (3, 'รัก', 'ชมน้องจังเลย', 'ชม น้อง จัง เลย'),
                # Appended intent name from processing
                (3, 'รัก', 'รัก', 'รัก'),
                (4, 'บัญชี', 'เปลี่ยนเบอร์ทำไง', 'เปลี่ยน เบอร์ ทำ ไง'),
                (4, 'บัญชี', 'อัพเดตประวัติส่วนตัวยังไง?', 'อัพเดต ประวัติ ส่วน ตัว ยังไง ?'),
                (4, 'บัญชี', 'เพิ่มข้อมูลส่วนตัวตรงไหน', 'เพิ่ม ข้อ มูล ส่วน ตัว ตรง ไหน'),
                # Appended intent name from processing
                (4, 'บัญชี', 'บัญชี', 'บัญชี'),
                (5, 'rút tiền', 'giới hạn rút tiền', 'giới hạn--||--rút tiền'),
                (5, 'rút tiền', 'rút bao nhiêu', 'rút--||--bao nhiêu'),
                (5, 'rút tiền', 'trạng thái lệnh rút tiền', 'trạng thái--||--lệnh--||--rút tiền'),
                # Appended intent name from processing
                (5, 'rút tiền', 'rút tiền', 'rút tiền'),
                # No Appended latin equivalent forms, as Vietnamese not the main language
            )
        },
    ]

    @staticmethod
    def get_lang_main(
            sample_training_data
    ):
        return sample_training_data[SampleTextClassificationData.TYPE_LANG_MAIN]

    @staticmethod
    def get_lang_additional(
            sample_training_data
    ):
        return list(sample_training_data[SampleTextClassificationData.TYPE_LANG_ADDITIONAL])

    @staticmethod
    def get_text_classification_training_data(
            sample_training_data,
            type_io = TYPE_IO_IN
    ):
        class_arr = [y_x[0] for y_x in sample_training_data[type_io]]
        class_name_arr = [y_x[1] for y_x in sample_training_data[type_io]]
        texts_arr = [y_x[2] for y_x in sample_training_data[type_io]]
        texts_seg_arr = None
        if type_io == 'out':
            texts_seg_arr = [y_x[3] for y_x in sample_training_data[type_io]]

        return {
            SampleTextClassificationData.COL_CLASS_ID: range(1000,1000+len(class_arr),1),
            SampleTextClassificationData.COL_CLASS: class_arr,
            SampleTextClassificationData.COL_CLASS_NAME: class_name_arr,
            SampleTextClassificationData.COL_TEXT: texts_arr,
            SampleTextClassificationData.COL_TEXT_ID: range(2000,2000+len(class_arr),1),
            SampleTextClassificationData.COL_TEXT_SEG: texts_seg_arr
        }


if __name__ == '__main__':
    data = SampleTextClassificationData.get_text_classification_training_data(
        sample_training_data = SampleTextClassificationData.SAMPLE_TRAINING_DATA[4]
    )
    df = pd.DataFrame(data)
    print(data)
    print(df)
