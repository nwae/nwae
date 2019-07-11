# -*- coding: utf-8 -*-

import re
import ie.app.ConfigFile as cf
import pandas as pd
import mozg.common.util.CommandLine as cmdline
import mozg.common.util.Profiling as pf
import mozg.common.bot.BotIntentAnswer as botiat
import ie.lib.chat.classification.training.ChatTrainingData as ctd
import mozg.common.data.BasicData as bd
import mozg.common.data.Campaign as campaign
import mozg.common.data.IntentCategory as inttcat
import mozg.common.data.Intent as intt
import mozg.common.data.IntentAnswer as inttans
import mozg.common.data.IntentTraining as intttr
import mozg.common.data.security.Auth as au
import mozg.common.util.Log as lg
import mozg.common.util.StringUtils as su
import json
import mozg.common.bot.protocol.bi.Message as bimsg


class BotDataImport:

    def __init__(
            self,
            db_profile,
            accountId,
            botId,
            lang,
            botkey,
            verbose=0
    ):
        self.db_profile = db_profile
        self.accountId = accountId
        self.botId = botId
        self.lang = lang
        self.botkey = botkey
        self.verbose = verbose

        #
        # Create DB objects
        #
        self.dbItCat = inttcat.IntentCategory(
            db_profile = self.db_profile,
            verbose    = self.verbose
        )
        self.dbIt = intt.Intent(
            db_profile = self.db_profile,
            verbose    = self.verbose
        )
        self.dbItAns = inttans.IntentAnswer(
            db_profile = self.db_profile,
            verbose    = self.verbose
        )

        self.bi_message = bimsg.Message(
            db_profile = self.db_profile,
            bot_id     = self.botId,
            verbose    = self.verbose
        )

        return

    def write_intent_categories_to_db(
            self,
            df_training_data_split
    ):
        ics = list(set(df_training_data_split[ctd.ChatTrainingData.COL_TDATA_CATEGORY]))
        ics.sort(reverse=False)

        df_ics = pd.DataFrame(columns=['id', 'fullPath', 'intentCategory', 'parentId'])

        lg.Log.log('Writing default root folder to DB..')
        root_id = self.dbItCat.insert_or_update_intent_category(
            intent_category = inttcat.IntentCategory.ROOT_DIRECTORY_RESERVED_NAME,
            parent_id       = None,
            bot_id          = int(self.botId)
        )

        # countpath = 0
        # Loop through all unique paths
        for full_path in ics:
            full_path = re.sub('[ ]*/[ ]*', '/', full_path)
            lg.Log.log('*** Looping path [' + full_path + ']...')

            # Split 'aaa/bbb/ccc' into ['aaa', 'bbb', 'ccc']
            ic_split = full_path.split('/')

            # Loop from highest ('aaa') to lowest level ('ccc')
            parent_id = root_id
            for i in range(0, len(ic_split), 1):
                ic = ic_split[i]
                ic = su.StringUtils.trim(ic)
                lg.Log.log('   *** Intent Category [' + ic + ']...')

                ic_id = None

                ic_record = df_ics[df_ics['fullPath']=='/'.join(ic_split[0:(i+1)])]

                # Write to DB
                if ic_record.shape[0] == 0:
                    lg.Log.log('      Writing to DB..')
                    ic_id = self.dbItCat.insert_or_update_intent_category(
                        intent_category = ic,
                        parent_id       = parent_id,
                        bot_id          = int(self.botId)
                    )

                    # Record down what was written to DB
                    df_ics = df_ics.append(
                        {
                            'id': ic_id,
                            'fullPath': full_path,
                            'intentCategory': ic,
                            'parentId': parent_id
                        },
                        ignore_index=True
                    )
                    # print(df_ics)

                else:
                    lg.Log.log('      Not writing to DB...')
                    ic_id = int(ic_record['id'])

                # Update parentId for next loop
                parent_id = ic_id

            #countpath = countpath + 1
            #if countpath > 2:
            #    break

        return

    def write_intents_trdata_answers_to_db(
            self,
            df_training_data_split,
            just_print_no_write_to_db = False
    ):
        br = botiat.BotIntentAnswer(
            # Not using DB of course, since we are writing csv data to DB
            use_db                       = False,
            db_profile                   = self.db_profile,
            account_id                   = self.accountId,
            bot_id                       = self.botId,
            bot_lang                     = self.lang,
            botkey                       = self.botkey,
            dirpath                      = cf.ConfigFile.DIR_INTENT_TRAINDATA,
            postfix_intent_answer_trdata = cf.ConfigFile.POSTFIX_INTENT_ANSWER_TRDATA_FILE,
            postfix_trdata               = cf.ConfigFile.POSTFIX_INTENT_TRAINING_FILES
        )

        # Get unique categories from csv
        all_cats_unique = list(set(df_training_data_split[ctd.ChatTrainingData.COL_TDATA_CATEGORY]))

        lg.Log.log('Writing Intents to DB...')
        for intentCategory_csv in all_cats_unique:

            df_tmp_intents_of_cat = df_training_data_split[
                df_training_data_split[ctd.ChatTrainingData.COL_TDATA_CATEGORY] == intentCategory_csv]

            # Perform similar regex as above when writing to DB, so we can find it
            intentCategoryFullPath = re.sub('[ ]*/[ ]*', '/', intentCategory_csv)
            intentCategoryFullPath = su.StringUtils.trim(intentCategoryFullPath)

            [intentCategoryId, parentId] = self.dbItCat.get_folder_id(
                folder_full_path = intentCategoryFullPath,
                bot_id           = self.botId
            )

            # All unique intents under this Intent category from csv file
            all_intents_unique_csv = list(set(df_tmp_intents_of_cat[ctd.ChatTrainingData.COL_TDATA_INTENT]))

            # Get the intents of this category already in DB
            all_intents_of_cat_in_db = self.dbIt.get(intentCategoryId=intentCategoryId)

            lg.Log.log('Folder=' + intentCategoryFullPath + ' (id ' + str(intentCategoryId), ', parentId ' + str(parentId) + ')')

            # Loop through all intents in the intent category
            for intent_csv in all_intents_unique_csv:
                # All training data of this intent
                df_tmp_intent_td = df_tmp_intents_of_cat[
                    df_tmp_intents_of_cat[ctd.ChatTrainingData.COL_TDATA_INTENT] == intent_csv]
                intent_type_csv = None

                # Get all answers for this intent
                intent_answers = []
                lg.Log.log('   Getting all answers for intent [' + intent_csv + '], cat [' + intentCategory_csv + ']')
                # Get intent answer from csv
                intentId_csv = intentCategory_csv + '/' + intent_csv
                rows_intent_answer = br.get_replies(intent_id=intentId_csv)
                if rows_intent_answer is not None:
                    for k in range(0, rows_intent_answer.shape[0], 1):
                        row = rows_intent_answer.loc[k]

                        weight = row[botiat.BotIntentAnswer.COL_WEIGHT]
                        try:
                            weight = int(weight)
                        except Exception as ex:
                            lg.Log.log(ex)
                            weight = int(1)

                        answer_txt = row[botiat.BotIntentAnswer.COL_ANSWER]

                        #
                        # Do basic processing
                        #

                        answer_txt_xml = answer_txt

                        # Convert to our BI Message format, put a <text> in front and </text> behind
                        m = re.search(pattern='<form', string=answer_txt_xml)
                        if m:
                            #lg.Log.log('Converting [' + answer_txt + '] to XML...')
                            answer_txt_xml = '<' + bimsg.Message.TAG_MESSAGE + '>' + answer_txt_xml + '</' + bimsg.Message.TAG_MESSAGE + '>'
                        else:
                            # lg.Log.log('Converting [' + answer_txt + '] to XML...')
                            # Normal text/intent link type
                            answer_txt_xml = '<' + bimsg.Message.TAG_TEXT + '>' + answer_txt_xml + '</' + bimsg.Message.TAG_TEXT + '>'
                            # Replace \n, \r
                            answer_txt_xml = re.sub(pattern='\n', repl='<br/>', string=answer_txt_xml)
                            answer_txt_xml = re.sub(pattern='\r', repl='', string=answer_txt_xml)
                            # Put <text> tags before/after Intent links
                            answer_txt_xml = re.sub(pattern='<intentlink>', repl='</text><intentlink>', string=answer_txt_xml)
                            answer_txt_xml = re.sub(pattern='</intentlink>', repl='</intentlink><text>', string=answer_txt_xml)
                            # Finally wrap in message tag
                            answer_txt_xml = '<' + bimsg.Message.TAG_MESSAGE + '>' + answer_txt_xml + '</' + bimsg.Message.TAG_MESSAGE + '>'

                        # lg.Log.log('Converted to [' + answer_txt_xml + ']')

                        self.bi_message.xml = answer_txt_xml
                        try:
                            self.bi_message.convert_xml_str_to_obj()

                            if self.bi_message.response is not None:
                                lg.Log.log('Converted [' + self.bi_message.xml + ']')
                                lg.Log.log(self.bi_message.response.get_json_string())
                        except Exception as ex:
                            lg.Log.log(str(self.__class__) + ': Cannot convert [' + answer_txt_xml + '].')
                            self.bi_message.xml = '<message><text>Error in bot data import from CSV</text></message>'
                            self.bi_message.convert_xml_str_to_obj()

                        intent_answers.append({
                            # This column is JSON type
                            # TODO Call the proper function to convert to our/Comm100 JSON format, don't do this
                            inttans.IntentAnswer.COL_ANSWER: self.bi_message.response.get_json_string(),
                            inttans.IntentAnswer.COL_WEIGHT: weight
                        })
                else:
                    lg.Log.log('No answers found for intent [' + intent_csv + '], cat [' + intentCategory_csv + ']')

                # Regex as per DB convention
                intent = su.StringUtils.trim(intent_csv)

                # Get intentId from DB (if exists)
                intentId = None
                for tmp_item in all_intents_of_cat_in_db:
                    if tmp_item[intt.Intent.COL_INTENT_NAME] == intent:
                        intentId = int(tmp_item[intt.Intent.COL_INTENT_ID])
                        break

                intentTrainings = []

                # Get all training data for this intent
                lg.Log.log('   Getting all training data for intent [' + intent + '], cat [' + intentCategoryFullPath + ']')
                df_tmp_intent_td = df_tmp_intent_td.reset_index(drop=True)
                for j in range(0, df_tmp_intent_td.shape[0], 1):
                    row = df_tmp_intent_td.loc[j]

                    trainingData = row[ctd.ChatTrainingData.COL_TDATA_TEXT]
                    trainingData = su.StringUtils.trim(trainingData)

                    trainingDataSegmented = row[ctd.ChatTrainingData.COL_TDATA_TEXT_SEGMENTED]
                    trainingDataSegmented = su.StringUtils.trim(trainingDataSegmented)

                    #lg.Log.log('Line ' + str(j+1) + '.' +
                    #            'Intent Category ID: ' + str(intentCategoryId) +
                    #            ', Intent Category: ' + str(intentCategory) +
                    #            ', Intent: ' + str(intent) +
                    #            ', TrData: ' + str(trainingData) +
                    #            ', TrData Segmented: ' + str(trainingDataSegmented)
                    #            )

                    tmp_td = {
                        intttr.IntentTraining.COL_SENTENCE: trainingData,
                        intttr.IntentTraining.COL_SPLIT_SENTENCE: trainingDataSegmented
                    }
                    intentTrainings.append(tmp_td)

                # If in debug mode, don't do anything to DB
                if just_print_no_write_to_db:
                    continue

                intent_type_csv = br.get_intent_type(intent_id=intentId)
                if intentId is None:
                    lg.Log.log('\t   Inserting intent "' + intent
                               + '", type "' + str(intent_type_csv)
                               + '" of category "' + intentCategoryFullPath
                               + '" to DB...')

                    try:
                        res = self.dbIt.insert_or_update(
                            operation  = bd.BasicData.OPERATION_INSERT,
                            intentName = intent,
                            intentType = intent_type_csv,
                            requireAuthentication = False,
                            notifyAgent = False,
                            intentCategoryId = intentCategoryId,
                            intentTrainings  = intentTrainings,
                            intentAnswers    = intent_answers
                        )
                        if res:
                            lg.Log.important('Successfully inserted.')
                    except Exception as ex:
                        lg.Log.error('Failed to insert: ' + str(ex))
                else:
                    lg.Log.log('\t   Updating intent "' + intent
                               + '", intent id ' + str(intentId)
                               + '", type "' + str(intent_type_csv)
                               + '" of category "' + intentCategoryFullPath
                               + '" to DB...')

                    try:
                        res = self.dbIt.insert_or_update(
                            operation = bd.BasicData.OPERATION_UPDATE,
                            intentName = intent,
                            intentType = intent_type_csv,
                            requireAuthentication = False,
                            notifyAgent = False,
                            intentCategoryId = intentCategoryId,
                            intentTrainings = intentTrainings,
                            intentAnswers = intent_answers,
                            intentId = intentId
                        )
                        if res:
                            lg.Log.important('Successfully updated.')
                    except Exception as ex:
                        lg.Log.error('Failed to update: ' + str(ex))
                # Debug break only, remove in normal run
                #break
            # Debug break, remove in normal run
            #break

    def run(self):
        ctdata = ctd.ChatTrainingData(
            # We are not using DB, since we are importing CSV to DB
            use_db                 = False,
            db_profile             = self.db_profile,
            account_id             = None,
            bot_id                 = None,
            lang                   = self.lang,
            # Brand must be empty, so we process the entire training data
            bot_key                = self.botkey,
            dirpath_traindata      = cf.ConfigFile.DIR_INTENT_TRAINDATA,
            postfix_training_files = cf.ConfigFile.POSTFIX_INTENT_TRAINING_FILES,
            dirpath_wordlist       = cf.ConfigFile.DIR_WORDLIST,
            dirpath_app_wordlist   = cf.ConfigFile.DIR_APP_WORDLIST,
            dirpath_synonymlist    = cf.ConfigFile.DIR_SYNONYMLIST
        )

        df_training_data_split = ctdata.get_split_training_data_from_file()

        #
        # Sort by our own Intent ID (IntentCategory/Intent), Intent Index
        #
        df_training_data_split = df_training_data_split.sort_values(
            by = [
                ctd.ChatTrainingData.COL_TDATA_INTENT_ID,
                ctd.ChatTrainingData.COL_TDATA_INTENT_INDEX
            ]
        )

        while True:
            print('Account ID=' + str(self.accountId)
                  + ', Bot ID=' + str(self.botId)
                  + ', Botkey=' + self.botkey + ': Choices')
            print('1: Import Intent Categories from csv file to DB.')
            print('2: Import Answers & Training Data from csv file to DB.')
            print('3: Export Training Data from DB to csv file.')
            print('e: Exit')
            user_choice = input('Enter Choice: ')

            start = pf.Profiling.start()
            print('Start Time: ' + str(start))

            if user_choice == '1':
                # Import Intent Categories first
                self.write_intent_categories_to_db(
                    df_training_data_split=df_training_data_split
                )

            if user_choice == '2':
                self.write_intents_trdata_answers_to_db(
                    df_training_data_split = df_training_data_split,
                    just_print_no_write_to_db = False
                )

            elif user_choice == '3':
                print('Sorry, function not yet implemented.')

            elif user_choice == 'e':
                break
            else:
                print('No such choice [' + user_choice + ']')

            stop = pf.Profiling.stop()
            print('Stop Time : ' + str(stop))
            print(pf.Profiling.get_time_dif_str(start, stop))


if __name__ == '__main__':
    db_profile = cf.ConfigFile.DB_PROFILE

    # DB Stuff initializations
    au.Auth.init_instances()

    [accountId, botId, botLang, botkey] = cmdline.CommandLine.get_parameters_to_run_bot(
        db_profile=cf.ConfigFile.DB_PROFILE
    )

    lg.Log.DEBUG_PRINT_ALL_TO_SCREEN = True

    bdi = BotDataImport(
        db_profile   = db_profile,
        accountId    = accountId,
        botId        = botId,
        lang         = botLang,
        botkey       = botkey,
    )
    bdi.run()
