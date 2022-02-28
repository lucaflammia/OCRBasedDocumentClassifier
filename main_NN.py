#!/usr/bin/env python3
import os
import sys
import time

from pprint import pprint
import sqlite3
import re
from nltk.corpus import stopwords
import matplotlib.pyplot as plt
import random

if sys.platform == 'win32':
    sys.path.append("C:\\Users\\Utente\\Documents\\Multitraccia\\Progetti\\Cobat"
                               "\\OCR_development")
else:
    sys.path.append("/Users/analisi/Luca/OCR_dev")

from OCR_DETECTION.conf_OCR import *
import string
import itertools
import numpy as np
import pandas as pd
import time
import traceback
import logging
from datetime import datetime
import tensorflow as tf
import tflearn
import simplejson
from tflearn import input_data, fully_connected, regression, DNN

now = datetime.now()
date_time = now.strftime("%Y-%m-%d %H-%M-%S")

if not (__name__ == '__main__') and sys.platform == 'win32':
    OCR_PATH = os.path.abspath("C:\\Users\\Utente\\Documents\\Multitraccia\\Progetti\\Cobat"
                               "\\OCR_development\\NEW_OCR")
    BASEPATH = os.path.abspath("C:\\Users\\Utente\\Documents\\Multitraccia\\Progetti\\Cobat"
                               "\\OCR_development")
    NN_PATH = os.path.abspath("C:\\Users\\Utente\\Documents\\Multitraccia\\Progetti\\Cobat\\OCR_development\\NN")
elif not (__name__ == '__main__') and sys.platform != 'win32':
    OCR_PATH = os.path.abspath("/Users/analisi/Luca/OCR_dev/OCR_DETECTION")
    BASEPATH = os.path.abspath("/Users/analisi/Luca/OCR_dev/")
    NN_PATH = os.path.abspath("/Users/analisi/Luca/OCR_dev/NN_CREATION")
else:
    OCR_PATH = os.path.abspath("../OCR_DETECTION/")
    BASEPATH = os.path.abspath(".")
    NN_PATH = os.path.abspath("../NN_CREATION/")

MODEL_PATH = os.path.join(NN_PATH, 'Model')

if not os.path.exists(os.path.join(NN_PATH, "Model")):
    os.makedirs(os.path.join(NN_PATH, "Model"))

if not os.path.exists(os.path.join(NN_PATH, "archive")):
    os.makedirs(os.path.join(NN_PATH, "archive"))

DB_OFFICIAL_PATH = os.path.join(OCR_PATH, 'DB_OFFICIAL')
ARCH_PATH = os.path.join(NN_PATH, "archive")
LOGFILE = "LOG_NN_{}.log".format(date_time)

format = '%(asctime)s : %(name)s : %(levelname)s : %(message)s'
logging.basicConfig(format=format)
logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)
# logging.basicConfig(filename=os.path.join(ARCH_PATH, LOGFILE), filemode='w', format=format)
output_file_handler = logging.FileHandler(os.path.join(ARCH_PATH, LOGFILE), mode='w', encoding='utf-8')
stdout_handler = logging.StreamHandler(sys.stdout)
formatter = logging.Formatter(format)
output_file_handler.setFormatter(formatter)
logger.addHandler(output_file_handler)
# logger.addHandler(stdout_handler)

log_error_path = os.path.join(ARCH_PATH, LOGFILE_ERROR)


class GetFirNN:
    def __init__(self, info_loaded_file='', logger='', type_nn='tflearn', build_model_from_app=False):
        # self.db = os.path.join(DB_OFFICIAL_PATH, 'OCR_FIR_MT.db')
        # self.conn = sqlite3.connect(self.db)
        # self.cur = self.conn.cursor()
        self.build_model_from_app = build_model_from_app
        self.logger = logger
        self.info_loaded_file = info_loaded_file
        self._perc_train_val_set = None
        self._hidden_nodes = None
        self._epochs = None
        self.type_NN = type_nn
        self.n_train_val = None
        self.n_train = None
        self.n_test = None

    def crea_strutture_training(self, nome_tipo):

        q = """
            SELECT file, ocr_prod FROM (
                SELECT t1.*, t2.tipologia FROM OCR_FIR_20210702 t1
                LEFT JOIN files_WEB_20210702 t2
                ON t1.file=t2.file
                UNION 
                SELECT t1.*, t2.tipologia  FROM OCR_FIR_20210708 t1
                LEFT JOIN files_WEB_20210708 t2
                ON t1.file=t2.file
                UNION 
                SELECT t1.*, t2.tipologia  FROM OCR_FIR_20210711 t1
                LEFT JOIN files_WEB_20210711 t2
                ON t1.file=t2.file
                UNION
                SELECT t1.*, t2.tipologia  FROM OCR_FIR_20210714 t1
                LEFT JOIN files_WEB_20210714 t2
                ON t1.file=t2.file
                UNION
                SELECT t1.*, t2.tipologia  FROM OCR_FIR_20210715 t1
                LEFT JOIN files_WEB_20210715 t2
                ON t1.file=t2.file
                ) AS U
            WHERE tipologia = '{nome_tipo}'
            ORDER BY file
        """.format(nome_tipo=nome_tipo)
        rows = self.cur.execute(q).fetchall()
        ocr_info_list = []
        self.logger.info('ROWS {}'.format(rows[:3]))
        for item in rows:
            par = re.sub("[\[\{\}\]\"\' ]", '', item[1])
            if re.search(',', par):
                par = par.split(',')
            if type(par) is str:
                par = [par]
            ocr_info_list.append((item[0], par))

        return ocr_info_list

    def crea_training_set(self, tot_info, tipologie, temi_tot):
        """
        Metodo che ritorna una tupla di due valori:
            - l'array degli input (train_x)
            - l'array degli output (train_y)

        I due array hanno lungezza fissa:
         - len(train_x) == len(temi)
         - len(train_y) == len(tipologie)
        """
        training = []
        output_vuota = [0] * len(tipologie)
        tipologie = list(tipologie)
        self.logger.info('LEN TOT INFO {0} {1}'.format(len(tot_info), tot_info[:5]))
        self.logger.info('TIPOLOGIE CONSIDERATE {}'.format(tipologie))

        for file, parole, tipologia in tot_info:
            ocr_parole_fir = [parola for parola in parole]

            # riempio la lista di input
            riga_input = [1 if t in ocr_parole_fir else 0 for t in temi_tot]

            # riempio la lista di output
            riga_output = output_vuota[:]
            riga_output[tipologie.index(tipologia)] = 1

            training.append([riga_input, riga_output])

        # mischio il mazzo (SOLO CASO DI UN NUOVO MODELLO CREATO DA APP OPPURE CHIAMATA DIRETTA DEL SORGENTE)
        if self.build_model_from_app or __name__ == '__main__':
            self.logger.info('MISCHIO IL TRAINING SET')
            random.shuffle(training)
        # trasformo in un array
        training = np.array(training, dtype=object)

        # e creo il training set
        train_x = list(training[:, 0])
        train_y = list(training[:, 1])
        return train_x, train_y

    def build_new_keras_nn(self, x, y, batch_size):
        # ckpt_path = ckpt_path.format(epoch=0, p_epoch= self._perc_train_val_set, hid_nodes=self._hidden_nodes)
        # UNA VOLTA CREATO IL MODELLO ALLORA EVITO DI MISCHIARE TRAINING SET CON self.build_model_from_app = False

        # mischio il mazzo anche nel caso percentuale training set di 100%
        # e prima avevo fatto 50% quindi serve mischiare ancora
        if self._perc_train_val_set == 100:
            self.build_model_from_app = False

        model = self.get_model(x, y)
        model.summary()
        model.compile(optimizer="adam", loss="categorical_crossentropy", metrics=["accuracy"])
        # cp_callback = tf.keras.callbacks.ModelCheckpoint(filepath=ckpt_path,
        #                                                  save_weights_only=False,
        #                                                  verbose=1,
        #                                                  save_freq='epoch')
        # tb_callback = tf.keras.callbacks.TensorBoard(log_dir='./logs', profile_batch=5)

        # SERIALIZZO IL MODELLO IN FORMATO JSON
        # https://stackoverflow.com/questions/35074549/how-to-load-a-model-from-an-hdf5-file-in-keras

        model_json = model.to_json()
        with open(os.path.join(MODEL_PATH,
                               "model_{0}PERC_TRAIN_{1}HID.json".format(self._perc_train_val_set, self._hidden_nodes)),
                  "w") as json_file:
            json_file.write(simplejson.dumps(simplejson.loads(model_json), indent=4))

        # model.save(os.path.join(MODEL_PATH, "model.h5"))

        model.fit(x, y, epochs=self._epochs, batch_size=batch_size)
        test_loss, test_acc = model.evaluate(x, y)
        self.logger.info("ACCURATEZZA TROVATA : {}".format(test_acc))

        saved_model = os.path.join(MODEL_PATH, "model_fit_eval_{0}PERC_TRAIN_{1}EPOCH_{2}HID.h5"
                                 .format(self._perc_train_val_set, self._epochs, self._hidden_nodes))
        self.logger.info('SALVATAGGIO MODELLO IN {}'.format(saved_model))
        model.save(saved_model)

        return model

    def classificatore_nn(self, x, y):
        """
        Questo metodo definisce e istruisce una
        ANN (Artificial Neural Network), di tipo
        DNN (Deep Neural Network) composta da:
            - un livello di input,
            - due hidden layer,
            - uno di output.
        Utilizza softmax come funzione di attivazione.

        I parametri sono:
           - X: array bidimensionale con i dati di input
           - y: array bidimensionale con i dati di output

        Una volta definita la struttura della rete neurale,
        ne viene fatto il training, e il modello viene
        salvato in un file, chiamato "rete.tflearn".
        """

        if self.type_NN == 'tflearn':
            model = self.get_model(x, y)
            ckpt_path = os.path.join(NN_PATH, "rete_{}perc_ep{}_hid_{}"
                                       .format(self._perc_train_val_set, self._epochs, self._hidden_nodes))
            try:
                model.load(ckpt_path)
            except:
                # resetto i dati del grafo
                tf.compat.v1.reset_default_graph()

                # Definire la Rete Neurale
                rete = input_data(shape=[None, len(x[0])])
                rete = fully_connected(rete, self._hidden_nodes)
                rete = fully_connected(rete, self._hidden_nodes)
                rete = fully_connected(rete, len(y[0]), activation='softmax')
                rete = regression(rete)

                model = DNN(rete, tensorboard_dir='logs', tensorboard_verbose=3)
                # Faccio il training
                model.fit(x, y, n_epoch=self._epochs, batch_size=8, validation_set=0.1, show_metric=True)
                # Salvataggio modello
                model.save(ckpt_path)

        elif self.type_NN == 'keras':
            # CONVERTIRE MODELLO TFLEARN A MODELLO KERAS
            # see https://stackoverflow.com/questions/59812388/converting-tflearn-model-to-keras

            self.logger.info(len(x))
            self.logger.info(len(y))
            self.logger.info(len(x[0]))
            self.logger.info(len(y[0]))

            x = np.array(x)
            y = np.array(y)
            # self.logger.info(x.shape)

            batch_size = 8

            # Train the model.
            # test_input = np.random.random((1831, len(x[0])))
            # test_target = np.random.random((1831, len(y[0])))
            # model.fit(test_input, test_target)
            # quit()

            # model = self.build_new_keras_nn(x, y, batch_size)
            #
            # return model

            # SE MODELLO RICHIESTO PRIMA VOLTA DA APP ALLORA LO COSTRUISCO DA ZERO
            # ALTRIMENTI PROVO A VEDERE SE HO MODELLO SALVATO
            if self.build_model_from_app:
                self.logger.info('{0} CREAZIONE NUOVO MODELLO {0}'.format('-' * 20))
                self.logger.info('DIMENSIONI MODELLO : INPUT {0}\t OUTPUT {1}'.format(x.shape, y.shape))
                model = self.build_new_keras_nn(x, y, batch_size)
            else:
                try:
                    # SEE https://www.tensorflow.org/guide/keras/save_and_serialize#savedmodel_format
                    # reconstructed_model = tf.keras.models.load_model(
                    #     os.path.join(MODEL_PATH, "model.h5"))
                    # reconstructed_model.fit(x, y)
                    reconstructed_model = tf.keras.models.load_model(
                        os.path.join(MODEL_PATH, "model_fit_eval_{0}PERC_TRAIN_{1}EPOCH_{2}HID.h5"
                                     .format(self._perc_train_val_set, self._epochs, self._hidden_nodes)))
                    reconstructed_model.fit(x, y)
                    self.logger.info('MODELLO RICOSTRUITO E CARICATO SU DISCO')
                    self.logger.info('DIMENSIONI MODELLO : INPUT {0}\t OUTPUT {1}'.format(x.shape, y.shape))
                    reconstructed_model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
                    score = reconstructed_model.evaluate(x, y, verbose=0)
                    self.logger.info("%s: %.2f%%" % (reconstructed_model.metrics_names[1], score[1] * 100))

                    return reconstructed_model
                except:
                    self.logger.info('{0} CREAZIONE NUOVO MODELLO {0}'.format('-' * 20))
                    self.logger.info('DIMENSIONI MODELLO : INPUT {0}\t OUTPUT {1}'.format(x.shape, y.shape))
                    model = self.build_new_keras_nn(x, y, batch_size)

            return model

    def get_model(self, x, y):

        if self.type_NN == 'tflearn':
            # resetto i dati del grafo
            tf.compat.v1.reset_default_graph()

            # Definire la Rete Neurale
            rete = input_data(shape=[None, len(x[0])])
            rete = fully_connected(rete, self._hidden_nodes)
            rete = fully_connected(rete, self._hidden_nodes)
            rete = fully_connected(rete, len(y[0]), activation='softmax')
            rete = regression(rete)

            model = DNN(rete, tensorboard_dir='logs', tensorboard_verbose=3)

        elif self.type_NN == 'keras':

            # Create a simple model.
            # see https://www.tensorflow.org/guide/keras/save_and_serialize
            # MODELLO FUNZIONALE GENERICO
            # inputs = tf.keras.Input(shape=(len(x[0]),))
            # outputs = tf.keras.layers.Dense(len(y[0]))(inputs)
            # model = tf.keras.Model(inputs, outputs)
            # model.compile(optimizer="adam", loss="categorical_crossentropy")

            # MODELLO SEQUENZIALE GENERICO
            # model = tf.keras.models.Sequential()
            # model.add(tf.keras.Input(shape=(len(x[0]))))
            # model.add(tf.keras.layers.Dense(self._hidden_nodes))
            # model.add(tf.keras.layers.Dense(self._hidden_nodes))
            # model.add(tf.keras.layers.Dense(len(y[0]), activation='softmax'))

            # MODELLO DA IMPLEMENTARE CONVERTITO DA QUELLO IN TFLEARN
            # see https://stackoverflow.com/questions/59812388/converting-tflearn-model-to-keras

            # MODELLO SEQUENZIALE OTTENUTO
            # see https://machinelearningmastery.com/tutorial-first-neural-network-python-keras/
            model = tf.keras.models.Sequential()
            model.add(tf.keras.layers.Dense(self._hidden_nodes, input_dim=len(x[0])))
            model.add(tf.keras.layers.Dense(self._hidden_nodes))
            model.add(tf.keras.layers.Dense(len(y[0]), activation='softmax'))

        return model

    def classifica(self, modello, tipologie, array):
        # genera le probabilità
        soglia_errore = 0.05
        if self.type_NN == 'keras':
            array = np.array([array])
            prob = modello.predict(array)[0]
        elif self.type_NN == 'tflearn':
            prob = modello.predict([array])[0]  # lista delle probabilità associate ad ognuna delle tipologie

        # filtro quelle che superano la soglia
        # prob_d = {v: prob[jj] for jj, v in enumerate(tipologie)}
        # print per test pochi dati
        # logger.info('mappa categoria - probabilita:\n{}'.format(prob_d))
        risultati = [
            [ii, p] for ii, p in enumerate(prob)
            if p > soglia_errore
        ]
        # ordino per le tipologie più probabili in ordine decrescente
        risultati.sort(key=lambda x: x[1], reverse=True)
        lista_tipologie = []
        for r in risultati:
            lista_tipologie.append((list(tipologie)[r[0]], r[1]))
        return lista_tipologie

    def estrai_temi(self, doc_in):
        """
        Estraggo i temi a partire
        dal documento in ingresso
        costituito dalla lista di tuple
        (parole OCR, tipologia)
        """
        temi_in = set()
        for par_OCR in doc_in:
            temi_in.update([par_OCR])

        temi_ocr_fir = list(temi_in)

        return temi_ocr_fir

    # DA LISTA DI PAROLE a LISTA DI 0 e 1
    def genera_input(self, lista_temi, temi_tot):
        """
        Conversione da lista di temi
        a lista di cifre costituita da 0 (no match) e 1 (match)
        """
        lista_input = [0]*len(temi_tot)
        for tema in lista_temi:
            for ii, t in enumerate(temi_tot):
                if t == tema:
                    lista_input[ii] = 1
        return np.array(lista_input)

    def trova_tipologie_predette(self, modello, temi_ocr_fir, tipologie, temi_tot):
        x = self.genera_input(temi_ocr_fir, temi_tot)
        # genero la classifica delle tipologie predette (almeno con 25%)
        # inserendo i temi di input nel modello di rete neurale
        tipologie_predette = self.classifica(modello, tipologie, x)

        if tipologie_predette:
            # SELEZIONO TIPOLOGIE CON PROBABILITA' >= SOGLIA_ERRORE = 25%
            return tipologie_predette

    def elabora_fir(self, rows):
        ocr_tipo_parole = []
        tipologie = set()
        temi_tot = set()

        self.logger.info('ROWS {}'.format(rows[:3]))
        for item in rows:
            parole_ocr = re.sub("[\[\{\}\]\"\' ]", '', item[1])
            if re.search(',', parole_ocr):
                parole_ocr = parole_ocr.split(',')
            if type(parole_ocr) is str:
                parole_ocr = [parole_ocr]
            temi_tot.update(parole_ocr)
            ocr_tipo_parole.append((item[0], parole_ocr, item[2]))
            tipologie.add(item[2])

        temi_tot = list(set(parola for parola in temi_tot))
        tipologie = list(tipologie)

        return temi_tot, tipologie, ocr_tipo_parole

    def building_nn(self):
        db = os.path.join(DB_OFFICIAL_PATH, 'OCR_FIR_MT.db')
        conn = sqlite3.connect(db)
        cur = conn.cursor()

        q = """
            SELECT file, ocr_prod, tipologia FROM (
                SELECT t1.*, t2.tipologia FROM OCR_FIR_20210702 t1
                LEFT JOIN files_WEB_20210702 t2
                ON t1.file=t2.file
                UNION 
                SELECT t1.*, t2.tipologia  FROM OCR_FIR_20210708 t1
                LEFT JOIN files_WEB_20210708 t2
                ON t1.file=t2.file
                UNION 
                SELECT t1.*, t2.tipologia  FROM OCR_FIR_20210711 t1
                LEFT JOIN files_WEB_20210711 t2
                ON t1.file=t2.file
                UNION
                SELECT t1.*, t2.tipologia  FROM OCR_FIR_20210714 t1
                LEFT JOIN files_WEB_20210714 t2
                ON t1.file=t2.file
                UNION
                SELECT t1.*, t2.tipologia  FROM OCR_FIR_20210715 t1
                LEFT JOIN files_WEB_20210715 t2
                ON t1.file=t2.file
            ) AS U 
            ORDER BY file;
        """
        rows = cur.execute(q).fetchall()
        conn.close()

        temi_tot, tipologie, tot_ocr_tipo_parole = self.elabora_fir(rows)
        if __name__ == '__main__':
            self.logger.info("NUMERO TIPOLOGIE DEI FIR INDIVIDUATE = {}".format(len(tipologie)))
            self.logger.info("NUMERO FIR CON OCR PAROLE = {}".format(len(tot_ocr_tipo_parole)))
            self.logger.info('LUNGHEZZA SET PAROLE DISTINTE = {}'.format(len(temi_tot)))

        TRAIN_TEST_RATIO = 70
        train_val_test_int = int(len(tot_ocr_tipo_parole) * int(TRAIN_TEST_RATIO) / 100)

        train_val_set = tot_ocr_tipo_parole[:train_val_test_int]
        train_val_int = int(len(train_val_set) * int(self._perc_train_val_set) / 100)

        train_set = train_val_set[:train_val_int]
        # NEL CASO SI CONSIDERA TRAINING SET NON COMPLETO MI ASSICURO CHE CI SIA DENTRO IL FILE CARICATO IN APP.py
        if not (len(train_val_set) == len(train_set)):
            # SOSTITUISCO ULTIMO ELEMENTO CON FILE CARICATO
            train_set[-1] = (self.info_loaded_file)

        val_set = train_val_set[train_val_int:]
        test_set = tot_ocr_tipo_parole[train_val_test_int:]

        self.n_train_val = len(train_val_set)
        self.n_train = len(train_set)
        self.n_test = len(test_set)

        if __name__ == '__main__':
            self.logger.info('TRAIN {}'.format(train_set[:5]))
            self.logger.info('TIPOLOGIE {}'.format(tipologie[:5]))

        xtrial, ytrial = self.crea_training_set(train_set, tipologie, temi_tot)
        # PROVO VALIDATION SET? PER MODEL.fit(validation_set=...)
        #xval, yval = self.crea_validation_set(test_set, tipologie, temi_tot)

        # self.save_load_simple_nn()
        # quit()

        modello = self.classificatore_nn(xtrial, ytrial)

        stat_nn = {}
        equality = {}

        for case in ['train_set', 'test_set']:
            if (__name__ == '__main__') and case == 'test_set':
                self.logger.info('{0} {1} {0}'.format('-' * 20, case.upper()))
                self.logger.info('PERC TRAINING SET {}'.format(self._perc_train_val_set))

            tipol_oss = []
            tipol_pred = []

            for i, (file_in, parola_in, tipologia_in) in enumerate(eval(case)):
                stat_nn[file_in] = {
                    'OCR_WORDS': None,
                    'TIPOL_OSS': None,
                    'GUESS_TIPOL_PRED': None,
                    'TIPOL_PRED_LIST': [],
                    'TYPE_SET': case.split('_')[0].upper()
                }
                temi_ocr_fir = self.estrai_temi(parola_in)
                tipol_oss.append(tipologia_in)
                t_pred = self.trova_tipologie_predette(modello, temi_ocr_fir, tipologie, temi_tot)
                tipol_pred.append(t_pred[0][0])

                stat_nn[file_in]['OCR_WORDS'] = parola_in
                stat_nn[file_in]['TIPOL_OSS'] = tipologia_in
                if tipologia_in == t_pred[0][0]:
                    stat_nn[file_in]['GUESS_TIPOL_PRED'] = 'True'
                else:
                    stat_nn[file_in]['GUESS_TIPOL_PRED'] = 'False'
                stat_nn[file_in]['TIPOL_PRED_LIST'] = [(t_pred[i][0], t_pred[i][1]) for i in range(len(t_pred))]

            if (__name__ == '__main__') and case == 'test_set':
                self.logger.info('TUTTE LE CATEGORIE OSSERVATE ASSOCIATE AI {} FIR ANALIZZATI CORRISPONDONO '
                                 'A QUELLE PREDETTE ? --> {}'.format(len(tipol_oss), np.all(tipol_oss == tipol_pred)))

                self.logger.info('{0}'.format('+' * 60))

            # comparazione categoria osservata e categoria predetta con maggiore probabilità
            tipol_oss = np.array(tipol_oss, dtype=object)
            tipol_pred = np.array(tipol_pred, dtype=object)

            data = {
                'Tipol. Osservata': tipol_oss,
                'Tipol. Predetta': tipol_pred
            }

            df_stat = pd.DataFrame(data, columns=data.keys())
            df_stat['Equality'] = tipol_oss == tipol_pred

            equality[case] = df_stat['Equality'].value_counts()
            if (__name__ == '__main__') and case == 'test_set':
                self.logger.info('UGUAGLIANZA TIPOLOGIA OSSERVATA VS PREDETTA {0}'.format(equality[case]))

        return stat_nn, equality

    def write_files_nn(self, nn_res={}, df={}, equality=None):
        test_files = []
        for file in nn_res:
            if nn_res[file]['TYPE_SET'] == 'TEST':
                test_files.append(file)

        test_df_cond = df['TYPE_SET'] == 'TEST'

        file_test = os.path.join(NN_PATH, 'TEST_30PERC_DATASET.txt')
        self.logger.info('SCRITTURA FILE {}'.format(file_test))

        with open(file_test, 'w') as f:
            f.write('{0} FILES NEL TEST SET\n'.format(len(test_files)))
            f.write('\n{}\n'.format('+' * 20))
            f.write('{0}\n {1} \n{0}'.format('-' * 20, df[test_df_cond].groupby('TIPOL_OSS')['FILES'].count()))
            f.write('\n{}\n'.format('+' * 20))
            f.write('\n{}\n'.format('+' * 20))
            f.write('{0} {1} {0}'.format('-' * 20, df[test_df_cond].groupby('OCR_TOT_WORDS')['FILES'].count()))
            f.write('\n{}\n'.format('+' * 20))
            for file in test_files:
                f.write('{0} - {1} - {2}\n'.format(file, nn_res[file]['TIPOL_OSS'], nn_res[file]['OCR_WORDS']))
            f.write('{}'.format('+' * 20))
            f.close()

        guess_cond = df['GUESS_TIPOL_PRED'] == 'True'

        str_param = '{0}PERC_TRAIN_ep_{1}_hid_nodes_{2}' \
            .format(self.perc_train_val_set, self.epochs, self.hidden_nodes)

        pred_test = os.path.join(NN_PATH, 'FILES_SET_{}.txt'.format(str_param))
        self.logger.info('SCRITTURA FILE {}'.format(pred_test))

        with open(pred_test, 'w') as f:
            f.write('DATETIME {}\n'.format(date_time))
            f.write('TRAINING SET FATTO SU {1} DEI {2} FIR ({0}% DEL SET TOTALE)\n'
                    .format(self.perc_train_val_set, self.n_train, self.n_train_val))
            f.write('{0}\nFILES PREDETTI CORRETTAMENTE SU {1} CONSIDERATI NEL TEST SET\n'
                    .format(equality['test_set'], self.n_test))
            f.write('\n{}\n'.format('+' * 20))
            f.write(
                '{0}\n {1} \n{0}'.format('-' * 20, df[test_df_cond & guess_cond].groupby('TIPOL_OSS')['FILES'].count()))
            f.write('\n{}\n'.format('+' * 20))
            for file in test_files:
                if nn_res[file]['GUESS_TIPOL_PRED'] == 'True':
                    f.write(
                        '{0} - {1} - {2}\n'.format(file, nn_res[file]['TIPOL_OSS'], nn_res[file]['TIPOL_PRED_LIST']))
            f.write('{}'.format('+' * 20))
            f.close()

    @property
    def perc_train_val_set(self):
        """I'm the perc_train_val_set property."""
        # self.logger.info("CHIAMATA GETTER DI perc_train_val_set")
        return self._perc_train_val_set

    @perc_train_val_set.setter
    def perc_train_val_set(self, value):
        # self.logger.info("CHIAMATA SETTER DI perc_train_val_set")
        self._perc_train_val_set = value

    @property
    def hidden_nodes(self):
        """I'm the hidden_nodes property."""
        # self.logger.info("CHIAMATA GETTER DI hidden_nodes")
        return self._hidden_nodes

    @hidden_nodes.setter
    def hidden_nodes(self, value):
        # self.logger.info("CHIAMATA SETTER DI hidden_nodes")
        self._hidden_nodes = value

    @property
    def epochs(self):
        """I'm the epochs property."""
        # self.logger.info("CHIAMATA GETTER DI epochs")
        return self._epochs

    @epochs.setter
    def epochs(self, value):
        # self.logger.info("CHIAMATA SETTER DI epochs")
        self._epochs = value


def get_plot_predizione(equality, train_set, test_set):
    fig, axes = plt.subplots(nrows=2, ncols=1, figsize=(10, 10))

    stat1 = equality['train_set']

    # -------------- PLOT --------------
    stat1.plot(kind='bar', ax=axes[0], width=.3, color='green')
    axes[0].set(title='NUMERO ELEMENTI TRAINING SET = {}'.format(len(train_set)))
    pc = [int(stat1.iloc[0]) / len(train_set) * 100, int(stat1.iloc[1]) / len(train_set) * 100]
    axes[0].set_xticks([stat1.index[1], stat1.index[0]])
    axes[0].set_xticklabels(['{}\n{}\n{:2.1f}%'.format(
        stat1.index[0], stat1.iloc[0], pc[0]),
        '{}\n{}\n{:2.1f}%'.format(stat1.index[1], stat1.iloc[1], pc[1])], rotation=0)
    axes[0].legend(
        ['train_perc={}\nhidden nodes={}\nepoch={}'.format(nn.perc_train_val_set, nn.hidden_nodes, nn.epochs)])

    stat2 = equality['test_set']

    # -------------- PLOT --------------
    stat2.plot(kind='bar', ax=axes[1], width=.3, color='green')
    axes[1].set(title='NUMERO ELEMENTI TEST SET = {}'.format(len(test_set)))
    pc = [int(stat2.iloc[0]) / len(test_set) * 100, int(stat2.iloc[1]) / len(test_set) * 100]
    axes[1].set_xticks([stat2.index[1], stat2.index[0]])
    axes[1].set_xticklabels(['{}\n{}\n{:2.1f}%'.format(stat2.index[0], stat2.iloc[0], pc[0]),
                             '{}\n{}\n{:2.1f}%'.format(stat2.index[1], stat2.iloc[1], pc[1])],
                            rotation=0)
    axes[1].legend(
        ['train_perc={}\nhidden nodes={}\nepoch={}'.format(nn.perc_train_val_set, nn.hidden_nodes, nn.epochs)])
    str_param = '{0}PERC_TRAIN_ep_{1}_hid_nodes_{2}' \
        .format(nn.perc_train_val_set, nn.epochs, nn.hidden_nodes)
    save_path = os.path.join(BASEPATH, 'Plot', "PREDIZIONE_TIPOLOGIA_FIR_{}.png".format(str_param))
    plt.savefig(save_path)
    plt.show()


def get_dataframe(nn_res={}):
    # DA DIZIONARIO A DF IN CUI LA CHIAVE (FIR) E' POSTO COME INDICE
    dframe = pd.DataFrame.from_dict(nn_res, orient='index')
    # SPOSTO INDICE COME COLONNA
    dframe = dframe.reset_index()
    dframe = dframe.rename(columns={'index': 'FILES'})
    dframe['OCR_TOT_WORDS'] = dframe["OCR_WORDS"].str.len()

    return dframe


def get_plot_predizione_tipologia():
    fig, axes = plt.subplots(nrows=2, ncols=1, figsize=(10, 10))
    # fig.set_figwidth(8)

    df = get_dataframe(nn_res=stat_nn)

    set_cond = df['TYPE_SET'] == 'TEST'
    guess_cond = df['GUESS_TIPOL_PRED'] == 'True'

    data = {
        'True': [],
        'False': []
    }

    logger.info('DF = 96532_DOC278 --> {}'.format(df[df['FILES'] == '96532_DOC278'].values))

    logger.info('{0} INIZIO GUESS TIPOLOGIA TOTALE CASI {0}'.format('#' * 20))
    for nword in range(1, 5):
        word_cond = df["OCR_WORDS"].str.len() == nword
        df_count_tot_guess = df[word_cond & set_cond]['FILES'].count()
        logger.info('{0}W TOT GUESS--> NO. {1}'.format(nword, df_count_tot_guess))
        df_count_ok_guess = df[word_cond & guess_cond & set_cond]['FILES'].count()
        logger.info('{0}W OK GUESS --> NO. {1}'.format(nword, df_count_ok_guess))
        data['True'].append(df_count_ok_guess)
        data['False'].append(df_count_tot_guess - df_count_ok_guess)

    word4p_cond = df["OCR_WORDS"].str.len() > 4
    df_count4p_tot_guess = df[word4p_cond & set_cond]['FILES'].count()
    logger.info('>4W TOT GUESS--> NO. {0}'.format(df_count4p_tot_guess))
    df_count4p_ok_guess = df[word4p_cond & guess_cond & set_cond]['FILES'].count()
    data['True'].append(df_count4p_ok_guess)
    data['False'].append(df_count4p_tot_guess - df_count4p_ok_guess)
    logger.info('>4W OK GUESS--> NO. {0}'.format(df_count4p_ok_guess))
    logger.info('{0} FINE GUESS TIPOLOGIA TOTALE CASI {0}'.format('#' * 20))

    index = ['1W', '2W', '3W', '4W', '>4W']

    plotdata1 = pd.DataFrame(data, index=index)
    plotdata1.plot.bar(ax=axes[0], stacked=True)
    axes[0].set_xticklabels(index, rotation=0)
    axes[0].set(ylabel='OCCORRENZE', title='RISCONTRO PREDIZIONE TIPOLOGIA FIR')
    axes[0].legend()

    tipologie = ['FIR - RIMONDI PAOLO SRL', 'FIR - TRS', 'FORMULARIO PULI ECOL',
                 'FORMULARIO RIFIUTI - ALLEGATO B - ETM', 'NIECO']
    data = {}

    for tipologia in tipologie:
        #logger.info('{0} INIZIO STAT TIPOLOGIA {1} {0}'.format('+' * 20, tipologia))
        data[tipologia] = []
        tipol_cond = df['TIPOL_OSS'] == tipologia
        for nword in range(1, 5):
            #logger.info('{0} INIZIO STAT PAROLE {1} {0}'.format('*' * 20, nword))
            word_cond = df["OCR_WORDS"].str.len() == nword
            df_count = df[word_cond & tipol_cond & guess_cond & set_cond]['FILES'].count()
            df_values = df[word_cond & tipol_cond & guess_cond & set_cond].values
            #logger.info('{0} : {1}W --> NO. {2} {3}'.format(tipologia, nword, df_count, df_values))
            #logger.info('{0} FINE STAT PAROLE {1} {0}'.format('*' * 20, nword))
            data[tipologia].append(df_count)

        #logger.info('{0} INIZIO STAT PAROLE >4 {0}'.format('*' * 20))
        word4p_cond = df["OCR_WORDS"].str.len() > 4
        df_count4p = df[word4p_cond & tipol_cond & guess_cond & set_cond]['FILES'].count()
        df_values4p = df[word4p_cond & tipol_cond & guess_cond & set_cond].values
        #logger.info('{0} : >4W --> NO. {1} {2}'.format(tipologia, df_count4p, df_values4p))
        data[tipologia].append(df_count4p)

        #logger.info('{0} FINE STAT PAROLE >4 {0}'.format('*' * 20))
        #logger.info('{0} FINE STAT TIPOLOGIA {1} {0}'.format('+' * 20, tipologia))
        # logger.info('{0} INIZIO LOG STAT {0}'.format('+' * 20))
        # logger.info('{0} : >4W {1}'.format(tipologia, df[word4p & (df['TIPOL_OSS'] == tipologia)
        #                                                  & guess_cond & set_cond]['FILES'].values))
        # logger.info('{0} : >4W {1}'.format(tipologia, df[word4p & (df['TIPOL_OSS'] == tipologia)
        #                                                  & guess_cond & set_cond]['FILES'].count()))

    logger.info('DATA {}'.format(data))

    plotdata2 = pd.DataFrame(data, index=index)
    plotdata2.plot.bar(ax=axes[1], stacked=True)
    axes[1].set_xticklabels(index, rotation=0)
    axes[1].set(ylabel='OCCORRENZE', title='PREDIZIONE CORRETTA PER TIPOLOGIA FIR')
    plt.suptitle('PREDIZIONE TIPOLOGIA FIR vs NUMERO PAROLE INDIVIDUATE DA OCR')

    str_param = '{0}PERC_TRAIN_ep_{1}_hid_nodes_{2}' \
        .format(nn.perc_train_val_set, nn.epochs, nn.hidden_nodes)
    save_path = os.path.join(BASEPATH, 'Plot', "PREDIZIONE_TIPOLOGIA_FIR_{}.png".format(str_param))
    plt.savefig(save_path)
    plt.show()

    return df


if __name__ == '__main__':
    logger.info('{0} INIZIO ESECUZIONE JOB PER DETERMINAZIONE TIPOLOGIA FIR {0}'.format('!' * 20))
    start_time = time.time()

    nn_param_setup = {
        'PERC TRAIN VAL SET': [100],
        'HIDDEN NODES': [64],
        'EPOCHS': [150]
    }

    nn_params = []
    for key, value in nn_param_setup.items():
        nn_params.append(nn_param_setup[key])

    nn = GetFirNN(logger=logger, type_nn='keras')
    for ii, nn_param in enumerate(list(itertools.product(*nn_params))):
        logger.info('{0} {1} DS PERC 70 - 30 {0}'.format('#' * 20, 'INIZIO CREAZIONE RETE NEURALE'))
        nn.perc_train_val_set = nn_param[0]
        nn.hidden_nodes = nn_param[1]
        nn.epochs = nn_param[2]

        stat_nn, equality = nn.building_nn()

        df = get_plot_predizione_tipologia()
        nn.write_files_nn(nn_res=stat_nn, equality=equality, df=df)

        logger.info('{0} {1} DS PERC 70 - 30 {0}'
                    .format('#' * 20, 'FINE CREAZIONE RETE NEURALE'))

    logger.info('ESECUZIONE AVVENUTA IN {0} SECONDI'.format(time.time() - start_time))
    logger.info('{0} ESECUZIONE JOB TERMINATA {0}'.format('!' * 20))
