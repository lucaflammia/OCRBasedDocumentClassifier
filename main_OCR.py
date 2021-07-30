#!/usr/bin/env python3
import os
import sys
import time
from pdf2image import convert_from_path

import conf_OCR
from pprint import pprint
from conf_OCR import *
from conf_OCR import CreateNewDatabase
from conf_OCR import QueryFir
import sqlite3
import re
import shutil
import cv2
from PIL import Image
import keras_ocr
import random
import string
from itertools import combinations
import numpy as np
import pandas as pd
import pytesseract
import enchant
import time
from nltk import word_tokenize
from pytesseract import Output
import traceback
import logging
from datetime import datetime

now = datetime.now()

PRED_PATH = conf_OCR.PRED_PATH
BASEPATH = conf_OCR.BASEPATH
IMAGE_PATH = conf_OCR.IMAGE_PATH
ARCH_PATH = conf_OCR.ARCH_PATH
PNG_IMAGE_PATH = conf_OCR.PNG_IMAGE_PATH
LOGFILE = conf_OCR.LOGFILE
LOGFILE_ERROR = conf_OCR.LOGFILE_ERROR
TIPOLOGIA_FIR = conf_OCR.TIPOLOGIA_FIR

DPI = conf_OCR.DPI

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

pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'


class GetFirOCR:
    def __init__(self, file='', logger='', web=True):
        self.file = file
        self.db = os.path.join(DB_BACKUP_PATH, 'OCR_MT_MERGE_STATIC_CHECK.db')
        self.conn = sqlite3.connect(self.db)
        self.cur = self.conn.cursor()
        self.web = web
        self.qy = QueryFir(self.web)
        self.check_dtm = ''
        self.logger = logger
        self.file_only = ''
        self.width = None
        self.height = None
        self.crop_width = None
        self.crop_height = None
        self.rotated_file = False
        self.flt = set(['GRAY'])
        self.accepted_words = set()
        self.nome_tipologia = 'NC'
        self.tipologia = 'NC'
        self.produttore = 'NOT FOUND'
        self.trasportatore = 'NOT FOUND'
        self.raccoglitore = 'NOT FOUND'
        self.cod_rifiuto = 'INFORMAZIONE NON FORNITA'
        self.peso_riscontrato = 'INFORMAZIONE NON FORNITA'
        self.ocr_fir = {}
        self.full_info = {}

    def word_like_cond(self, target, fieldname='parola', perc=False):
        wcond = {}
        if type(target) is list:
            for word in target:
                word_l = list(word)
                wlike = []
                or_lett = []
                for i, lett in enumerate(word_l):
                    or_lett.append(lett)
                    word_l[i] = '_'
                    if i > 0:
                        word_l[i - 1] = or_lett[i - 1]
                    wlike.append(''.join(word_l))
                wcond[word] = {
                    "{fieldname} like '{perc}{el}{perc}'".format(fieldname=fieldname,
                                                                 el=el, perc='%' if perc else '') for el in wlike
                }
        else:
            word_l = list(target)
            wlike = []
            or_lett = []
            for i, lett in enumerate(word_l):
                or_lett.append(lett)
                word_l[i] = '_'
                if i > 0:
                    word_l[i - 1] = or_lett[i - 1]
                wlike.append(''.join(word_l))
            wcond[target] = {
                "{fieldname} like '{perc}{el}{perc}'".format(fieldname=fieldname,
                                                             el=el, perc='%' if perc else '') for el in wlike
            }

        return wcond

    def save_move_delete_png(self, info='', delete_from_folder=''):
        img = Image.open(self.file)
        img_copy = img.copy()
        img.close()
        if info:
            copy_filepath = os.path.join(PNG_IMAGE_PATH, self.nome_tipologia,
                                         self.file_only + '_{}'.format(info) + '.png')
        else:
            copy_filepath = os.path.join(PNG_IMAGE_PATH, self.nome_tipologia,
                                         self.file_only + '.png')
        img_copy.save(copy_filepath, 'png')

        if delete_from_folder:
            if os.path.exists(os.path.join(PNG_IMAGE_PATH, delete_from_folder, self.file_only + '_PRODUTTORE.png')):
                os.remove(os.path.join(PNG_IMAGE_PATH, delete_from_folder, self.file_only + '_PRODUTTORE.png'))


    def check_from_old_db(self):
        # CASO RICERCA DI FALSI POSITIVI DI UNA TIPOLOGIA INIZIALMENTE INDIVIDUATA
        # SE TROVATI ALLORA EVITO ANALISI OCR
        # CONSIDERA IL DB TOTALE E VERIFICA INFO FIR PREGRESSE
        img = self.open_fir()
        self.fir_properties()
        dtms = ['20210702', '20210708', '20210711', '20210714', '20210715']
        for kk, dtm in enumerate(dtms):
            q = """
                SELECT file, tipologia FROM files_WEB_{dtm}
                where file like '{file}%';
            """.format(dtm=dtm, file=self.file_only)
            res = self.cur.execute(q).fetchall()
            if (not kk == len(dtms) - 1) and (not res):
                continue
            elif (kk == len(dtms) - 1) and (not res):
                self.logger.info('FILE {0} NON ANALIZZATO IN PASSATO'.format(self.file_only))
                return self.ocr_fir

            self.file_only = self.cur.execute(q).fetchall()[0][0]
            nome_tipologia_to_check = self.cur.execute(q).fetchall()[0][1]
            logger.info('FILE ANALIZZATO IN PASSATO E PRESENTE IN DB')
            self.logger.info('TIPOLOGIA DA VERIFICARE {}'.format(nome_tipologia_to_check))
            if nome_tipologia_to_check == 'NC':
                self.logger.info('TIPOLOGIA NC INDIVIDUATA PER FILE {}'.format(self.file_only))
                if os.path.exists(os.path.join(PNG_IMAGE_PATH, "NC", self.file_only + '.png')):
                    os.remove(os.path.join(PNG_IMAGE_PATH, 'NC', self.file_only + '.png'))

            self.check_dtm = dtm
            break

        self.logger.info('{0} IGNORO ANALISI OCR E VERIFICO CORRETTEZZA TIPOLOGIA {0}'.format('-' * 20))

        tipo_fir_list = []
        for elem in TIPO_FIR:
            if not elem == 'NC':
                tipo_fir_list.append(elem)

        word_like = {}
        self.logger.info('RICERCA SU DB {}'.format(self.check_dtm))
        for jj, tipo in enumerate(tipo_fir_list):
            tlist = TIPO_FIR.get(tipo)['TEXT']
            nwlist = []
            for (nword, divy) in TIPO_FIR[tipo]['NO_WORD']:
                nwlist.append(nword)

            wlist = tlist + nwlist

            word_like[tipo] = self.word_like_cond(wlist)

            # PER FARE CHECK SU QUEL DB DEFINITO DA DATETIME DTM
            self.qy = QueryFir(self.web, self.check_dtm)

            self.get_tipologia(tipo, word_like[tipo])
            if self.tipologia == 'NC':
                continue
            TIPO_FIR[self.tipologia]['FILES'].append(self.file_only)
            # INSERISCO MODIFICHE NEL DB ODIERNO PER DOUBLE CHECK
            # db_now = os.path.join(DB_PATH, 'OCR_MT.db')
            # self.logger.info('DB NOW {}'.format(db_now))
            # conn = sqlite3.connect(db_now)
            # cur = conn.cursor()
            # q = """
            #     INSERT INTO files_WEB (file,tipologia,produttore,trasportatore,raccoglitore,ts) VALUES
            #     ('{file}','{tipol}','','','', CURRENT_TIMESTAMP );
            # """.format(file=self.file_only, tipol=self.nome_tipologia)
            # cur.execute(q)
            # conn.commit()
            # conn.close()
            if not nome_tipologia_to_check == self.nome_tipologia:
                # SPOSTO IMMAGINE PNG NELLA CARTELLA "NUOVA TIPOLOGIA"
                # E LO RIMUOVO DA QUELLA VECCHIA
                if not nome_tipologia_to_check == 'NC':
                    self.save_move_delete_png(info='PRODUTTORE', delete_from_folder=nome_tipologia_to_check)
                else:
                    self.save_move_delete_png(delete_from_folder=nome_tipologia_to_check)
            else:
                self.logger.info('{0} TIPOLOGIA PER FILE {1} CONFERMATA A {2} {0}'
                                 .format('+' * 20, self.file_only, nome_tipologia_to_check))
                # SE IMMAGINE CARICATA DA WEB APP ALLORA SALVO ANCHE SU DIRECTORY ASSOCIATA DI NEW_OCR
                filetosave = os.path.join(PNG_IMAGE_PATH, self.nome_tipologia, self.file_only + '.png')
                filePRODtosave = os.path.join(PNG_IMAGE_PATH, self.nome_tipologia, self.file_only + '_PRODUTTORE.png')
                if (not os.path.exists(filetosave)) or (not os.path.exists(filePRODtosave)):
                    self.save_move_delete_png(info='PRODUTTORE')
                q = """
                    SELECT *
                    FROM "{table}"
                    WHERE file = '{file}'
                """.format(table="OCR_FIR_{}".format(self.check_dtm), file=self.file_only)
                res = self.cur.execute(q).fetchall()
                if res:
                    item = res[0]
                    self.ocr_fir = {'ocr_prod': item[4], 'ocr_trasp': item[5], 'ocr_racc': item[6],
                                    'ocr_size': item[2]}

                    self.full_info = self.read_full_info_from_csv(info='PRODUTTORI')
                    ok_words = set()
                    for k, lst in self.full_info['PRODUTTORI'].items():
                        for elem in lst:
                            if len(elem) > 4 or (len(elem) >= 4 and re.search('[aeiou]$', elem)):
                                ok_words.add(elem)

                    for key_tipo, val_d in TIPO_FIR.items():
                        for key, val in val_d.items():
                            if val == self.nome_tipologia:
                                tipologia = key_tipo

                    # AGGIUNGO E RIMUOVO PAROLE DA QUELLE FINORA ACCETTATE
                    self.accepted_words = set(list(set(ok_words) - set(INFO_FIR['PROD']['NO_WORD_OCR']))
                                              + COMMON_FIR_INFO[tipologia])

                    self.check_esclusione_ocr_fir()
            break

        if self.nome_tipologia != nome_tipologia_to_check:
            self.logger.info('FILE {0} AGGIORNATO DA TIPOLOGIA {1} A {2}'
                             .format(self.file_only, nome_tipologia_to_check, self.nome_tipologia))
            q = """
               UPDATE "{table}"
               SET tipologia = "{val_field}"
               WHERE file = "{file}"
            """.format(table='files_WEB_{}'.format(self.check_dtm) if self.web else 'files',
                       val_field=self.nome_tipologia, file=self.file_only)
            self.cur.execute(q)
            self.conn.commit()
            # SE FIR NON E' STATO IDENTIFICATO ALLORA CANCELLO INFO
            # SPOSTO IMMAGINE PNG NELLA CARTELLA "NC"
            # E LO RIMUOVO DA QUELLA VECCHIA
            if self.nome_tipologia == 'NC':
                self.cur = self.conn.cursor()
                q = """
                   DELETE FROM {table}
                   WHERE file = "{file}"
                """.format(table='OCR_FIR_{}'.format(self.check_dtm), file=self.file_only)
                self.cur.execute(q)
                self.conn.commit()
                self.save_move_delete_png(delete_from_folder=nome_tipologia_to_check)
                self.logger.info('FILE {0} NON CONFERMATO PER {1} E NON IDENTIFICATO A NUOVA TIPOLOGIA'
                                 .format(self.file_only, nome_tipologia_to_check))

        return self.ocr_fir

    def open_fir(self):
        if sys.platform == 'win32':
            self.file_only = '_'.join(self.file.split('\\')[-1].split('.')[0].split('_')[:2])
        else:
            self.file_only = '_'.join(self.file.split('/')[-1].split('.')[0].split('_')[:2])

        Image.MAX_IMAGE_PIXELS = 1000000000
        img = Image.open(self.file)
        self.width, self.height = img.size

        return img

    def fir_properties(self):
        self.logger.info('INFO ESATTE CERCATE NEL DB PER FILE {}'.format(self.file_only))
        self.produttore = self.get_exact_info(info='prod')
        self.logger.info('{0} PRODUTTORE : {1} {0}'.format('-' * 20, self.produttore))
        self.trasportatore = self.get_exact_info(info='trasp')
        self.logger.info('{0} TRASPORTATORE : {1} {0}'.format('-' * 20, self.trasportatore))
        self.raccoglitore = self.get_exact_info(info='racc')
        self.logger.info('{0} DESTINATARIO : {1} {0}'.format('-' * 20, self.raccoglitore))
        self.get_exact_info(info='FIR')
        self.logger.info('{0} COD RIFIUTO : {1} {0}'.format('-' * 20, self.cod_rifiuto))
        self.logger.info('{0} PESO RISCONTRATO : {1} {0}'.format('-' * 20, self.peso_riscontrato))

    def perform_ocr_fir(self):
        word_like = {}

        img = self.open_fir()

        self.logger.info('{0} ESECUZIONE OCR PER FILE : {1} {0}'.format('+' * 20, self.file_only))
        self.logger.info('SIZE IMMAGINE : {0} w - {1} h'.format(self.width, self.height))

        # CREA NUOVO DB PER NUOVI FIR ESEGUENDO OCR COMPLETO
        # OPPURE FACCIO OCR RITAGLIO NEL CASO DI FIR GIA' ANALIZZATO
        if not self.check_dtm:
            self.db = os.path.join(DB_PATH, 'OCR_MT.db')
            self.conn = sqlite3.connect(self.db)
            self.cur = self.conn.cursor()
            self.logger.info('RICERCA SU NUOVO DB CREATO')
            CreateNewDatabase(self.db, self.web)
        else:
            self.logger.info('RICERCA SU DB CREATO IN DATA {}'.format(self.check_dtm))

        res_file = self.check_file(table='files_WEB')
        # CERCA SE IL FILE E' STATO SALVATO CON ROTAZIONE
        if not res_file:
            res_file = self.check_file(table='files_WEB', rotation=True)
            if res_file:
                self.rotated_file = True
        res_parole = self.check_file(table='parole_WEB')
        if not res_parole:
            self.logger.info('FILE NON ANALIZZATO CON OCR INIZIALE')
            self.logger.info('ANALISI INIZIALE OCR PER FILE {0}'.format(self.file_only))
            self.ocr_analysis(img)
            self.logger.info('FINE ANALISI INIZIALE OCR PER FILE {0}'.format(self.file_only))
        else:
            # SE FILE GIA' REGISTRATO IN DB ALLORA DETERMINO LA TIPOLOGIA A PARTIRE DAL NOME TIPOLOGIA
            for row in res_file:
                self.nome_tipologia = row[2]
            for key_tipo, val_d in TIPO_FIR.items():
                for key, val in val_d.items():
                    if val == self.nome_tipologia:
                        self.tipologia = key_tipo

        if not self.tipologia == 'NC':
            self.logger.info("PER FILE {0} TIPOLOGIA GIA' INDIVIDUATA --> {1}"
                             .format(self.file_only, self.nome_tipologia))
            if self.rotated_file:
                # SE HO GIA' FIR ANALIZZATO CON ROTAZIONE IL FILENAME HA IN AGGIUNTA "_rot"
                orig_filename = self.file_only + '.png'
                self.logger.info('NOME PRIMA ROTAZIONE {}'.format(orig_filename))
                img = Image.open(os.path.join(PNG_IMAGE_PATH, orig_filename))
                img_copy = img.copy()
                img.close()
                copy_filepath = os.path.join(PNG_IMAGE_PATH, self.file_only + '.png')
                rot = int(self.file_only.split('_rot')[1])
                img_copy = self.rotate_file(img_copy, rot=rot)
                img_copy.save(copy_filepath, 'png')
                os.remove(os.path.join(PNG_IMAGE_PATH, orig_filename))
                # if self.rotated_file:
                #     rot = self.file_only.split('_')[-1]
                #     self.update_rotated_filename(rot)
        else:
            # INDIVIDUO LA TIPOLOGIA DEL FIR ANALIZZATO
            for tipo in TIPO_FIR:
                if tipo == 'NC':
                    continue

                tlist = TIPO_FIR.get(tipo)['TEXT']
                nwlist = []
                for (nword, divy) in TIPO_FIR[tipo]['NO_WORD']:
                    nwlist.append(nword)

                wlist = tlist + nwlist

                word_like[tipo] = self.word_like_cond(wlist)

                self.qy = QueryFir(self.web, self.check_dtm)

                self.get_tipologia(tipo, word_like[tipo])
                if not self.tipologia == 'NC':
                    TIPO_FIR[self.tipologia]['FILES'].append(self.file_only)
                    self.aggiorna_campo_tabella(field='tipologia', val_field=self.nome_tipologia)
                    break

            if not os.path.exists(os.path.join(PNG_IMAGE_PATH, "{}".format(self.nome_tipologia))):
                os.makedirs(os.path.join(PNG_IMAGE_PATH, "{}".format(self.nome_tipologia)))

            if self.tipologia == 'NC':
                TIPO_FIR['NC']['FILES'].append(self.file_only)
                # SPOSTO IMMAGINE PNG NELLA CARTELLA "NC"
                self.save_move_delete_png()
                self.logger.info('TIPOLOGIA NON DETERMINATA PER {} --> '
                                 'NESSUNA INFO A DISPOSIZIONE\n\n\n'.format(self.file_only))
                for tipo in TIPO_FIR:
                    if os.path.exists(os.path.join(
                            PNG_IMAGE_PATH, tipo, self.file_only + '_PRODUTTORE.png')):
                        os.remove(os.path.join(PNG_IMAGE_PATH, tipo, self.file_only + '_PRODUTTORE.png'))
                return

        res = self.check_file(table='OCR_FIR')
        # INSERISCO PAROLE DA ACCETTARE QUALORA VENISSERO INDIVIDUATE DURANTE OCR
        # self.insert_common_words(info_fir='PRODUTTORE')
        # self.read_full_info_from_csv(info='FIR')
        if res:
            self.logger.info("PAROLE DETERMINATE DA OCR GIA' ACQUISITE")
            self.logger.info("ESECUZIONE PER FILE {} TERMINATA".format(self.file_only))
            self.check_esclusione_ocr_fir()
            return self.ocr_fir
        # IN CASO DI TIPOLOGIA TROVATA E NON ANALIZZATA, SI CERCANO LE INFO
        for inf in ['prod', 'trasp', 'racc']:
            self.full_info = self.read_full_info_from_csv(info=INFO_FIR[inf.upper()]['TABLE'])
            self.get_ocr_info(inf)
            break

        return self.ocr_fir

    def check_esclusione_ocr_fir(self):
        """
        IGNORO RISULTATI OCR IN CUI TROVO PAROLE NON APPARTENENTI A PAROLE REALMENTE ESISTENTI NEI FIR
        CONSIDERANDO QUEI SET DI PAROLE OCR CON LUNGHEZZA INFERIORE O UGUALE 2
        :return:
        """
        self.logger.info('CONTROLLO SE RISULTATI OCR SONO DETERMINATI DA SCRITTURA INCOMPRENSIBILE')
        ocr_info_list = []
        # par = re.sub('\W', '', self.ocr_fir['ocr_prod'])
        par = re.sub("[\[\{\}\]\"\' ]", '', str(self.ocr_fir['ocr_prod']))
        if not re.search(',', par):
            ocr_info_list.append(par)
        else:
            par = par.split(',')
            for elem in par:
                ocr_info_list.append(elem)

        self.logger.info('RISULTATO OCR DA VERIFICARE {}'.format(ocr_info_list))
        flag = True
        if len(ocr_info_list) <= 3:
            for val in ocr_info_list:
                if val not in self.accepted_words:
                    flag = False
                    self.logger.info("PAROLA OCR '{}' NON ACCETTATO DERIVANTE DA SCRITTURA POCO COMPRENSIBILE"
                                     .format(val))
                    ocr_info_list.remove(val)
                    data = set(ocr_info_list)
                    self.ocr_fir['ocr_prod'] = data
                    if len(self.ocr_fir['ocr_prod']) > 0:
                        self.update_info_db(data)
                    else:
                        self.delete_table(table='OCR_FIR')
                    filesaved = os.path.join(PNG_IMAGE_PATH, self.nome_tipologia, self.file_only + '.png')
                    if not os.path.exists(filesaved):
                        self.save_move_delete_png(delete_from_folder=self.nome_tipologia)
        if flag:
            self.logger.info('RISULTATO OCR CORRETTO E MANTENUTO INALTERATO')

    def esclusione_parole_tipologia(self, tipo, word_like, pid):

        for (nword, divy) in TIPO_FIR[tipo]['NO_WORD']:
            # RICERCA PAROLA DA ESCLUDERE VICINA A QUELLE CERCATE
            clike = '(' + ' or '.join(word_like[nword]) + ')'
            if len(nword) > 10:
                plike = """
                    ( parola like '{s00}%{s01}' OR
                    parola like '%{s10}' OR
                    parola like '{s20}%')
                """.format(s00=nword[:3], s01=nword[-3:], s10=nword[-8:], s20=nword[4:])
            else:
                plike = """
                    ( parola like '{s00}%{s01}')
                """.format(s00=nword[:3], s01=nword[-3:])

            nowq = """
                {sub_body} WHERE
                ({clike} OR {plike} ) AND
                p.id < {pid} + {did} AND
                p.id > {pid} - {did} AND
                file = '{file}';
            """.format(sub_body=self.qy.sub_body, clike=clike, plike=plike,
                       pid=pid, did=TIPO_FIR[tipo]['DLT_ID'], file=self.file_only)

            nwres = self.cur.execute(nowq).fetchall()
            if nwres:
                self.logger.info('TROVA PAROLA INTRUSA PER QUERY: \n{}'.format(nowq))
                return nwres

        # PAROLA LONTANA DA QUELLE CERCATE (ES. "ROTTAMI") MA CHE,
        # SE TROVATA, ESCLUDE LA TIPOLOGIA (ES. "TIPOLOGIA A")
        # ---- LA PAROLA INTRUSA PUO' TROVARSI IN UN QUADRANTE DIVERSO (DEFINITO DA divy) ----
        # ---- LA PAROLA INTRUSA VIENE CERCATA NELLA PARTE SUPERIORE (coor_y < 600) ----
        nowords = [str(w) for (w, divy) in TIPO_FIR[tipo]['NO_WORD']]
        exc_word = "('" + "','".join(nowords) + "')"
        nowq = """
            {sub_body} WHERE
            (parola in {exc_word}) AND
            div_x = '1-2' AND
            div_y = '{divy}' AND
            coor_y < 600 AND
            file = '{file}';
        """.format(sub_body=self.qy.sub_body, exc_word=exc_word, divy=divy, file=self.file_only)

        nwres = self.cur.execute(nowq).fetchall()
        if nwres:
            self.logger.info('ESCLUSIONE CON PAROLA LONTANA {}'.format(nowq))
            return nwres

        return None

    def get_tipologia(self, tipo, word_like):

        occ_l = []

        for ii, txt in enumerate(TIPO_FIR[tipo]['TEXT']):
            if not ii == len(TIPO_FIR[tipo]['TEXT']) - 1:
                txt = TIPO_FIR[tipo]['TEXT'][0]
                clike = '(' + ' or '.join(word_like[txt]) + ')'
                plike = """
                    ( parola like '{s00}%{s01}' OR
                    parola like '{s00}%' OR
                    parola like '%{s01}' )
                """.format(s00=txt[:3], s01=txt[-3:])

                subq = """
                    {sub_body} WHERE
                    ({clike} OR {plike}) AND
                    div_x = '1-2' AND
                    div_y = '1-4' AND
                    file = '{file}'
                    LIMIT 1;
                """.format(sub_body=self.qy.sub_body, clike=clike, plike=plike, file=self.file_only)

                self.logger.debug('RICERCA {0} : {1}'.format(tipo.upper(), subq))
                sres = self.cur.execute(subq).fetchall()

                if not sres:
                    self.logger.info("NESSUN RISULTATO IDONEO PER LA PAROLA '{0}'".format(txt))
                    self.logger.info("FILE {0} NON APPARTENENTE ALLA TIPOLOGIA {1}".format(self.file_only, tipo))
                    return

                pid = sres[0][0]

                if TIPO_FIR[tipo]['NO_WORD']:
                    # RICERCA FALSI POSITIVI. SE HO UN RISCONTRO ESCLUDO LA TIPOLOGIA
                    nwres = self.esclusione_parole_tipologia(tipo, word_like, pid)
                    if nwres:
                        self.logger.info('TROVATA PAROLA CHE ESCLUDE IL FILE {0} DALLA TIPOLOGIA {1}'
                                         .format(self.file_only, tipo))
                        return

            else:
                clike = '(' + ' or '.join(word_like[txt]) + ')'
                if len(txt) > 8:
                    plike = """
                        (parola like '{s00}%{s01}' OR
                        parola like '%{s10}%' OR
                        parola like '{s20}%')
                    """.format(s00=txt[:2], s01=txt[-2:], s10=txt[3:-3], s20=txt[:4])
                else:
                    plike = """
                        (parola like '{s00}%{s01}' OR
                        parola like '{s20}%')
                    """.format(s00=txt[:2], s01=txt[-2:], s20=txt[:4])

                q = """
                    {body} WHERE
                    ({clike} OR {plike}) AND
                    p.id < {pid} + {did} AND
                    p.id > {pid} - {did} AND
                    file = '{file}';
                """.format(body=self.qy.body, clike=clike, plike=plike,
                           pid=pid, did=TIPO_FIR[tipo]['DLT_ID'], file=self.file_only)

                self.logger.debug('RICERCA {0} : {1}'.format(tipo.upper(), q))
                res = self.cur.execute(q).fetchall()

                occ_l.append(len(res))

                if occ_l == [0 * i for i in range(len(occ_l))]:
                    self.logger.info("NESSUN RISULTATO IDONEO PER LA PAROLA '{0}'".format(txt))
                    self.logger.info("FILE {0} NON APPARTENENTE ALLA TIPOLOGIA {1}".format(self.file_only, tipo))
                    return

                self.tipologia = tipo
                self.nome_tipologia = TIPO_FIR['{}'.format(self.tipologia)]['NAME']
                self.logger.info('{0} TIPOLOGIA FIR : {1} {0}'.format('-' * 20, self.nome_tipologia))

    def image_preprocessing(self, cfilepath):
        # PREPROCESSING
        im = cv2.imread(cfilepath)
        # SOVRASCRIVO RITAGLIO CON IMMAGINE GRIGIA
        # gray = cv2.cvtColor(np.uint8(im), cv2.COLOR_BGR2GRAY)
        gray = self.get_grayscale(np.uint8(im))
        if self.nome_tipologia == 'FIR - TRS':
            self.flt.add('THRS')
            self.flt.add('BLUR_GAUSS')
            gray = self.thresholding(gray)
            # TYPE = MEDIAN PUO' DARE UN RISULTATO MIGLIORE
            # VERIFICA IN FUTURO SE VUOI ACCEDERE A QUESTO FITRO PER OCR
            gray = self.remove_noise(gray, type='gaussian')
        # NUOVA SOGLIA IMMAGINE, PIXELS IN RISALTO A 255 MENTRE QUELLI IN BACKGROUN VANNO A 0
        # gray = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)[1]
        # RIMUOVO IL RUMORE
        # gray = cv2.medianBlur(gray, 3)
        return gray

    def crop_top_area(self, top_ini=0):
        q = """
            SELECT coor_y 
            FROM parole_WEB{dtm} t1
            LEFT JOIN files_WEB{dtm} t2
            ON (t1.id_file=t2.id)
            WHERE file = '{file}' 
            ORDER BY t1.id ASC LIMIT 1;
        """.format(dtm='_{}'.format(self.check_dtm) if self.check_dtm else '', file=self.file_only)
        cy = self.cur.execute(q).fetchall()[0][0]
        # SE PRIMA PAROLA HA COORDINATA Y MAGGIORE DI 200 ALLORA FACCIO TAGLIO SU TOP
        # PER RIMUOVERE IL BIANCO FOGLIO DATO DALLA ROTAZIONE
        top_out = top_ini
        if cy > 200:
            top_out = top_ini + 300

        return top_out

    def ocr_analysis_ritaglio(self, info, cutoff_width=0, config_ocr=r'--oem 3 --psm 4'):
        info_fir = INFO_FIR[info.upper()]['TEXT']
        # PROVATO https://www.geeksforgeeks.org/text-detection-and-extraction-using-opencv-and-ocr/
        # ALTRO MODO https://arxiv.org/ftp/arxiv/papers/1509/1509.03456.pdf
        # USATO https://www.pyimagesearch.com/2017/07/10/using-tesseract-ocr-python/

        # PER INFO PYTESSERACT
        # https://jaafarbenabderrazak-info.medium.com/ocr-with-tesseract-opencv-and-python-d2c4ec097866
        # BEST PRACTICES https://ai-facets.org/tesseract-ocr-best-practices/

        # SE FILE CARICATO DA WEB_APP ALLORA LO COPIO NELLA DIRECTORY NEW_OCR/IMAGES
        try:
            orig_filepath = os.path.join(PNG_IMAGE_PATH, '{0}.png'.format(self.file_only))
            img = Image.open(orig_filepath)
        except Exception:
            self.logger.info('FILE CARICATO DA WEB_APP E NON PRESENTE IN {}'.format(os.path.abspath(PNG_IMAGE_PATH)))
            self.logger.info('UTILIZZO FIR PRESENTE NELLA DIRECTORY {} PER FORNIRE COPIA NELLA DIRECTORY'
                             .format(os.path.join(PRED_PATH, 'Demo_APP', 'WEB_APP', 'images')))
            orig_filepath = os.path.join(PRED_PATH, 'Demo_APP', 'WEB_APP', 'images', '{0}.png'.format(self.file_only))
            img = Image.open(orig_filepath)
            img_copy = img.copy()
            img_copy.save(os.path.join(PNG_IMAGE_PATH, '{0}.png'.format(self.file_only)), 'png')
            img_copy.close()

        # SE HO FILENAME MODIFICATO CON "_rot" ALLORA CONSIDERO IL FILE ORIGINALE SENZA QUESTA DICITURA
        # PER EFFETTUARE IL RITAGLIO
        # if self.rotated_file:
        #     orig_filepath = os.path.join(PNG_IMAGE_PATH, '{0}.png'.format(self.file_only))
        left = TIPO_FIR['{}'.format(self.tipologia)]['SIZE_OCR'][0]
        top = TIPO_FIR['{}'.format(self.tipologia)]['SIZE_OCR'][1]
        bottom = TIPO_FIR['{}'.format(self.tipologia)]['SIZE_OCR'][3]

        if not self.rotated_file:
            right = self.width + 5 + cutoff_width  # SCELGO UN RITAGLIO PER TUTTA
            # LA LARGHEZZA DEL FIR + 5 (SCELTA CHE PORTA SOLO A MIGLIORARE OCR) + CUTOFF INSERITO
        else:
            # LA ROTAZIONE CONSIDERA UNA MAGGIORE ESTENSIONE DEL FOGLIO IN LARGHEZZA CHE NON E' OTTIMALE
            # DIMINUISCO DI 1000 IN LARGHEZZA
            right = self.width - 1550 + 5 + cutoff_width

        # PER FIR - TRS ALTEZZA RITAGLIO PROD E' NEL RANGE [650,1050] per SIZE STANDARD INPUT FIR di 3334
        # MANTENGO RAPPORTO 650 / 3334 = 0.1949.. NEL CASO DI ALTEZZE DIVERSE IN INPUT
        # MANTENGO RAPPORTO 1050 / 3334 = 0.3149.. NEL CASO DI ALTEZZE DIVERSE IN INPUT

        if self.nome_tipologia == 'FORMULARIO RIFIUTI - ALLEGATO B - ETM':
            # LA ROTAZIONE CONSIDERA UNA MAGGIORE ESTENSIONE DEL FOGLIO IN ALTEZZA CHE NON E' OTTIMALE
            # TAGLIO 300 IN ALTEZZA
            if self.rotated_file:
                top = self.crop_top_area()
                bottom = (self.height * 0.2099) + 300
            else:
                top = 0
                bottom = (self.height * 0.2099)
        elif self.nome_tipologia in ('FIR - TRS', 'FORMULARIO PULI ECOL'):
            # LA ROTAZIONE CONSIDERA UNA MAGGIORE ESTENSIONE DEL FOGLIO IN ALTEZZA CHE NON E' OTTIMALE
            # TAGLIO 300 IN ALTEZZA
            if self.rotated_file:
                top = self.crop_top_area(top_ini=TIPO_FIR['{}'.format(self.tipologia)]['SIZE_OCR'][1])
            else:
                top = (self.height * 0.1949) if self.nome_tipologia == 'FIR - TRS' else (self.height * 0.16496)
            bottom = (self.height * 0.3149)

        wsize = int(right) - int(left)
        hsize = int(bottom) - int(top)

        self.logger.info('ANALISI OCR RITAGLIO {0} PER FILE {1} : SIZE ( {2} - {3} )'
                         .format(INFO_FIR[info.upper()]['TEXT'], self.file_only, wsize, hsize))

        img_copy = img.copy()
        img.close()

        copy_filepath = os.path.join(PNG_IMAGE_PATH, '{}_copy.png'.format(self.file_only))

        img_copy.save(copy_filepath, 'png')

        img_copy = Image.open(copy_filepath)

        img_crop = img_copy.crop((left, top, right, bottom))
        self.crop_width, self.crop_height = img_crop.size

        cfilename = '{0}_{1}.png'.format(self.file_only, INFO_FIR[info.upper()]['TEXT'])
        cfilepath = os.path.join(PNG_IMAGE_PATH, '{}'.format(self.nome_tipologia), cfilename)
        img_crop.save(cfilepath)

        gray = self.image_preprocessing(cfilepath)
        # im = cv2.imread(cfilepath)
        # gray = cv2.cvtColor(np.uint8(im), cv2.COLOR_BGR2GRAY)

        cv2.imwrite("{}".format(cfilepath), gray)
        # RIMUOVO COPIA E FILE INTERO. MANTENGO SOLO IL RITAGLIO
        os.remove(copy_filepath)

        img_crop = Image.open(cfilepath)
        # d = pytesseract.image_to_data(img_crop, output_type=Output.DICT)
        text = pytesseract.image_to_string(img_crop, config=config_ocr)
        # self.logger.info('DICT {}'.format(d['text']))
        text_l = text.split('\n')
        data = {'{}'.format(info_fir): []}
        foo = []
        for txt in text_l:
            if txt:
                foo.append(txt.split(' '))

        for t_l in foo:
            for t in t_l:
                if re.search("\w", t):
                    t = t.lower()
                    data.get(info_fir).append(t)

        self.logger.info('DATA {}'.format(data))
        self.logger.info('{0} {1} RECORDS TROVATI {0}'.format('-' * 20, len(data.get(info_fir))))

        if not data.get(info_fir):
            return None, None, None

        parole, id_dict = self.query_info_db(data)
        id_st = id_dict['ID_START']
        id_fin = id_dict['ID_FIN']

        self.logger.info('FINE ANALISI OCR RITAGLIO {0} PER FILE {1}'
                         .format(INFO_FIR[info.upper()]['TEXT'], self.file_only))

        return parole, id_st, id_fin

    def update_rotated_filename(self, rot):
        orig_file = self.file
        orig_file_only = self.file_only
        self.logger.info('FILE NAME PRIMA ROTAZIONE {}'.format(self.file_only))
        self.file = self.file.split('.png')[0] + '_rot{}'.format(rot) + '.png'
        # split('_')[3] poichè considero stringa aggiuntiva '_rot'
        if sys.platform == 'win32':
            self.file_only = '_'.join(self.file.split('\\')[-1].split('.')[0].split('_')[:3])
        else:
            self.file_only = '_'.join(self.file.split('/')[-1].split('.')[0].split('_')[:3])
        self.logger.info('NUOVO NOME DOPO ROTAZIONE : {}'.format(self.file_only))
        q = """
            UPDATE files_WEB{dtm} SET file = '{rot_file}'
            WHERE file = '{orig_file}'
        """.format(dtm='_{}'.format(self.check_dtm) if self.check_dtm else '',
                   rot_file=self.file_only, orig_file=orig_file_only)
        self.cur.execute(q)
        self.conn.commit()
        # FILE E' CHIAMATO COME MODULO ESTERNO ALLORA EVITO CANCELLAZIONE FILE
        if __name__ == 'main':
            os.remove(os.path.join(orig_file))

    def ocr_analysis(self, img):
        # SEE https://github.com/faustomorales/keras-ocr/issues/65
        # Disable GPU, use CPU only
        os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
        # alphabet = string.digits + string.ascii_letters + '!?. '
        # recognizer_alphabet = ''.join(sorted(set(alphabet.lower())))
        # recognizer = keras_ocr.recognition.Recognizer(
        #     alphabet=recognizer_alphabet,
        #     weights='kurapan'
        # )
        # recognizer.model.load_weights(os.path.join(PRED_PATH, 'Tuning_Recognizer', 'recognizer_borndigital.h5'))
        # pipeline = keras_ocr.pipeline.Pipeline(recognizer=recognizer)
        pipeline = keras_ocr.pipeline.Pipeline()
        # PROVO INIZIALMENTE OCR DEL FILE SENZA ROTAZIONE
        # SE HO RISULTATO CORRETTO ESCO DAL CICLO FOR
        # ROT 180 SIGNIFICA CHE SBAGLIANDO LA ROTAZIONE DI 90 DA UNA PARTE LA CORREGGO CON STESSA ROTAZIONE
        # DALL'ALTRA PARTE
        for rot in [0, 90, 180]:
            img = self.rotate_file(img, rot=rot)
            # CONSIDERO I RITAGLI (lunghezza / nw, altezza / nh)
            nw = 2
            nh = 4
            data = []
            # CONSIDERO SOLO I PRIMI DUE RITAGLI SUPERIORI
            self.logger.info("{0} ESECUZIONE OCR PER FILE {1} {0}".format('-' * 20, self.file_only))
            for ih in [1, 2]:
                top = (ih - 1) * self.height / nh
                bottom = ih * self.height / nh
                div_y = "{:.3f}H".format(float(ih / nh))
                # CONSIDERO SOLO IL PRIMO RITAGLIO DA SINISTRA
                for iw in [1]:
                    image = []
                    left = (iw - 1) * self.width / nw
                    right = iw * self.width / nw
                    img_crop = img.crop((left, top, right, bottom))
                    div_x = "{:.3f}W".format(float(iw / nw))
                    cfilepath = os.path.join(IMAGE_PATH, 'CROP_{0}_{1}-{2}.png'.format(self.file_only, div_x, div_y))
                    img_crop.save("{}".format(cfilepath), 'png')
                    image.append(keras_ocr.tools.read(cfilepath))
                    self.logger.info("{0} INIZIO PER RITAGLIO {1}-{2} {0}".format('-' * 20, div_x, div_y))
                    tstart = time.time()
                    # OGNI ELEMENTO OCR E' UNA LISTA DI TUPLE (parola, coordinate_parola)
                    raw_data = pipeline.recognize(image)
                    self.logger.info('{0} FINE ESECUZIONE IN {1} SECONDI {0}'.format('-' * 20, time.time() - tstart))
                    os.remove(cfilepath)

                    for (t, c) in raw_data[0]:
                        data.append((t, c.tolist(), '{}-{}'.format(iw, nw), '{}-{}'.format(ih, nh),))

            self.logger.info('{0} FINE ESECUZIONE OCR {0}'.format('-' * 20))
            self.insert_new_records_table(table='files_WEB', dpi=DPI)
            self.insert_new_records_table(data=data, table='parole_WEB', dpi=DPI)

            # CHECK SE OCR RISULTA INADEGUATO POICHE' IL FILE E' RUOTATO E NON LEGGO BENE PAROLE
            # VERIFICANDO ESISTENZA DI ALMENO UNA PAROLA SENSATA
            common_fir_words = ['detentore', 'produttore', 'denominazione', 'ragione', 'sociale', 'identificazione',
                                'unita', 'locale', 'codice', 'fiscale', 'autorizzazione', 'formulario', 'trasporto',
                                'futuro', 'sostenibile']
            accepted_word = False
            for par, (lu, ru, ld, rd), div_x, div_y in data:
                if par in common_fir_words:
                    accepted_word = True
                    break
            self.logger.info('RICERCA DI PAROLE COMUNI NEL FIR : ESITO -> {}'.format(accepted_word))
            # TRAMITE PERCENTUALE DI RIGHE RUOTATE
            res = self.check_file(table='parole_WEB', rotation=True)
            for row in res:
                tilted_rows = row[0]
            perc_tilted_rows = int(tilted_rows / len(data) * 100)
            self.logger.info('PERCENTUALE RIGHE SOSPETTE (LUNGHEZZA 0 OPPURE 1) : {}%'.format(perc_tilted_rows))

            # SE LA PERCENTUALE DI RIGHE NON ACCETTATE E' INFERIORE AL 60% ALLORA ACCETTO IL RISULTATO
            # E CONTEMPORANEAMENTE SE ESISTE ALMENO UNA PAROLA SENSATA
            if (not perc_tilted_rows > 60) and (accepted_word):
                if self.rotated_file:
                    self.update_rotated_filename(rot)
                    self.logger.info('OCR INIZIALE FILE {} VALIDO CON ROTAZIONE {}'
                                     .format(self.file_only, rot))
                else:
                    self.logger.info('OCR INIZIALE FILE {} VALIDO CON ROTAZIONE NULLA'
                                     .format(self.file_only))
                img.save(self.file)
                break

    def check_file(self, table, rotation=False):
        data_info = None
        if table.startswith('OCR_FIR'):
            q = """
                SELECT *
                FROM OCR_FIR{dtm}
                WHERE file = '{file}'
            """.format(dtm='_{}'.format(self.check_dtm) if self.check_dtm else '', file=self.file_only)
        elif table.startswith('files_WEB'):
            if rotation:
                q = """
                    SELECT *
                    FROM files_WEB{dtm}
                    WHERE file like '{file}_rot%'
                    AND
                    tipologia != 'NC'
                """.format(dtm='_{}'
                           .format(self.check_dtm) if self.check_dtm else '', file=self.file_only)
            else:
                q = """
                    SELECT *
                    FROM files_WEB{dtm}
                    WHERE file = '{file}'
                    AND
                    tipologia != 'NC'
                """.format(dtm='_{}'
                           .format(self.check_dtm) if self.check_dtm else '', file=self.file_only)
        elif table.startswith('parole_WEB'):
            if rotation:
                q = """
                    SELECT count(parola) FROM parole_WEB{dtm} p
                    LEFT JOIN files_WEB{dtm} f
                    ON (f.id=p.id_file)
                    WHERE file = '{file}' AND 
                    length(parola) in (0, 1);
                """.format(dtm='_{}'
                           .format(self.check_dtm) if self.check_dtm else '', file=self.file_only)
            else:
                q = """
                    SELECT * FROM parole_WEB{dtm} p
                    LEFT JOIN files_WEB{dtm} f
                    ON (f.id=p.id_file)
                    WHERE file = '{file}';
                """.format(dtm='_{}'.format(self.check_dtm) if self.check_dtm else '', file=self.file_only)
        else:
            q = """
               SELECT *
               FROM {table}{dtm} t1
               LEFT JOIN files_WEB{dtm} t2
               ON (t1.id_file=t2.id)
               WHERE file = '{file}'
           """.format(table=table, dtm='_{}'
                      .format(self.check_dtm) if self.check_dtm else '', file=self.file_only)
        res = self.cur.execute(q).fetchall()
        if table.startswith('files_WEB'):
            row = [elem for elem in res]
            try:
                self.file_only = row[0][1]
                self.file = os.path.join(PNG_IMAGE_PATH, self.file_only + '.png')
            except Exception:
                self.logger.info('RISULTATO NON TROVATO NEL DB PRESENTE CON ROTAZIONE = {}'.format(rotation))
        if table.startswith('OCR_FIR') and res:
            item = res[0]
            self.ocr_fir = {'ocr_prod': item[4], 'ocr_trasp': item[5], 'ocr_racc': item[6], 'ocr_size': item[2]}

        return res

    def query_info_db(self, data):
        q = """
            SELECT id FROM files_WEB{dtm}
            WHERE file = "{file}"
        """.format(dtm='_{}'.format(self.check_dtm) if self.check_dtm else '', file=self.file_only)

        id_file = self.cur.execute(q).fetchall()[0][0]

        info_fir = None
        for key in data.keys():
            info_fir = key

        res = self.check_file(table='OCR_{}'.format(info_fir))

        ok_words = set()
        for k, lst in self.full_info['PRODUTTORI'].items():
            for elem in lst:
                if len(elem) > 4 or (len(elem) >= 4 and re.search('[aeiou]$', elem)):
                    ok_words.add(elem)

        # AGGIUNGO E RIMUOVO PAROLE DA QUELLE FINORA ACCETTATE
        self.accepted_words = set(list(set(ok_words) - set(INFO_FIR['PROD']['NO_WORD_OCR']))
                             + COMMON_FIR_INFO[self.tipologia])

        if not res:
            for par in data.get(info_fir):
                # ELIMINO CARATTERE UNDERSCORE (POICHE' E' ALFANUMERICO)
                par = re.sub('_', '', par)
                if par in ['s.r.l', 'sr.l', 's.rl']:
                    res_par = 'srl'
                # SE TROVO CARATTERI SPECIALI NELLA PAROLA
                elif re.search('\W', par) and len(par) > 3:
                    res_par = re.split('\W', par)
                    for rpar in res_par:
                        # DEVO SEPARARE CARATTERI DA CIFRE ES. AUTORICAMBI3.GI
                        if re.search('\d', rpar):
                            res_par = re.split('\d', rpar)
                # SE TROVO CARATTERI ORDINALI INSIEME A CIFRE
                elif re.search('\w', par) and re.search('\d', par) and len(par) > 3:
                    res_par = re.split('\d', par)
                # CONSIDERO CASO PAROLA LUNGA DATO DA INSIEME PAROLE SENSO COMPIUTO (ES. "SEARISORSESPA")
                elif len(par) >= 8 and par not in self.accepted_words:
                    fragment_words = ['spa', 'srl', 'sea']
                    res_par = []
                    foo = set()
                    for fragment_word in fragment_words:
                        if fragment_word in par:
                            lst = par.split(fragment_word)
                            res_par.append(fragment_word)
                            for txt in lst:
                                if len(txt) >= 5:
                                    for word in self.accepted_words:
                                        if len(word) >= 5 and word in txt:
                                            foo.add(word)
                    for elem in foo:
                        res_par.append(elem)
                else:
                    res_par = ''
                    for c in par:
                        if c in string.printable:
                            res_par += c
                if isinstance(res_par, str):
                    # CASO OCR (CON PYTESSERACT POSSO OTTENERE CARATTERI INTRUSI ES. '‘PRODUTTORE')
                    res_par = re.sub('\W', '', res_par)
                    # res_par = re.sub('[\[()‘“~/?+"\'_.=~\-\]]', '', res_par)
                    if res_par:
                        self.conn = sqlite3.connect(self.db)
                        self.cur = self.conn.cursor()
                        q = """
                            INSERT INTO {table}{dtm}(parola,id_file,flt,ts)
                            VALUES ("{par}", "{id_file}", "{flt}", CURRENT_TIMESTAMP)
                        """.format(table='OCR_{}'.format(info_fir),
                                   dtm='_{}'.format(self.check_dtm) if self.check_dtm else '', par=res_par,
                                   id_file=id_file, flt=self.flt)
                        self.cur.execute(q)
                        self.conn.commit()
                elif isinstance(res_par, list):  # CASO 'PRODUTTORE/DETENTORE'
                    for rpar in res_par:
                        rpar = re.sub('_', '', rpar)
                        if rpar:
                            self.conn = sqlite3.connect(self.db)
                            self.cur = self.conn.cursor()
                            q = """
                                INSERT INTO {table}{dtm}(parola,id_file,flt,ts)
                                VALUES ("{par}", "{id_file}", "{flt}", CURRENT_TIMESTAMP)
                            """.format(table='OCR_{}'.format(info_fir),
                                       dtm='_{}'.format(self.check_dtm) if self.check_dtm else '', par=rpar,
                                       id_file=id_file, flt=self.flt)
                            self.cur.execute(q)
                            self.conn.commit()
        else:
            self.logger.info("RITAGLIO GIA' ANALIZZATO DA OCR")
            self.logger.info('NESSUN INSERIMENTO IN TABELLA OCR_{} PER FILE {}'.format(info_fir, self.file_only))

        q = """
            SELECT parola FROM {table}{dtm}
            WHERE id_file = '{id_file}'
        """.format(table='OCR_{}'.format(info_fir),
                   dtm='_{}'.format(self.check_dtm) if self.check_dtm else '', id_file=id_file)

        parole = []
        for par in self.cur.execute(q).fetchall():
            parole.append(par[0])

        id_dict = {}
        for ord in ['ASC', 'DESC']:

            q = """
                SELECT id FROM {table}{dtm}
                WHERE id_file = '{id_file}'
                ORDER BY id {ord}
                LIMIT 1
            """.format(table='OCR_{}'.format(info_fir),
                       dtm='_{}'.format(self.check_dtm) if self.check_dtm else '', id_file=id_file, ord=ord)

            if ord == 'ASC':
                id_dict['ID_START'] = self.cur.execute(q).fetchall()[0][0]
            else:
                id_dict['ID_FIN'] = self.cur.execute(q).fetchall()[0][0]

        return parole, id_dict

    def update_info_db(self, data={}):
        self.conn = sqlite3.connect(self.db)
        self.cur = self.conn.cursor()
        q = """
            UPDATE OCR_FIR{dtm} 
            SET ocr_prod = "{ocr_prod}"
            WHERE file = '{file}'
        """.format(dtm='_{}'.format(self.check_dtm) if self.check_dtm else '',
                   ocr_prod='[{}]'.format(data) if data else self.ocr_fir['ocr_prod'], file=self.file_only)

        self.cur.execute(q)
        self.conn.commit()

    def insert_info_db(self, data):
        self.conn = sqlite3.connect(self.db)
        self.cur = self.conn.cursor()
        q = """
            INSERT INTO OCR_FIR{dtm} (file,ocr_size, flt,ocr_prod,ocr_trasp,ocr_racc,ts)
                VALUES ("{file}", "{ocr_size}", "{flt}", "{ocr_prod}", "{ocr_trasp}", "{ocr_racc}", CURRENT_TIMESTAMP)
        """.format(dtm='_{}'.format(self.check_dtm) if self.check_dtm else '',
                   file=self.file_only, ocr_size=data['ocr_size'], flt=self.flt, ocr_prod=data['ocr_prod'],
                   ocr_trasp=data['ocr_trasp'], ocr_racc=data['ocr_racc'])

        self.cur.execute(q)
        self.conn.commit()

    def insert_new_records_table(self, data=[], table='files_WEB', dpi=200, flt=''):
        if self.web:
            q = """
                SELECT * FROM files_WEB WHERE file = '{file}'
            """.format(file=self.file_only)
            # CONTROLLA SE FILE E' GIA' STATO CONSIDERATO
            res = self.cur.execute(q).fetchall()

            q = 'SELECT id FROM files_WEB ORDER BY id DESC LIMIT 1'

            if self.cur.execute(q).fetchall():
                last_id = self.cur.execute(q).fetchall()[0][0]
            else:
                last_id = 0

            new_id = last_id + 1

            if table == 'files_WEB' and not res:
                q = """
                    INSERT INTO files_WEB (id,file,tipologia,produttore,trasportatore,raccoglitore,ts)
                    VALUES ('{id}', '{file}', '{tipologia}', '', '', '', CURRENT_TIMESTAMP)
                """.format(id=new_id, file=self.file_only, tipologia=self.nome_tipologia)
                self.cur.execute(q)

            elif table == 'parole_WEB':
                # CANCELLO PRECEDENTE OCR NON OTTIMALE
                self.delete_table(table='parole_WEB')
                self.logger.info('INSERIMENTO DI {0} RECORDS PER FILE {1}'.format(len(data), self.file_only))
                # NUOVO INSERIMENTO OCR
                for par, (lu, ru, ld, rd), div_x, div_y in data:
                    q = """
                        INSERT INTO parole_WEB (parola, coor_x, coor_y, id_file, div_x, div_y, dpi, flt, ts)
                        VALUES
                            ("{0}", "{1}", "{2}", "{3}", "{4}", "{5}", "{6}", "{7}", CURRENT_TIMESTAMP)
                    """.format(par, lu[0], lu[1], last_id, div_x, div_y, dpi, flt)
                    self.cur.execute(q)

        self.conn.commit()

    def aggiorna_campo_tabella(self, field='', val_field=''):

        q = """
            UPDATE {table}{dtm}
            SET {field} = "{val_field}"
            WHERE file = "{file}"
        """.format(table='files_WEB' if self.web else 'files',
                   dtm='_{}'.format(self.check_dtm) if self.check_dtm else '', field=field,
                   val_field=val_field, file=self.file_only)

        self.cur.execute(q)
        self.conn.commit()

    def get_delim_words(self, info, btw_words, id_st, id_fin):
        delim_words = {}
        len_dw_st = 0
        low = []

        for ii, words_lst in enumerate(btw_words):
            word_like = self.word_like_cond(words_lst)
            if ii == 1:
                # NUMERO PAROLE INIZIALI INTERCETTATE
                len_dw_st = len(delim_words)
            for jj, txt in enumerate(words_lst):
                clike = '(' + ' or '.join(word_like[txt]) + ')'
                q = """
                    SELECT t1.id, parola
                    FROM {table}{dtm} t1
                    LEFT JOIN files_WEB{dtm} t2
                    ON (t1.id_file=t2.id)
                    WHERE
                    file = '{file}' AND
                    {clike}
                    LIMIT 1;
                """.format(table='OCR_{}'.format(INFO_FIR[info.upper()]['TEXT']),
                           dtm='_{}'.format(self.check_dtm) if self.check_dtm else '',
                           file=self.file_only, clike=clike)

                if self.cur.execute(q).fetchall():
                    delim_words[txt] = [(item[0], item[1], '{}'.format('ALTO' if ii == 0 else 'BASSO'))
                                        for item in self.cur.execute(q).fetchall()]
                    if delim_words[txt][0][2] == 'BASSO':
                        low.append(delim_words[txt][0][0])
                elif (not self.cur.execute(q).fetchall()) and (ii == 0) \
                        and (jj == len(btw_words[0]) - 1) and (len(delim_words) == 0):
                    # SE NON TROVO ALCUNA PAROLA CHE IDENTIFICA INIZIO RITAGLIO
                    self.logger.info('NON TROVATA ALCUNA PAROLA CHE IDENTIFICA INIZIO RITAGLIO')
                    self.logger.info('CONSIDERO INIZIO INFO DAL PRIMO ID')
                    delim_words['START_INFO'] = [(id_st, 'NO WORD', 'EOF')]
                elif (not self.cur.execute(q).fetchall()) and (ii == 1) \
                        and (jj == len(btw_words[1]) - 1) and (len(delim_words) - len_dw_st == 0):
                    # SE NON TROVO ALCUNA PAROLA CHE IDENTIFICA FINE RITAGLIO
                    self.logger.info('NON TROVATA ALCUNA PAROLA CHE IDENTIFICA FINE RITAGLIO')
                    self.logger.info("VALUTO FINE INFO INCLUDENDO FINO ULTIMO ID")
                    delim_words['END_INFO'] = [(id_fin, 'NO WORD', 'EOF')]

        self.logger.info('PAROLE CHIAVE TROVATE CHE DELIMITANO INFO {}'.format(delim_words))

        # SE HO PAROLE DI TIPO "BASSO" POSSO FARE RICERCA SUCCESSIVA
        if low:
            minlow = min(low)

            # ESCLUDO CASO IN CUI PAROLA ALTA ABBIA ID MAGGIORE DI QUELLA "BASSA"
            # (ES. "Denominazione" PRESA DA SEZIONE DESTINATARIO E NON PRODUTTORE per 105613_DUG4748772020)
            delim_words_ok = {}

            for txt, info_list in delim_words.items():
                for (w_id, par, cc) in info_list:
                    if cc == 'ALTO' and w_id > minlow:
                        continue
                    delim_words_ok[txt] = [(w_id, par, cc)]
            if not delim_words_ok == delim_words:
                delim_words_ok['START_INFO'] = [(id_st, 'NO WORD', 'BOF')]
            else:
                self.logger.info('PAROLE CHIAVE TROVATE SONO TUTTE ACCETTATE')
                return delim_words_ok
        else:
            self.logger.info('NESSUNA PAROLA INDIVIDUATA COME TIPO "BASSO"')
            self.logger.info('PAROLE CHIAVE TROVATE SONO TUTTE ACCETTATE')
            delim_words_ok = delim_words
            return delim_words_ok

        self.logger.info('PAROLE CHIAVE ACCETTATE CHE DELIMITANO INFO {}'.format(delim_words_ok))
        return delim_words_ok

    def get_ocr_info(self, info):
        id_coor = {}

        # CERCO TRA PAROLA INIZIALE E FINALE PER RICERCA INTERNA DELLA INFO CERCATA
        btw_words = [INFO_FIR[info.upper()]['BTWN_WORD']['INIZ'], INFO_FIR[info.upper()]['BTWN_WORD']['FIN']]

        for ii, words_lst in enumerate(btw_words):

            word_like = self.word_like_cond(words_lst)
            self.logger.info('BTWN WORDS {}'.format(words_lst))
            for txt in words_lst:
                clike = '(' + ' or '.join(word_like[txt]) + ')'
                subq = """
                    {sub_body} WHERE
                    file = '{file}' AND
                    {clike} AND
                    div_y in ({divy})
                    ORDER BY p.id {ord}
                    LIMIT 1;
                """.format(sub_body=self.qy.sub_body, file=self.file_only, clike=clike,
                           divy="'1-4', '2-4'" if (info in ('trasp', 'racc')) or self.rotated_file else "'1-4'",
                           ord='ASC' if info == 'prod' else 'DESC')

                self.logger.debug('FILE {0} RICERCA BASE {1} : {2}'
                                  .format(self.file_only, INFO_FIR[info.upper()]['TEXT'], subq))
                # se parola like dà risultato esco subito
                res = self.cur.execute(subq).fetchall()
                if res:
                    for r in res:
                        id_coor[txt] = (r, '{}'.format('ALTO' if ii == 0 else 'BASSO'))
                        self.logger.info('RISULTATO : {}'.format(id_coor))
                    break
                # avendo parola di tanti caratteri provo a fare like con %
                if len(txt) > 7:
                    plike = 'OR ( parola like "%{s0}%" or parola like "%{s1}%")'.format(s0=txt[:5], s1=txt[-5:])
                    subq = """
                       {sub_body} WHERE
                       file = '{file}' AND
                       ({clike} {plike})
                       ORDER BY p.id ASC
                       LIMIT 1;
                    """.format(sub_body=self.qy.sub_body, file=self.file_only, clike=clike, plike=plike)

                    self.logger.debug('RICERCA PAROLA LUNGA {0} : {1}'.format(INFO_FIR[info.upper()]['TEXT'], subq))
                    res = self.cur.execute(subq).fetchall()
                    if res:
                        for r in res:
                            id_coor[txt] = (r, '{}'.format('ALTO' if ii == 0 else 'BASSO'))
                        break

            if (len(id_coor) == 0 and ii == 0) or (len(id_coor) == 1 and ii == 1):
                self.logger.error('RICERCA ZONA OCR NON CORRETTA')
                try:
                    self.logger.error('TROVATA SOLO PAROLA ZONA INTERMEDIA {}'
                                      .format([v[1] for v in id_coor.values()][0]))
                except Exception:
                    self.logger.info('LISTA VUOTA --> NESSUNA PAROLA TROVATA')
                # SPOSTO IMMAGINE PNG NELLA CARTELLA TIPOLOGIA ASSOCIATA
                # SE IMMAGINE CARICATA DA WEB APP ALLORA RIMUOVO ANCHE SU DIRECTORY ASSOCIATA DI NEW_OCR
                filetosave = os.path.join(PNG_IMAGE_PATH, self.file_only + '.png')
                if os.path.exists(filetosave):
                    self.save_move_delete_png(delete_from_folder=self.nome_tipologia)
                return

        self.logger.info('PAROLE CHE DETERMINANO OCR PER ZONA {0} :\n{1}'
                         .format(INFO_FIR[info.upper()]['TEXT'], id_coor))

        self.logger.info('{0} PRIMO TENTATIVO CREAZIONE RITAGLIO OCR {1} PER FILE {2} {0}'
                         .format('+' * 20, INFO_FIR[info.upper()]['TEXT'], self.file_only))

        words, id_st, id_fin = self.ocr_analysis_ritaglio(info)

        if not words:
            self.logger.info('ANALISI OCR NON HA INDIVIDUATO ALCUNA PAROLA. '
                             'ESECUZIONE FILE {} TERMINATA'.format(self.file_only))
            self.save_move_delete_png(delete_from_folder=self.nome_tipologia)
            return

        self.logger.info('\n{0} RICERCA RITAGLIO {1} {0}\n'.format('#' * 20, INFO_FIR[info.upper()]['TEXT']))
        self.logger.info('RANGE PAROLE INDIVIDUATE {}'.format(words))

        delim_words = self.get_delim_words(info, btw_words, id_st, id_fin)

        self.check_ritaglio(delim_words, info)

        NTENTATIVI = 4

        # SECONDO TENTATIVO OCR RITAGLIO --> ALLARGO RITAGLIO E RIPROVO
        if not self.ocr_fir:
            for itentativo in range(NTENTATIVI):
                # CAMBIO SIZE DEL RITAGLIO
                self.delete_table(table='ocr', info_fir=INFO_FIR[info.upper()]['TEXT'])
                self.logger.info('{0} TENTATIVO NO. {1} PER CREAZIONE RITAGLIO OCR {2} PER FILE {3} {0}'
                                 .format('+' * 20, itentativo + 2, INFO_FIR[info.upper()]['TEXT'], self.file_only))

                # PER MAC OCR TESSERACT -->  'PROD': [0, 0, self.width + 10, 700]
                # PER WIN OCR TESSERACT -->  'PROD': [0, 0, self.width, self.height / 3]
                if itentativo % 2 == 0:
                    itime = 1
                    molt = itentativo // 2
                else:
                    itime = - 1
                    molt = itentativo // 2

                cutoff_width = ((molt + 1) * itime * 5)

                words, id_st, id_fin = self.ocr_analysis_ritaglio(info, cutoff_width=cutoff_width)

                if not words:
                    self.logger.info('ANALISI OCR NON HA INDIVIDUATO ALCUNA PAROLA. '
                                     'ESECUZIONE FILE {} TERMINATA'.format(self.file_only))
                    if os.path.exists(os.path.join(
                            PNG_IMAGE_PATH, self.nome_tipologia, self.file_only + '_PRODUTTORE.png')):
                        os.remove(os.path.join(PNG_IMAGE_PATH, self.nome_tipologia, self.file_only + '_PRODUTTORE.png'))
                    return

                self.logger.info('\n{0} RICERCA RITAGLIO {1} {0}\n'.format('#' * 20, INFO_FIR[info.upper()]['TEXT']))
                self.logger.info('RANGE PAROLE INDIVIDUATE {}'.format(words))

                delim_words = self.get_delim_words(info, btw_words, id_st, id_fin)

                self.check_ritaglio(delim_words, info)
                # SE TROVO RISULTATO ESCO
                if self.ocr_fir:
                    break

        if not self.ocr_fir:
            # NEL CASO NON ABBIA RISULTATI CAMBIO PARAMETRI OCR DI PYTESSEACT (--oem 3 --psm 6)  E RIPETO
            for itent in range(2):
                self.delete_table(table='ocr', info_fir=INFO_FIR[info.upper()]['TEXT'])
                self.logger.info('{0} TENTATIVO NO. {1} CREAZIONE RITAGLIO OCR {2} PER FILE {3} '
                                 'CONFIG OCR DIVERSO {0}'
                                 .format('+' * 20, itent + 1, INFO_FIR[info.upper()]['TEXT'], self.file_only))
                if itent % 2 == 0:
                    itime = 1
                    molt = itent // 2
                else:
                    itime = - 1
                    molt = itent // 2

                cutoff_width = ((molt + 1) * itime * 5)
                words, id_st, id_fin = self.ocr_analysis_ritaglio(info, cutoff_width=cutoff_width,
                                                                  config_ocr=r'--oem 3 --psm 6')

                if not words:
                    self.logger.info('ANALISI OCR NON HA INDIVIDUATO ALCUNA PAROLA. '
                                     'ESECUZIONE FILE {} TERMINATA'.format(self.file_only))

                    if self.rotated_file:
                        orig_filename = self.file_only.split('_rot.png')[0]

                        if os.path.exists(os.path.join(
                                PNG_IMAGE_PATH, self.nome_tipologia, orig_filename + '_PRODUTTORE.png')):
                            os.remove(
                                os.path.join(PNG_IMAGE_PATH, self.nome_tipologia, orig_filename + '_PRODUTTORE.png'))
                    if os.path.exists(os.path.join(
                            PNG_IMAGE_PATH, self.nome_tipologia, self.file_only + '_PRODUTTORE.png')):
                        os.remove(os.path.join(PNG_IMAGE_PATH, self.nome_tipologia, self.file_only + '_PRODUTTORE.png'))
                    return

                self.logger.info('\n{0} RICERCA RITAGLIO {1} {0}\n'
                                 .format('#' * 20, INFO_FIR[info.upper()]['TEXT']))
                self.logger.info('RANGE PAROLE INDIVIDUATE {}'.format(words))

                delim_words = self.get_delim_words(info, btw_words, id_st, id_fin)

                self.check_ritaglio(delim_words, info)
                # SE TROVO RISULTATO ESCO
                if self.ocr_fir:
                    break

        if self.ocr_fir:
            if os.path.exists(os.path.join(
                    PNG_IMAGE_PATH, self.nome_tipologia, self.file_only + '.png')):
                os.remove(os.path.join(PNG_IMAGE_PATH, self.nome_tipologia, self.file_only + '.png'))
            self.check_esclusione_ocr_fir()
            if self.ocr_fir['ocr_prod']:
                self.logger.info('INSERIMENTO COMPLETO IN TABELLA OCR_FIR')
                self.insert_info_db(self.ocr_fir)

            # SE LUNGHEZZA SET PAROLE TROVATE NEL RITAGLIO OCR MAGGIORE > 2 ALLORA CERCO INFO ASSOCIATA CON QUERY
            # for item in ['prod', 'trasp', 'dest']:
            #     if len(self.ocr_fir['ocr_{}'.format(item)][0]) > 2 :
            #         q = """
            #             SELECT a_rag_soc_prod
            #             FROM {table}
            #             WHERE id_fir = '{file}'
            #         """.format(table='INFO_{}'.format(INFO_FIR[item]['TABLE']), file=self.file_only.split('_')[0])
            #         rag_soc = self.cur.execute(q).fetchall()[0][0]
            #         self.produttore = rag_soc

    # def check_ocr_files(self, info_ocr=None):
    #     q = """
    #         SELECT DISTINCT file
    #         FROM "{table}"
    #     """.format(table='files_WEB' if self.web else 'files')
    #
    #     files_lst = []
    #     for row in self.cur.execute(q).fetchall():
    #         files_lst.append(row[0])
    #
    #     tot_files = "'" + "','".join(files_lst) + "'"
    #
    #     q = """
    #         SELECT DISTINCT file
    #         FROM "{table}"
    #         WHERE file in ({tot_files})
    #     """.format(table='OCR_FIR', tot_files=tot_files)
    #
    #     ocr_files_lst = []
    #     for row in self.cur.execute(q).fetchall():
    #         ocr_files_lst.append(row[0])
    #
    #     self.logger.info('{0} {1} FILES ANALIZZATI PER OCR {2} {0}'
    #                      .format('+' * 20, len(ocr_files_lst), INFO_FIR[info_ocr.upper()]['TEXT']))
    #
    #     tot_missed_ocr = len(files_lst) - len(ocr_files_lst)
    #     if not (tot_missed_ocr == 0):
    #         self.logger.info('{0} {1} FILES MANCANTI PER OCR {2} {0}'
    #                          .format('!' * 20, tot_missed_ocr, INFO_FIR[info_ocr.upper()]['TEXT']))
    #         missed_ocr = set(files_lst) - set(ocr_files_lst)
    #         self.logger.info('OCR MANCANTE PER I SEGUENTI FILES: {}'.format(missed_ocr))

    def read_full_info_from_csv(self, info='',):
        full_info_dict = {
            'FIR': {},
            'PRODUTTORI': {},
            'TRASPORTATORI': {},
            'DESTINATARI': {}
        }
        stopwords = []
        with open(os.path.join(PRED_PATH, 'stopwords.txt'), 'r', encoding='utf-8') as f:
            text = f.readlines()
            for t in text:
                stopwords.append(t.replace('\n', ''))
            f.close()

        if not os.path.exists(os.path.join(PRED_PATH, "FULL_INFO_{}.csv".format(info))):
            df = pd.read_csv(os.path.join(PRED_PATH, "INFO_DB_FULL.csv"), encoding='utf-8',
                             error_bad_lines=False, skiprows=1, sep=';')
            logger.info('INDEXES = {}'.format(df.columns))
        else:
            df = pd.read_csv(os.path.join(PRED_PATH, "FULL_INFO_{}.csv".format(info)), encoding='utf-8',
                             error_bad_lines=False, sep=',')

        if info == 'FIR':
            columns = ['id_ordine', 'c_cod_rifiuto', 'd_peso_riscontrato',
                       'b_data_emissione_fir']
            for col in columns:
                if col == 'id_ordine':
                    df[col] = df[col].fillna(0)
                    df[col] = df[col].astype(int)
                    continue
                df[col] = df[col].fillna('')
                df[col] = df[col].astype(str)
                # df[col] = df[col].str.replace('', '') # RIMUOVERE LA STRINGA ' \"" ' (FATTO MANUALMENTE)
            data_prod = {
                'id_fir': df['id_fir'].to_numpy(),
                'id_ordine': df['id_ordine'].to_numpy(),
                'c_cod_rifiuto': df['c_cod_rifiuto'].to_numpy(),
                'd_peso_riscontrato': df['d_peso_riscontrato'].to_numpy(),
                'b_data_emissione_fir': df['b_data_emissione_fir'].to_numpy()
            }
            df_prod = pd.DataFrame(data=data_prod)
            df_prod.to_csv(os.path.join(PRED_PATH, "FULL_INFO_FIR.csv"))
            return None

        if info == 'PRODUTTORI':
            columns = ['c_cod_rifiuto', 'a_piva_prod', 'a_prov_prod', 'a_comune_prod', 'a_via_prod', 'a_cap_prod']
            for col in columns:
                if col in ('a_cap_prod'):
                    df[col] = df[col].fillna(0)
                    df[col] = df[col].astype(int)
                    continue
                df[col] = df[col].fillna('')
                df[col] = df[col].astype(str)
                # df[col] = df[col].str.replace('', '') # RIMUOVERE LA STRINGA ' \"" ' (FATTO MANUALMENTE)
            data_prod = {
                'id_fir': df['id_fir'].to_numpy(),
                'c_cod_rifiuto': df['c_cod_rifiuto'].to_numpy(),
                'a_piva_prod': df['a_piva_prod'].to_numpy(),
                'a_rag_soc_prod': df['a_rag_soc_prod'].to_numpy(),
                'a_prov_prod': df['a_prov_prod'].to_numpy(),
                'a_comune_prod': df['a_comune_prod'].to_numpy(),
                'a_via_prod': df['a_via_prod'].to_numpy(),
                'a_cap_prod': df['a_cap_prod'].to_numpy()
            }
            df_prod = pd.DataFrame(data=data_prod)
            df_prod.to_csv(os.path.join(PRED_PATH, "FULL_INFO_PRODUTTORI.csv"))

            prod_dict = {}
            for col in ['a_rag_soc_prod', 'a_comune_prod', 'a_via_prod']:
                df_prod[col] = df_prod[col].apply(lambda cl: cl.lower())
                val_lst = df_prod[col].values
                words_prod = set()
                for txt in val_lst:
                    # logger.info(txt)
                    words_lst = [re.sub('\W', '', p) for p in word_tokenize(txt) if p not in stopwords
                                 and p not in string.punctuation and not re.search('\d', p)
                                 and not (len(p) <= 2 and re.search('\W', p))]
                    # logger.info(words_lst)
                    words_prod.update(words_lst)

                val_set = set(parola for parola in words_prod)
                foo = [el for el in val_set]
                prod_dict[col] = sorted(foo)
                # logger.info('{0} : {1} - {2}'.format(col, prod_dict[col][-30:-1], len(prod_dict[col])))
                full_info_dict[info][col] = prod_dict[col]
            return full_info_dict

        if info == 'TRASPORTATORI':
            columns = ['c_cod_rifiuto', 'a_piva_trasp', 'a_prov_trasp', 'a_comune_trasp', 'a_via_trasp',
                       'a_cap_trasp']
            for col in columns:
                if col in ('a_cap_trasp'):
                    df[col] = df[col].fillna(0)
                    df[col] = df[col].astype(int)
                    continue
                df[col] = df[col].fillna('')
                df[col] = df[col].astype(str)
                # df[col] = df[col].str.replace('', '') # RIMUOVERE LA STRINGA ' \"" ' (FATTO MANUALMENTE)
            data_trasp = {
                'id_fir': df['id_fir'].to_numpy(),
                'c_cod_rifiuto': df['c_cod_rifiuto'].to_numpy(),
                'a_piva_trasp': df['a_piva_trasp'].to_numpy(),
                'a_rag_soc_trasp': df['a_rag_soc_trasp'].to_numpy(),
                'a_prov_trasp': df['a_prov_trasp'].to_numpy(),
                'a_comune_trasp': df['a_comune_trasp'].to_numpy(),
                'a_via_trasp': df['a_via_trasp'].to_numpy(),
                'a_cap_trasp': df['a_cap_trasp'].to_numpy()
            }
            df_trasp = pd.DataFrame(data=data_trasp)
            df_trasp.to_csv(os.path.join(PRED_PATH, "FULL_INFO_TRASPORTATORI.csv"))

            trasp_dict = {}
            for col in ['a_rag_soc_trasp', 'a_comune_trasp', 'a_via_trasp']:
                df_trasp[col] = df_trasp[col].apply(lambda cl: cl.lower())
                val_lst = df_trasp[col].values
                words_prod = set()
                for txt in val_lst:
                    # logger.info(txt)
                    words_lst = [re.sub('\W', '', p) for p in word_tokenize(txt) if p not in stopwords
                                 and p not in string.punctuation and not re.search('\d', p)
                                 and not (len(p) <= 2 and re.search('\W', p))]
                    # logger.info(words_lst)
                    words_prod.update(words_lst)

                val_set = set(parola for parola in words_prod)
                foo = [el for el in val_set]
                trasp_dict[col] = sorted(foo)
                # logger.info('{0} : {1} - {2}'.format(col, trasp_dict[col][-30:-1], len(trasp_dict[col])))
                full_info_dict[info][col] = trasp_dict[col]
            return full_info_dict

        if info == 'DESTINATARI':
            columns = ['c_cod_rifiuto', 'a_piva_destinatario_fir', 'a_prov_destinatario_fir',
                       'a_comune_destinatario_fir', 'a_via_destinatario_fir', 'a_cap_destinatario_fir']
            for col in columns:
                if col in ('a_cap_destinatario_fir'):
                    df[col] = df[col].fillna(0)
                    df[col] = df[col].astype(int)
                    continue
                df[col] = df[col].fillna('')
                df[col] = df[col].astype(str)
                # df[col] = df[col].str.replace('', '') # RIMUOVERE LA STRINGA ' \"" ' (FATTO MANUALMENTE)
            data_racc = {
                'id_fir': df['id_fir'].to_numpy(),
                'c_cod_rifiuto': df['c_cod_rifiuto'].to_numpy(),
                'a_piva_destinatario_fir': df['a_piva_destinatario_fir'].to_numpy(),
                'a_rag_soc_destinatario_fir': df['a_rag_soc_destinatario_fir'].to_numpy(),
                'a_prov_destinatario_fir': df['a_prov_destinatario_fir'].to_numpy(),
                'a_comune_destinatario_fir': df['a_comune_destinatario_fir'].to_numpy(),
                'a_via_destinatario_fir': df['a_via_destinatario_fir'].to_numpy(),
                'a_cap_destinatario_fir': df['a_cap_destinatario_fir'].to_numpy()
            }
            df_racc = pd.DataFrame(data=data_racc)
            df_racc.to_csv(os.path.join(PRED_PATH, "FULL_INFO_DESTINATARI.csv"))

            racc_dict = {}
            for col in ['a_rag_soc_destinatario_fir', 'a_comune_destinatario_fir', 'a_via_destinatario_fir']:
                df_racc[col] = df_racc[col].apply(lambda cl: cl.lower())
                val_lst = df_racc[col].values
                words_prod = set()
                for txt in val_lst:
                    # logger.info(txt)
                    words_lst = [re.sub('\W', '', p) for p in word_tokenize(txt) if p not in stopwords
                                 and p not in string.punctuation and not re.search('\d', p)
                                 and not (len(p) <= 2 and re.search('\W', p))]
                    # logger.info(words_lst)
                    words_prod.update(words_lst)

                val_set = set(parola for parola in words_prod)
                foo = [el for el in val_set]
                racc_dict[col] = sorted(foo)
                # logger.info('{0} : {1} - {2}'.format(col, racc_dict[col][-30:-1], len(racc_dict[col])))
                full_info_dict[info][col] = racc_dict[col]

            return full_info_dict

    def get_exact_info(self, info):
        if not info == 'FIR':
            if info == 'racc':
                rag = 'a_rag_soc_destinatario_fir'
            else:
                rag = 'a_rag_soc_{}'.format(info)
            q = """
                SELECT {rag}
                FROM {table}
                WHERE id_fir = '{file}'
            """.format(rag=rag, table='INFO_{}'.format(INFO_FIR[info.upper()]['TEXT']),
                       file=self.file_only.split('_')[0])
            if self.cur.execute(q).fetchall():
                info_rag_soc = self.cur.execute(q).fetchall()[0][0]
            else:
                info_rag_soc = 'INFORMAZIONE NON FORNITA'

            return info_rag_soc
        else:
            q = """
                SELECT c_cod_rifiuto, d_peso_riscontrato
                FROM INFO_FIR
                WHERE id_fir = '{file}'
            """.format(file=self.file_only.split('_')[0])
            if self.cur.execute(q).fetchall():
                self.cod_rifiuto = self.cur.execute(q).fetchall()[0][0]
                self.peso_riscontrato = self.cur.execute(q).fetchall()[0][1]
            else:
                self.logger.info('INFORMAZIONE PER FILE {} NON FORNITA DAL DB'.format(self.file_only))

    def check_ritaglio(self, delim_words, info):
        idx = []

        for txt, info_list in delim_words.items():
            for (w_id, par, cc) in info_list:
                idx.append((w_id, cc))

        # METTI AL PRIMO POSTO LA TUPLA CON "START_INFO" SE E' PRESENTE NELLA LISTA IN FONDO
        minidx = min([item[0] for item in idx[:-1]])
        for ii, (id, coo) in enumerate(idx):
            if ii == len(idx) - 1 and id < minidx:
                idx.insert(0, (id, coo))
                del idx[-1]

        r_idx = []
        id_past = (-1, -1)

        for (w_id, cc) in idx:
            # SE DOPO "PRODUTTORE" HO INFO CERCATA ALLORA MI FERMO E NON CERCO "RAGIONE" o "SOCIALE"
            if w_id > (id_past[0] + 1) and id_past[0] > 0:
                # ESCLUDO DAL RANGE (ID_INF, ID_SUP) DOVE ID_INF VIENE PRESA DA "BASSO" e ID_SUP DA "SOPRA"
                if not (id_past[1] != cc and id_past[0] > w_id):
                    self.logger.info('RANGE ID [{}, {}] PER RICERCA INFO'.format(id_past[0], w_id))
                    r_idx.append((id_past[0], w_id))

            id_past = (w_id, cc)

        rwords = []

        for ii, rx in enumerate(r_idx):
            q = """
                SELECT parola
                FROM {table}{dtm} t1
                LEFT JOIN files_WEB{dtm} t2
                ON (t1.id_file=t2.id)
                WHERE
                file = '{file}' AND
                t1.id BETWEEN {rx_st} AND {rx_fin};
            """.format(table='OCR_{}'.format(INFO_FIR[info.upper()]['TEXT']),
                       dtm='_{}'.format(self.check_dtm) if self.check_dtm else '',
                       file=self.file_only, rx_st=rx[0] + 1, rx_fin=rx[1] - 1)

            rwords.append([item[0].lower() for item in self.cur.execute(q).fetchall()])

        self.logger.info('RANGE PAROLE TROVATE ZONA {} : {}'.format(INFO_FIR[info.upper()]['TEXT'], rwords))

        # ok_words = set()
        # for k, lst in self.full_info['PRODUTTORI'].items():
        #     for elem in lst:
        #         if len(elem) > 4 or (len(elem) >= 4 and re.search('[aeiou]$', elem)):
        #             ok_words.add(elem)
        #
        # # AGGIUNGO E RIMUOVO PAROLE DA QUELLE FINORA ACCETTATE
        # self.accepted_words = set(list(set(ok_words) - set(INFO_FIR['PROD']['NO_WORD_OCR']))
        #                      + COMMON_FIR_INFO[self.tipologia])

        # TRATTENGO ALCUNE PAROLE SPECIFICHE
        for rwords_lst in rwords:
            for ii, rword in enumerate(rwords_lst):
                if rword in self.accepted_words:
                    rwords_lst[ii] = rword
                    continue

                if len(rword) == 1 or (rword in INFO_FIR[info.upper()]['NO_WORD_OCR']):
                    # RIMUOVO ELEMENTO MANTENDO POSIZIONE MA INSERENDO STRINGA VUOTA DA ELIMINARE ALLA FINE
                    rwords_lst[ii] = ''
                    continue

                # FAI RICERCA PAROLA LIKE DEI COMMON_FIR_INFO
                # if len(rword) > 3:
                #     common_word_like = self.word_like_cond(rword)
                #     clike = '(' + ' or '.join(common_word_like[rword]) + ')'
                #     q = """
                #         SELECT parola
                #         FROM COMMON_WORDS
                #         WHERE
                #         {clike} AND
                #         tipologia = '{tipologia}'
                #         LIMIT 1;
                #     """.format(clike=clike, tipologia=self.nome_tipologia)
                #     res = self.cur.execute(q).fetchall()
                #     if res:
                #         rword = res[0][0]
                #         rwords_lst[ii] = rword
                #         if rwords_lst[ii] in INFO_FIR[info.upper()]['NO_WORD_OCR']['{}'.format(self.tipologia)]:
                #             rwords_lst[ii] = ''
                #         continue

                rwords_lst[ii] = rword
                if (not (re.search('[aeiou]$', rword) and len(rword) > 3)) \
                        or (re.search('^[jkxyw]', rword)):
                    # RIMUOVO ELEMENTO MANTENDO POSIZIONE MA INSERENDO STRINGA VUOTA DA ELIMINARE ALLA FINE
                    rwords_lst[ii] = ''

        # CONSIDERO DIZIONARIO PAROLE ITALIANE
        d_it = enchant.Dict("it_IT")

        rws = []
        for w_l in rwords:
            if len(w_l) == 1 and w_l[0] == '':
                continue
            foo = set()
            rws.append(foo)
            for rw in w_l:
                if rw:
                    # CERCO SE PAROLA APPARTIENE ALLA LINGUA ITALIANA
                    chk = d_it.check(rw)
                    # self.logger.info('CONTROLLO APPARTENENZA {0} ALLA LINGUA ITALIANA -> {1}'.format(rw, chk))
                    if chk or rw in self.accepted_words:
                        foo.add(rw)

            # SE IL SET RISULTA VUOTO LO CANCELLO
            if len(foo) == 0:
                rws.remove(foo)

        # SE HO DIVERSI SET ALLORA RAGGRUPPO TUTTO IN UNO SOLO CONSIDERANDO TUTTE LE PAROLE DISTINTE
        if len(rws) > 1:
            foo = set()
            rws_agg = [foo]
            for rws_set in rws:
                for rw_el in rws_set:
                    foo.add(rw_el)

            rws = rws_agg

        # SE LA LISTA RISULTA VUOTA NON SCRIVO SUL DB
        if len(rws) == 0:
            self.logger.info('NESSUNA PAROLA INDIVIDUATA. '
                             'NESSUN INSERIMENTO IN OCR {0} PER FILE {1}.'
                             .format(INFO_FIR[info.upper()]['TEXT'], self.file_only))
            self.save_move_delete_png(delete_from_folder=self.nome_tipologia)
            return

        # ELIMINO PAROLE OTTENUTE DA OCR AVENTI UNA LETTERA DIVERSA (ES. codine)
        # RISPETTO ALLE PAROLE NO_WORDS GIA' ELIMINATE
        btw_words = INFO_FIR[info.upper()]['BTWN_WORD']['INIZ'] + INFO_FIR[info.upper()]['BTWN_WORD']['FIN']
        ignored_words = []
        for words_lst in btw_words:
            word_like = self.word_like_cond(words_lst)

            clike = '(' + ' or '.join(word_like[words_lst]) + ')'
            subq = """
                SELECT p.parola
                FROM {table}{dtm} p
                LEFT JOIN files_WEB{dtm} f
                ON (p.id_file=f.id)WHERE
                file = '{file}' AND
                {clike};
            """.format(table='OCR_{}'.format(INFO_FIR[info.upper()]['TEXT']),
                       dtm='_{}'.format(self.check_dtm) if self.check_dtm else '',
                       file=self.file_only, clike=clike)

            # se parola like dà risultato esco subito
            res = self.cur.execute(subq).fetchall()
            if res:
                for item in self.cur.execute(subq).fetchall():
                    ignored_words.append(item[0].lower())

        for el in list(set(ignored_words)):
            if el in rws[0]:
                rws[0].remove(el)

        self.logger.info('RANGE PAROLE SELEZIONATE ZONA {} : {}'.format(INFO_FIR[info.upper()]['TEXT'], rws))
        self.logger.info('\n{0} FINE RICERCA ZONA {1} {0}\n'.format('#' * 20, INFO_FIR[info.upper()]['TEXT']))

        # ELIMINO DICITURA PRODUTTORE PER FILE CHE E' STATO RUOTATO
        if self.rotated_file:
            orig_filename = self.file_only.split('_rot.png')[0]

            if os.path.exists(os.path.join(
                    PNG_IMAGE_PATH, self.nome_tipologia, orig_filename + '.png')):
                os.remove(os.path.join(PNG_IMAGE_PATH, self.nome_tipologia, orig_filename + '.png'))

        self.ocr_fir = {'ocr_prod': None, 'ocr_trasp': None, 'ocr_racc': None, 'ocr_{}'.format(info): rws,
                        'ocr_size': '( {} - {} )'.format(self.crop_width, self.crop_height)}

    def get_grayscale(self, image):
        return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    def remove_noise(self, image, type='', opt=3):
        if type == 'gaussian':
            return cv2.GaussianBlur(image, (7, 7), 3)

        return cv2.medianBlur(image, opt)

    def thresholding(self, image, type='adaptive'):
        if type == 'bin+otsu':
            return cv2.threshold(image, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1]

        thresh = cv2.adaptiveThreshold(image, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2)
        return cv2.bitwise_not(thresh)

    def rotate_file(self, img, rot=0):
        self.logger.info('FILE {0} RUOTATO DI {1} GRADI'.format(self.file_only, rot))
        img_rot = img.rotate(rot, expand=True)
        if not rot == 0:
            width_rot, height_rot = img_rot.size
            self.logger.info('NUOVI VALORI SIZE PER ROTAZIONE : {0} w - {1} h'.format(width_rot, height_rot))
            self.rotated_file = True
        return img_rot

    def delete_table(self, table='', info_fir=''):

        q = """
            SELECT id FROM {table}{dtm}
            WHERE file = "{file}"
        """.format(table='files_WEB' if self.web else 'files',
                   dtm='_{}'.format(self.check_dtm) if self.check_dtm else '', file=self.file_only)
        curr_id = self.cur.execute(q).fetchall()[0][0]

        if table == 'parole_WEB':
            q = """
                DELETE FROM {table}{dtm}
                WHERE id_file = "{id_file}"
            """.format(table='parole_WEB' if self.web else 'parole',
                       dtm='_{}'.format(self.check_dtm) if self.check_dtm else '', id_file=curr_id)

            self.logger.info('ELIMINO RISULTATI PER FILE {0} NELLA TABELLA {1}'
                             .format(self.file_only, 'parole_WEB' if self.web else 'parole'))
            self.cur.execute(q)
            self.conn.commit()

        elif table == 'OCR_FIR':
            q = """
                DELETE FROM {table}{dtm}
                WHERE file = "{file}"
            """.format(table=table,
                       dtm='_{}'.format(self.check_dtm) if self.check_dtm else '', file=self.file_only)

            self.logger.info('ELIMINO RISULTATI PER FILE {0} NELLA TABELLA {1}'.format(self.file_only, table))
            self.cur.execute(q)
            self.conn.commit()

        elif table == 'ocr':
            q = """
                DELETE FROM {table}{dtm}
                WHERE id_file = "{id_file}"
            """.format(table='OCR_{}'.format(info_fir),
                       dtm='_{}'.format(self.check_dtm) if self.check_dtm else '', id_file=curr_id)

            self.logger.info('ELIMINO RISULTATI PER FILE {0} NELLA TABELLA {1}'.format(self.file_only,
                                                                                       'OCR_{}'.format(info_fir)))
            self.cur.execute(q)
            self.conn.commit()


def underscore_split(file):
    occ = file.count("_")
    if occ >= 2:
        file = "_".join(file.split("_", 2)[:2])

    if re.search('\W', file):
        file = re.sub('\W', '', file)

    return file


def process_image(curr_file):
    filepath = os.path.join(IMAGE_PATH, curr_file)
    newfilepath = os.path.join(PNG_IMAGE_PATH, curr_file)
    try:
        logger.info('INIZIALMENTE VERIFICO SE FILE {0}.png ESISTE IN {1}'.format(curr_file, PNG_IMAGE_PATH))
        img = Image.open(newfilepath + '.png')
    except FileNotFoundError:
        # ACCETTO DIVERSI FORMATI DI IMMAGINE DA CONVERTIRE IN PDF (PER AUMENTARE DPI)
        for ext in ['jpg', 'jpeg', 'tiff']:
            try:
                logger.info("CERCO FILE {0} IN FORMATO {1}".format(curr_file, ext))
                img = Image.open(filepath + '.' + ext)
            except FileNotFoundError:
                logger.info("FORMATO IMMAGINE {0} PER FILE {1} NON TROVATO".format(ext, curr_file))
                continue

            img.save(filepath + '.pdf')
            # os.remove(filepath + '.' + ext)
            break

        # CONVERTO IN PNG (NECESSARIO PER OCR)
        logger.info("SALVATAGGIO FILE {0} IN FORMATO PNG SU {1}".format(curr_file, PNG_IMAGE_PATH))
        pages = convert_from_path(filepath + '.pdf', DPI)
        for page in pages:
            page.save('{0}.png'.format(newfilepath), 'png')

        os.remove(filepath + '.pdf')
        img = Image.open(newfilepath + '.png')

    return img


# DA MODIFICARE SE VUOI USARLO
def write_info_produttori_to_csv(prod_dict):
    with open(os.path.join(PRED_PATH, 'SORTED_INFO_PROD.txt'), 'w') as f:
        f.write('PAROLE NON BANALI IN ELENCO ALFABETICO PER OGNI CAMPO DEL PRODUTTORE\n')
        for col in ['a_rag_soc_prod', 'a_comune_prod', 'a_via_prod']:
            f.write('\n{0} {1} {0}\n'.format('-' * 20, col))
            f.write('\n{}\n'.format(prod_dict[col]))
            f.write('\n{0} END {1} {0}\n'.format('-' * 20, col))

        f.close()


def write_fir_list_todo():
    with open('TOTAL_FIRLIST.txt', 'w') as f:
        for item in os.listdir(IMAGE_PATH):
            f.write('{}\n'.format(item.split('.jpg')[0]))
        f.close()


def check_duplicate_tipo_a():
    firlist_tipo_a = [
        file.split('.png')[0] for file in os.listdir(os.path.join(PNG_IMAGE_PATH, TIPO_FIR['TIPO_A']['NAME']))
    ]
    single = []
    duplicate = []
    for elem in firlist_tipo_a:
        elem = elem.split('_')[0]
        if elem in single:
            duplicate.append(elem)
        single.append(elem)

    return duplicate


def check_firlist_tipologia(tipo='', ocr_from_tipologia=False, do_ocr=False):
    firlist_tipologia = [
        file.split('.png')[0] for file in os.listdir(os.path.join(PNG_IMAGE_PATH,
                                                                  TIPO_FIR['{}'.format(tipo.upper())]['NAME']))
    ]
    firset_from_db = []
    firset_from_db_bulk = []
    no_ocr_ritaglio = []
    from_folder = []
    todo = []
    diffdb = []
    db = os.path.join(DB_BACKUP_PATH, 'OCR_MT_MERGE_STATIC_CHECK.db')
    logger.info('FIR ANALIZZATI DA DB {}'.format(db))
    conn = sqlite3.connect(db)
    cur = conn.cursor()
    for fir in firlist_tipologia:
        # CONSIDERO QUELLI CHE SONO STATI ANALIZZATI NEL RITAGLIO PRODUTTORE
        if not fir.endswith('_PRODUTTORE'):
            no_ocr_ritaglio.append('_'.join(fir.split('_')[:2]))
        else:
            from_folder.append('_'.join(fir.split('_')[:2]))

    if not tipo == 'NC':
        sub_q = """
            SELECT file FROM (
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
        """
    else:
        sub_q = """
            SELECT file FROM (
            SELECT * FROM files_WEB_20210702 t1
            UNION 
            SELECT * FROM files_WEB_20210708 t1
            UNION 
            SELECT * FROM files_WEB_20210711 t1
            UNION 
            SELECT * FROM files_WEB_20210714 t1
            UNION 
            SELECT * FROM files_WEB_20210715 t1
            ) AS U
        """

    q = """
        {sub_q} WHERE tipologia = '{tipo}'
        ORDER BY file;
    """.format(sub_q=sub_q, tipo=TIPO_FIR['{}'.format(tipo.upper())]['NAME'])
    for item in cur.execute(q).fetchall():
        firset_from_db.append('_'.join(item[0].split('_')[:2]))
        firset_from_db_bulk.append('_'.join(item[0].split('_')[:1]))
    conn.close()
    for elem in from_folder:
        if elem not in firset_from_db:
            todo.append(elem)
    for elem in firset_from_db:
        if elem not in from_folder:
            diffdb.append(elem)
    logger.info('FIRSET FROM DB : {}'.format(len(firset_from_db)))
    logger.info('FROM FOLDER : {0} {1}'.format(len(from_folder), from_folder[:10]))
    logger.info('FROM FOLDER {0} = {1} FIR DI TIPO {2} TROVATI'
                .format(tipo.upper(), len(firlist_tipologia), TIPO_FIR['{}'.format(tipo.upper())]['NAME']))
    logger.info('NO OCR RITAGLIO : {}'.format(len(no_ocr_ritaglio)))
    logger.info('FROM FOLDER =  FROM FOLDER {0} - NO OCR RITAGLIO --> {1}'.format(tipo.upper(), len(from_folder)))
    logger.info('FIR TODO = FROM FOLDER - FIRSET FROM DB : {0} --> {1}'.format(len(todo), todo))
    logger.info('FIR DIFFDB = FIRSET FROM DB - FROM FOLDER : {0} --> {1}'.format(len(diffdb), diffdb))
    if ocr_from_tipologia:
        return firlist_tipologia
    # NEL CASO NON ABBIA FIR NELLA CARTELLA ASSOCIATA O NC ALLORA CERCO FILE DIRETTAMENTE DA FIR BULK
    if do_ocr:
        ocr_list = []
        all_fir = [item.split('.jpg')[0] for item in os.listdir(IMAGE_PATH)]
        for elem in all_fir:
            elem_rid = '_'.join(elem.split('_')[:1])
            if elem_rid in firset_from_db_bulk:
                ocr_list.append(elem)

        logger.info('ESEGUO {0} OCR PER TIPOLOGIA {1}'.format(len(ocr_list), tipo.upper()))

        return ocr_list
    return diffdb


if __name__ == '__main__':
    logger.info('{0} INIZIO ESECUZIONE SCANSIONE FORMULARI RIFIUTI {0}'.format('!' * 20))
    start_time = time.time()

    # FACCIO PARTIRE I PRIMI 1000 DEI FIR CARTELLA "BULK"
    # RIMUOVI OPPURE MANTIENI ESTENSIONE FILE IN load_files_tmp!!
    # listfir_todo = check_remaining_firlist_todo()
    # PORTARE I FIR TIPO_A DA NC NELLA PROPRIA CARTELLA CON PROCESSO AUTOMATIZZATO
    # FAI CHECK DB STATIC E QUELLI MODIFICATI CON FUNZIONE CHECK_FIRLIST_TIPO_A()
    # CONSIDERA OCR_MT_CHECK_TIPOA_20210712 PER INSERIMENTO A DB MERGE
    listfir_todo = check_firlist_tipologia(tipo='tipo_a', ocr_from_tipologia=True)
    # check_firlist_tipo_nc()
    # listfir_todo = os.listdir(os.path.join(PNG_IMAGE_PATH, 'NC'))[12:]
    # FAI CHECK TIPOLOGIA (FIR - TRS, NIECO) NEL DB STATIC_CHECK CON CODICE CHECK TIPOLOGIA
    listfir_todo = ['101578_DUG473036-2020'] #[item for item in listfir_todo]
    # [item for item in listfir_todo]
    #logger.info(len(listfir_todo))
    # load_files_tmp = listfir_todo # os.listdir(IMAGE_PATH)[1990:1992]#enumerate(os.listdir(IMAGE_PATH))
    load_files = []
    for elem in listfir_todo:  # listfir_todo[:1300]:
        load_files.append(elem.split('.jpg')[0])
    files = []
    # SE FILES CARICATI HANNO TANTI "_" ALLORA NE CONSIDERO SOLO UNO PER SEMPLICITA'
    for load_file in load_files:
        logger.info("VERIFICO IDENEITA' CARATTERI PER FILE {}".format(load_file))
        load_accepted_file = underscore_split(load_file)
        logger.info('NOME FILENAME ACCETTATO --> {}'.format(load_accepted_file))
        load_accepted_file = load_accepted_file + '.jpg'
        # POTREBBE ESSERE BLOCCATO IN CARTELLA IMAGES IN FORMATO PNG E QUINDI DARE ERRORE
        # LO VERIFICO POI ATTRAVERSO FUNZIONE "PROCESS_IMAGE()"
        # CONSIDERA RITAGLIO PARTE SUPERIORE PER FIR RUOTATI
        try:
            logger.info('FILE RINOMINATO DA {0} A {1} IN {2}'.format(load_file, load_accepted_file, IMAGE_PATH))
            os.rename(os.path.join(IMAGE_PATH, load_file + '.jpg'), os.path.join(IMAGE_PATH, load_accepted_file))
        except FileNotFoundError:
            logger.info('FILE NON TROVATO IN {}'.format(IMAGE_PATH))
            logger.info('CERCO {0}.png in {1}'.format(load_accepted_file.split('.')[0], PNG_IMAGE_PATH))
        finally:
            files.append(load_accepted_file)

    for file in files:
        if file == '.DS_Store':
            continue
        try:
            file_only = file.split('.')[0]
            file = os.path.join(IMAGE_PATH, file)
            logger.info('\n{0} ANALISI FILE {1} {0}\n'.format('-o' * 20, file_only))
            process_image(file_only)
            file_png = os.path.join(PNG_IMAGE_PATH, file_only + '.png')
            info = GetFirOCR(file_png, logger=logger, web=True)
            ocr_fir = info.check_from_old_db()
            if not ocr_fir:
                ocr_fir = info.perform_ocr_fir()

            # info.select_prod()
            logger.info('\n{0} SOMMARIO FILE {1} {0}\n'.format('@' * 20, info.file_only))
            logger.info("\nFILE {0} : {1} {2}\n".format(info.file_only, info.nome_tipologia, ocr_fir))
            logger.info('\n{0} FINE SOMMARIO FILE {1} {0}\n'.format('@' * 20, info.file_only))
            if os.path.exists(os.path.join(info.file)):
                os.remove(info.file)
            # if ii == len(os.listdir(IMAGE_PATH)) - 1:
            #     info.check_ocr_files(info_ocr='prod')
        except Exception as e:
            logger.info(logging.exception('ERROR MESSAGE:'))
            with open(log_error_path, "a") as logf:
                logf.write('ERROR REPORT DATETIME {}'.format(now))
                logf.write('{0} ERROR IN FILE {1} {0}\n'.format('-' * 20, file_only))
                logf.write('{}\n'.format(str(traceback.extract_tb(e.__traceback__))))
                # logf.write('{}\n'.format(logging.exception('ERROR MESSAGE:')))
                logf.write("Exception - {0}\n".format(str(e)))
                logf.close()

    logger.info('{0} FILES PROCESSATI IN {1} SECONDI'.format(len(files), time.time() - start_time))
    logger.info('{0} ESECUZIONE SCANSIONE FORMULARI RIFIUTI TERMINATA {0}'.format('!' * 20))
