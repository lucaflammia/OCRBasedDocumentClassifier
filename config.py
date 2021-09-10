import sqlite3
import os
from datetime import datetime

now = datetime.now()
date_time = now.strftime("%Y-%m-%d %H-%M-%S")

BASEPATH = os.path.abspath(".")
PRED_PATH = os.path.abspath("../../")
IMAGE_PATH_BULK = os.path.join(PRED_PATH, 'FIR_BULK', 'jpg_archive')
PNG_IMAGE_PATH_APP = os.path.join(BASEPATH, 'images')
STATIC_PATH = os.path.join(BASEPATH, 'static')

if not os.path.exists(os.path.join(BASEPATH, "archive")):
    os.makedirs(os.path.join(BASEPATH, "archive"))

ARCH_PATH = os.path.join(BASEPATH, "archive")
# LOGFILE = "LOG_DEMO_OCR.log"
LOGFILE = "LOG_API_FIR_{}.log".format(date_time)

INFO_FIR = {
    'PROD': {
        'TEXT': 'PRODUTTORE',
        'TABLE': 'PRODUTTORI',
        'BTWN_WORD': {
            'INIZ': ['detentore'],
            'FIN': ['locale']
        },
        'NO_WORD': ['produttore', 'detentore', 'registro', 'numero',
                    'ragione', 'sociale', 'denominazione', 'unita']
    },
    'TRASP': {
        'TEXT': 'TRASPORTATORE',
        'TABLE': 'TRASPORTATORI',
        'BTWN_WORD': {
            'INIZ': ['trasportatore'],
            'FIN': ['indirizzo']
        },
        'NO_WORD': ['trasportatore', 'rifiuto', 'ragione', 'sociale', 'denominazione', 'luogo', 'indirizzo']
    },
    'RACC': {
        'TEXT': 'DESTINATARIO',
        'TABLE': 'RACCOGLITORI',
        'BTWN_WORD': {
            'INIZ': ['destinatario'],
            'FIN': ['destinazione']
        },
        'NO_WORD': ['albo', 'ragione', 'sociale', 'denominazione', 'luogo', 'destinazione', 'destinatario']
    }
}

TIPO_A = {
    'TEXT': ["formulario", "rifiuti"],
    'NO_WORD': ["identificazione"],
    'SIGN': ["<", "<"],
    'NAME': 'FORMULARIO RIFIUTI - ALLEGATO B',
    'FILES': []
}

TIPO_C = {
    'TEXT': ["recuperi", "ecol"],
    'NO_WORD': ['denominazione'],
    'SIGN': ["<", "<"],
    'NAME': 'FORMULARIO PULI ECOL',
    'FILES': []
}

TIPO_F = {
    'TEXT': ["itrofer", "circolare"],
    'NO_WORD': [],
    'SIGN': ["<", "<"],
    'NAME': 'FORMULARIO ITROFER',
    'FILES': []
}

NC = {
    'NC': []
}

# TIPO_B = {
#     'TEXT': ["recycling", "systems"],
#     'SIGN': ["<", "<"],
#     'FILES': []
# }

# TIPO_C = {
#     'TEXT': ["trasporto", "rifiuti"],
#     'SIGN': [">", "<"],
#     'FILES': []
# }

# TIPO A: TEMPLATE PER FIR "FORMULARIO RIFIUTI"
# TIPO B: TEMPLATE PER FIR "RIFIUTI PIOMBOSI" (SOLO IN ALTO A DESTRA)
# TIPO C: TEMPLATE PER FIR "PULI ECOL RECUPERI" (SOLO IN ALTO A SINISTRA)
# TIPO D: TEMPLATE PER FIR "CONSORZIO NAZIONALE RACCOLTO RICICLO" (SOLO IN ALTO AL CENTRO)
# TIPO E: TEMPLATE PER FIR "VERONICO" (SOLO IN ALTO A SINISTRA)
# TIPO F: TEMPLATE PER FIR "ITROFER CIRCOLARE" (SOLO IN ALTO A SINISTRA)

TIPO_FIR = {
    'TIPO_A': TIPO_A,
    'TIPO_C': TIPO_C,
    'TIPO_F': TIPO_F,
    'NC': NC
}

DLT_ID = 25


class CreazioneDatabase:
    def __init__(self, db, web=False):
        self.conn = sqlite3.connect(db)
        self.cur = self.conn.cursor()
        self.tb_files = """
            CREATE TABLE if not exists files 
            (id INTEGER PRIMARY KEY AUTOINCREMENT, file VARCHAR(50) NOT NULL, 
            tipologia VARCHAR(50) NOT NULL, ts TIMESTAMP);
        """
        self.tb_parole = """
            CREATE TABLE if not exists parole (  
            id INTEGER PRIMARY KEY AUTOINCREMENT, 
            parola VARCHAR(255) NOT NULL, 
            coor_x FLOAT(10,5) NOT NULL, 
            coor_y FLOAT(10,5) NOT NULL, 
            id_file INTEGER NOT NULL, 
            ts TIMESTAMP, 
            FOREIGN KEY (id_file) REFERENCES files (id) );
        """
        self.tb_files_WEB = """
            CREATE TABLE if not exists files_WEB 
            (id INTEGER PRIMARY KEY AUTOINCREMENT, file VARCHAR(50) NOT NULL, 
            tipologia VARCHAR(50) NOT NULL, ts TIMESTAMP);
        """
        self.tb_parole_WEB = """
            CREATE TABLE if not exists parole_WEB (  
            id INTEGER PRIMARY KEY AUTOINCREMENT, 
            parola VARCHAR(255) NOT NULL, 
            coor_x FLOAT(10,5) NOT NULL, 
            coor_y FLOAT(10,5) NOT NULL, 
            id_file INTEGER NOT NULL, 
            div_x VARCHAR(255) NOT NULL, 
            div_y VARCHAR(255) NOT NULL,
            dpi INTEGER NOT NULL, 
            flt VARCHAR(255) NOT NULL, 
            ts TIMESTAMP, 
            FOREIGN KEY (id_file) REFERENCES files_WEB (id) );
        """
        if web:
            self.cur.execute(self.tb_files_WEB)
            self.cur.execute(self.tb_parole_WEB)
        else:
            self.cur.execute(self.tb_files)
            self.cur.execute(self.tb_parole)

        self.conn.commit()

    def __del__(self):
        self.conn.close()


class QueryFir:
    def __init__(self, web=False):
        if web:
            self.body = """
                SELECT parola, coor_x, coor_y, file
                FROM parole_WEB p
                LEFT JOIN "files_WEB" f
                ON (p.id_file=f.id)
            """
            self.sub_body = """
                SELECT p.id
                FROM parole_WEB p
                LEFT JOIN files_WEB f
                ON (p.id_file=f.id)
            """
        else:
            self.body = """
                SELECT parola, coor_x, coor_y, file
                FROM parole p
                LEFT JOIN "files" f
                ON (p.id_file=f.id)
            """
            self.sub_body = """
                SELECT p.id
                FROM parole p
                LEFT JOIN files f
                ON (p.id_file=f.id)
            """
