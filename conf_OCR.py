import sqlite3
import os
from datetime import datetime

now = datetime.now()
date_time = now.strftime("%Y-%m-%d %H-%M-%S")

BASEPATH = os.path.abspath(".")
PRED_PATH = os.path.abspath("..")
DB_PATH = os.path.join(PRED_PATH, 'DEMO_APP', 'WEB_APP')
DB_BACKUP_PATH = os.path.join(BASEPATH, 'DB_BACKUP')
DB_STATIC_PATH = os.path.join(PRED_PATH, 'DB_STATIC')
IMAGE_PATH = os.path.join(PRED_PATH, 'FIR_BULK', 'jpg_archive')

if not os.path.exists(os.path.join(BASEPATH, "archive")):
    os.makedirs(os.path.join(BASEPATH, "archive"))

if not os.path.exists(os.path.join(BASEPATH, "images")):
    os.makedirs(os.path.join(BASEPATH, "images"))

ARCH_PATH = os.path.join(BASEPATH, "archive")
PNG_IMAGE_PATH = os.path.join(BASEPATH, 'images')
LOGFILE = "LOG_NN_{}.log".format(date_time)
LOGFILE_ERROR = "LOG_ERROR_{}.log".format(date_time)
TIPOLOGIA_FIR = "TIPOLOGIA_FIR_LIST_{}.txt".format(date_time)

DPI = 200

INFO_FIR = {
    'PROD': {
        'TEXT': 'PRODUTTORE',
        'TABLE': 'PRODUTTORI',
        'BTWN_WORD': {
            'INIZ': ['detentore', 'produttore', 'denominazione', 'ragione', 'sociale'],
            'FIN': ['unita', 'locale', 'codice', 'fiscale', 'autorizzazione', 'albo', 'destinatario']
        },
        'NO_WORD_OCR': [
            'produttore', 'detentore', 'registro', 'numero', 'luogo', 'destinatario', 'albo', 'trasportatore',
            'ragione', 'sociale', 'codice', 'fiscale', 'denominazione', 'unita', 'locale', 'data', 'sede',
            'indirizzo', 'destinazione', 'rifiuti', 'formulario', 'ministero', 'aree', 'ambiente', 'effettiva',
            'argine', 'pieve', 'regione', 'autorizzazione', 'emissione', 'allegato', 'legale', 'resto', 'alle',
            'maggio', 'febbraio', 'ambiente', 'sensi', 'codi', 'decreto', 'batterie', 'identificazione', 'aprile',
            'obbligatorio', 'rifiuto', 'parte', 'soggetta', 'esauste', 'coordinamento', 'detentore', 'agucchi',
            'destinatario', 'direzione', 'sede', 'trasportatore', 'serie', 'stoccaggio', 'rare', 'provvisorio',
            'amministrativa', 'amministrazione', 'concessionaria', 'formulario', 'registro', 'numero',
            'identificazione', 'imprese', 'arte', 'rifiuto', 'serie', 'detentore', 'registro', 'numero', 'maggio',
            'registro', 'rifiuto', 'identificazione', 'numero', 'serie', 'albo', 'successive', 'modifiche',
            'integrazioni', 'detentore', 'telefono', 'soluzioni', 'conforme', 'tipografia', 'salute', 'tara', 'ragone',
            'bara', 'sino', 'tecnico', 'recale', 'partita', 'sogemont', 'anale', 'entro', 'lario', 'falbo', 'soma'
        ]
    },
    'TRASP': {
        'TEXT': 'TRASPORTATORE',
        'TABLE': 'TRASPORTATORI',
        'BTWN_WORD': {
            'INIZ': ['trasportatore'],
            'FIN': ['indirizzo', 'sociale']
        },
        'NO_WORD_OCR': ['trasportatore', 'rifiuto', 'ragione', 'sociale', 'locale',
                    'unita', 'denominazione', 'luogo', 'indirizzo']
    },
    'RACC': {
        'TEXT': 'DESTINATARIO',
        'TABLE': 'RACCOGLITORI',
        'BTWN_WORD': {
            'INIZ': ['destinatario'],
            'FIN': ['luogo', 'destinazione']
        },
        'NO_WORD_OCR': ['albo', 'ragione', 'sociale', 'denominazione', 'luogo', 'destinazione', 'destinatario']
    }
}

COMMON_NO_WORD = [("torinese", "1-4"), ("pericolosi", "1-4"), ('filippo', "1-4"),
                  ('giovanni', "1-4"), ('viterbo', "1-4"), ("siciliarottami", "1-4"), ("sicilia", "1-4"),
                  ("logistica", "1-4"), ('brugnano', "1-4"), ('cigacars', '1-4')]

TIPO_A = {
    'TEXT': ["formulario", "rifiuti"],
    'SIZE_OCR': [0, 0, 2356, 700],
    'NO_WORD': [("identificazione", "1-4"), ("mondoservizi", "1-4"),
                ('systems', '1-4'), ('balvano', '1-4'), ("ecologiche", "1-4"),
                ("itrofer", "1-4"), ("circolare", "1-4"), ('rosignano', '1-4'),
                ('morelline', '1-4'),] + COMMON_NO_WORD,
    'DLT_ID': 25,
    'NAME': 'FORMULARIO RIFIUTI - ALLEGATO B - ETM',
    'FILES': []
}

TIPO_A_BIS = {
    'TEXT': ["allegato", "formulario"],
    'SIZE_OCR': [0, 0, 2356, 700],
    'NO_WORD': [("identificazione", "1-4"), ("mondoservizi", "1-4"),
                ('systems', '1-4'), ('balvano', '1-4'), ("ecologiche", "1-4"),
                ("itrofer", "1-4"), ("circolare", "1-4"), ('rosignano', '1-4'),
                ('morelline', '1-4'),] + COMMON_NO_WORD,
    'DLT_ID': 25,
    'NAME': 'FORMULARIO RIFIUTI - ALLEGATO B - ETM',
    'FILES': []
}


TIPO_B = {
    'TEXT': ["identificazione", "rimondi"],
    'SIZE_OCR': [0, 300, 2356, 1000],
    'NO_WORD': COMMON_NO_WORD + [('annunziata', "1-4"), ('ecologia', '1-4')],
    'DLT_ID': 45,
    'NAME': 'FIR - COBAT',
    'FILES': []
}

TIPO_C = {
    'TEXT': ["ecologia", "unipersonale"],
    'SIZE_OCR': [0, 650, 2356, 1050],
    'NO_WORD': [('aglioni', "1-4"), ('angelo', "1-4"), ('rosignano', '1-4'), ('morelline', '1-4'),
                ('systems', '1-4'), ('ecologica', '1-4'), ('telefono', '1-4')] + COMMON_NO_WORD,
    'DLT_ID': 25,
    'NAME': 'FIR - TRS',
    'FILES': []
}

TIPO_E = {
    'TEXT': ["diego", "lequile"],
    'SIZE_OCR': [0, 650, 2356, 1050],
    'NO_WORD': COMMON_NO_WORD,
    'DLT_ID': 25,
    'NAME': 'ECOTECNICA',
    'FILES': []
}

TIPO_D = {
    'TEXT': ["recuperi", "severino"],
    'SIZE_OCR': [0, 550, 2356, 1050],
    'NO_WORD': [('denominazione', '1-4')] + COMMON_NO_WORD,
    'DLT_ID': 25,
    'NAME': 'FORMULARIO PULI ECOL',
    'FILES': []
}

# TIPO_F = {
#     'TEXT': ["itrofer", "circolare"],
#     'NO_WORD': [],
#     'SIGN': ["<", "<"],
#     'NAME': 'FORMULARIO ITROFER',
#     'FILES': []
# }

TIPO_G = {
    'TEXT': ["futuro", "sostenibile"],
    'SIZE_OCR': [800, 250, 2356, 750],
    'NO_WORD': COMMON_NO_WORD,
    'DLT_ID': 25,
    'NAME': 'NIECO',
    'FILES': []
}

TIPO_H = {
    'TEXT': ['robiambiente'],
    'SIZE_OCR': [0, 550, 2356, 1050],
    'NO_WORD': COMMON_NO_WORD,
    'DLT_ID': 25,
    'NAME': 'ROBI',
    'FILES': []
}

NC = {
    'TEXT': [],
    'FILES': [],
    'NAME': 'NC'
}

# TIPO_B1 = {
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
# TIPO G: TEMPLATE PER FIR "NIECO"

TIPO_FIR = {
    'TIPO_A': TIPO_A,
    'TIPO_A_BIS': TIPO_A_BIS,
    'TIPO_B': TIPO_B,
    'TIPO_C': TIPO_C,
    'TIPO_D': TIPO_D,
    'TIPO_E': TIPO_E,
#    'TIPO_F': TIPO_F,
    'TIPO_G': TIPO_G,
#    'TIPO_H': TIPO_H,
    'NC': NC
}

common_fir_info_tipo_a = \
    ['srl', 'sas', 'diaz', 'sicon', '36031', 'povolaro', 'dueville', "dellindustria", 'enrico', 'como',
     'vicenza', 'fasano', 'linussio', 'giorgio', 'nogaro', 'san', 'spa', 'udine', 'rps', 'riviera',
     'recuperi', 'autofficina', 'molini', 'quarto', 'daltino', 'venezia', 'vertivsrl', 'ecocentro',
     'porto', 'cavergnago', 'veritas', 'arbe', 'torino', 'romano', 'veritasspa', 'traelet', 'piave',
     'codroipo', 'vertiv', 'tommaso', 'acquasanta', 'salerno', 'laverda', 'breganze', 'stazzi', '22100',
     'piovanelli', 'firenze', 'licata', 'duca', 'giordano', 'casanio', 'bovezzo', 'villastorta',
     'ambiente', 'marosticana', 'vicenza', 'zorza', 'castelverde', 'savoia', 'autoricambi', 'gi', 'due',
     '80005370137', 'bianchi', 'pneumatici', 'deambrosis', 'maurizio', 'san', 'pietro', 'casorzo', '14032',
     'elettrauto', 'auto', 'lunger', 'welschnofen', 'levante', 'nova', 'gerenzano', 'inglesina', '21040',
     'milano', 'snc', 'sea', 'valle', 'camonica', 'rigamonti', 'darfo', 'servizi', 'mollo', 'sonico',
     'isola', 'baiso', 'europa', 'cem', 'pessano', 'bornago', 'caserta']

COMMON_FIR_INFO = {
    'TIPO_A': common_fir_info_tipo_a,
    'TIPO_A_BIS': common_fir_info_tipo_a,
    'TIPO_B': ['newave'],
    'TIPO_C': ['srl', 'cavagna', 'toscanini', 'renato', 'caorso', '43010', 'fontevivo', 'pontetaro',
               '29012', 'maggio', '01103640338', 'ecologia', 'autoservice', 'melissano', 'pontenure', '29010'
               ],
    'TIPO_D': ['autoservice', 'guglielmo', 'nuzzo'],
    'TIPO_E': [],
    'TIPO_F': [],
    'TIPO_G': []
}


class CreateNewDatabase:
    def __init__(self, db, web=False):
        self.conn = sqlite3.connect(db)
        self.cur = self.conn.cursor()
        self.tb_files_WEB = """
            CREATE TABLE if not exists files_WEB
            (id INTEGER PRIMARY KEY AUTOINCREMENT, file VARCHAR(50) NOT NULL,
            tipologia VARCHAR(50) NOT NULL, produttore VARCHAR(50) NOT NULL,
            trasportatore VARCHAR(50) NOT NULL, raccoglitore VARCHAR(50) NOT NULL, ts TIMESTAMP);
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
        self.cur.execute(self.tb_files_WEB)
        self.cur.execute(self.tb_parole_WEB)

        self.tb_OCR_prod = """
            CREATE TABLE if not exists OCR_PRODUTTORE (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            parola VARCHAR(255) NOT NULL,
            id_file INTEGER NOT NULL,
            flt VARCHAR(255) NOT NULL,
            ts TIMESTAMP,
            FOREIGN KEY (id_file) REFERENCES files_WEB (id) );
        """
        self.cur.execute(self.tb_OCR_prod)

        self.tb_OCR_FIR = """
            CREATE TABLE if not exists OCR_FIR (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            file VARCHAR(255) NOT NULL,
            ocr_size VARCHAR(255) NOT NULL,
            flt VARCHAR(255) NOT NULL,
            ocr_prod VARCHAR(255) NOT NULL,
            ocr_trasp VARCHAR(255) NOT NULL,
            ocr_racc VARCHAR(255) NOT NULL,
            ts TIMESTAMP);
        """
        self.cur.execute(self.tb_OCR_FIR)

        self.conn.commit()

    def __del__(self):
        self.conn.close()


class QueryFir:
    def __init__(self, web=False, dtm=''):
        if web:
            self.body = """
                SELECT parola, coor_x, coor_y, file
                FROM "{table1}" p
                LEFT JOIN "{table2}" f
                ON (p.id_file=f.id)
            """.format(table1='parole_WEB_{}'.format(dtm) if dtm else 'parole_WEB',
                       table2='files_WEB_{}'.format(dtm) if dtm else 'files_WEB')
            self.sub_body = """
                SELECT p.id, p.coor_x, p.coor_y
                FROM "{table1}" p
                LEFT JOIN "{table2}" f
                ON (p.id_file=f.id)
            """.format(table1='parole_WEB_{}'.format(dtm) if dtm else 'parole_WEB',
                       table2='files_WEB_{}'.format(dtm) if dtm else 'files_WEB')
        else:
            self.body = """
                SELECT parola, coor_x, coor_y, file
                FROM parole p
                LEFT JOIN "files" f
                ON (p.id_file=f.id)
            """
            self.sub_body = """
                SELECT p.id, p.coor_x, p.coor_y
                FROM parole p
                LEFT JOIN files f
                ON (p.id_file=f.id)
            """

    def load_db(self):
        db_from = os.listdir(DB_BACKUP_PATH)
        return db_from
