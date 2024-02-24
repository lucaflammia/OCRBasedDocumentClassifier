import uuid
import os
import sys
import re

if sys.platform == 'win32':
    sys.path.append('C:\\Users\\Utente\\Documents\\Multitraccia\\Progetti\\Cobat\\OCR_development\\NEW_OCR')
    sys.path.append('C:\\Users\\Utente\\Documents\\Multitraccia\\Progetti\\Cobat\\OCR_development\\NN')
else:
    sys.path.append("/Users/analisi/Luca/OCR_dev/OCR_DETECTION")
    sys.path.append("/Users/analisi/Luca/OCR_dev/NN_CREATION")

import main_OCR as model_OCR
import main_NN as model_NN
from pdf2image import convert_from_path
from PIL import Image, ImageFile
import config as config_app
# from model_OCR import *
import OCR_DETECTION.conf_OCR as conf_OCR
# from NEW_OCR.conf_OCR import *

from flask import Flask, jsonify, render_template, request
import logging

os.environ['TF_XLA_FLAGS'] = '--tf_xla_enable_xla_devices'

IMAGE_PATH_BULK = config_app.IMAGE_PATH_BULK

ARCH_PATH = config_app.ARCH_PATH
STATIC_PATH = config_app.STATIC_PATH
LOGFILE = config_app.LOGFILE
PRED_PATH = config_app.PRED_PATH
PNG_IMAGE_PATH = conf_OCR.PNG_IMAGE_PATH
PNG_IMAGE_PATH_APP = config_app.PNG_IMAGE_PATH_APP

if sys.platform == 'win32':
    OCR_PATH = os.path.abspath("C:\\Users\\Utente\\Documents\\Multitraccia\\Progetti\\Cobat"
                               "\\OCR_development\\NEW_OCR")
else:
    OCR_PATH = os.path.abspath("/Users/analisi/Luca/OCR_dev/OCR_DETECTION")

format='%(asctime)s : %(name)s : %(levelname)s : %(message)s'
logging.basicConfig(format=format)
logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)
output_file_handler = logging.FileHandler(os.path.join(ARCH_PATH, LOGFILE), mode='w', encoding='utf-8')
formatter = logging.Formatter(format)
output_file_handler.setFormatter(formatter)
logger.addHandler(output_file_handler)

# Istanzio applicazione Flask
app = Flask(__name__)
app.secret_key = "s3cr3t"
app.debug = False


@app.route("/", methods=["GET"])
def index():
    remove_id_files()
    return render_template("layouts/index.html", nr_app_run=0)


@app.route("/loadedFIR=<nr_app_run>", methods=["GET"])
def index_reset(nr_app_run):
    remove_id_files()
    return render_template("layouts/index.html", nr_app_run=nr_app_run)


@app.route("/log_report/", methods=["GET"])
def log_report():
    title = "Log Report"
    with open("archive/{0}".format(LOGFILE), 'r') as f:
        text = f.readlines()

    # INSERIMENTO SOLO NUOVI LOGFILES
    check_txt = []
    unique_text = []
    chk = True
    for txt in text:
        if txt in check_txt:
            chk = False
            continue
        if '+' in txt:
            check_txt.append(txt)
        elif '+' not in txt and not chk:
            continue
        else:
            chk = True
        unique_text.append(txt)

    return render_template("layouts/log_report.html", title=title, text=unique_text)


@app.route("/results/id=<unique_id>&loadedFIR=<nr_app_run>", methods=["GET"])
def result_for_uuid(unique_id, nr_app_run):
    title = "Scansione FIR"
    loaded_file = get_file_content(get_filename(unique_id))
    load_accepted_file_no_ext = underscore_split(loaded_file)
    load_accepted_file = load_accepted_file_no_ext + '.jpg'
    logger.info('{0} NUOVA SCANSIONE PER FILE {1} {0}'.format('!' * 20, loaded_file))
    logger.info('RINOMINATO FILE DA {0} A {1}'.format(loaded_file + '.jpg', load_accepted_file))
    os.rename(os.path.join(IMAGE_PATH_BULK, loaded_file + '.jpg'),
              os.path.join(IMAGE_PATH_BULK, load_accepted_file))
    file_only = load_accepted_file.split('.')[0]
    model_OCR.process_png_image(file_only)
    file_png = os.path.join(PNG_IMAGE_PATH, file_only + '.png')
    info = model_OCR.GetFirOCR(file_png, logger=logger, web=True)
    ocr_fir = info.check_from_old_db()
    if not ocr_fir:
        ocr_fir = info.perform_ocr_fir()
    logger.info('\n{0} SOMMARIO FILE {1} {0}\n'.format('@' * 20, info.file_only))
    logger.info("\nFILE {0} : {1} {2}\n".format(info.file_only, info.nome_tipologia, ocr_fir))
    logger.info('\n{0} FINE SOMMARIO FILE {1} {0}\n'.format('@' * 20, info.file_only))

    # SE FIR CARICATO HA UNA TIPOLOGIA E HA PAROLE OCR NELLA SEZIONE PRODUTTORE ALLORA COSTRUISCO RETE NEURALE
    if info.nome_tipologia != 'NC' and ocr_fir['ocr_prod']:
        # PARAMETRI COSTRUZIONE RETE NEURALE
        nn_param_setup = {
            'PERC TRAIN VAL SET': [50, 100],
            'HIDDEN NODES': [64],
            'EPOCHS': [150]
        }
        print(eval(nr_app_run))
        if eval(nr_app_run) > 1:
            build_model_from_app = False
        else:
            build_model_from_app = True

        nn = model_NN.GetFirNN(info_loaded_file=(load_accepted_file_no_ext, ocr_fir['ocr_prod'], info.nome_tipologia),
                               logger=logger, type_nn='keras',
                               build_model_from_app=build_model_from_app)
        nn.perc_train_val_set = nn_param_setup['PERC TRAIN VAL SET'][0]
        nn.hidden_nodes = nn_param_setup['HIDDEN NODES'][0]
        nn.epochs = nn_param_setup['EPOCHS'][0]

        stat_nn_50perc, equality = nn.building_nn()
        df = model_NN.get_dataframe(nn_res=stat_nn_50perc)
        nn.write_files_nn(nn_res=stat_nn_50perc, equality=equality, df=df)

        nn.perc_train_val_set = nn_param_setup['PERC TRAIN VAL SET'][1]

        stat_nn_100perc, equality = nn.building_nn()
        df = model_NN.get_dataframe(nn_res=stat_nn_100perc)
        nn.write_files_nn(nn_res=stat_nn_100perc, equality=equality, df=df)

    details = {
        'FILE': info.file_only,
        'TIPOLOGIA OSSERVATA': info.nome_tipologia,
        'TIPOLOGIA PREDETTA 50% TRAIN/VAL':
            stat_nn_50perc[load_accepted_file_no_ext]['TIPOL_PRED_LIST']
            if info.nome_tipologia != 'NC' and ocr_fir['ocr_prod'] else 'NON DATO',
        'TIPOLOGIA PREDETTA 100% TRAIN/VAL':
            stat_nn_100perc[load_accepted_file_no_ext]['TIPOL_PRED_LIST']
            if info.nome_tipologia != 'NC' and ocr_fir['ocr_prod'] else 'NON DATO',
        'OCR PROD': ocr_fir,
        'PROD': [info.produttore], #[info.produttore, info.ris_prod],
        'TRASP': [info.trasportatore], #[info.trasportatore, info.ris_trasp],
        'DEST': [info.raccoglitore],
        'DATA EMISSIONE FIR': info.data_emissione,
        'DATA INIZIO TRASP': info.data_inizio_trasp,
        'DATA FINE TRASP': info.data_fine_trasp,
        'CODICE RIFIUTO': [info.cod_rifiuto],
        'DEST RIFIUTO': info.dest_rifiuto,
        'COD DEST RIFIUTO': info.cod_dest_rifiuto,
        'STATO RIFIUTO': info.stato_rifiuto,
        'PESO RISCONTRATO': info.peso_riscontrato
    }
    if os.path.exists(os.path.join(conf_OCR.BASEPATH, info.file_only + '.png')):
        os.remove(conf_OCR.BASEPATH, info.file_only + '.png')
    if details['OCR PROD']:
        check_files = [info.file_only + '_PRODUTTORE.png', info.file_only + '.png']
        for check_file in check_files:
            orig_file_path = os.path.join(OCR_PATH, 'images', info.nome_tipologia, check_file)
            if os.path.exists(orig_file_path):
                get_ritaglio_fir(info.file_only, orig_file_path)
    # crop_list = crop_image(file_png, img)
    # get_info_csv(details)
    return render_template("layouts/result.html", title=title, details=details)


@app.route("/postmethod_reset", methods=["POST"])
def post_javascript_data_reset():
    jsdata = request.form.get("data")
    nr_app_run = jsdata.split('"nr_app_run":')[1].replace('}', '')
    nr_app_run = int(nr_app_run)
    logger.info('PRE APP RESET RUN {}'.format(nr_app_run))
    nr_app_run += 1
    logger.info('POST APP RESET RUN {}'.format(nr_app_run))
    params = {'nr_app_run': nr_app_run}
    return jsonify(params)


@app.route("/postmethod", methods=["POST"])
def post_javascript_data():
    jsdata = request.form.get("img_data")
    logger.info('REQUEST {} {}'.format(request.form.get("img_data"), type(request.form.get("img_data"))))
    file = jsdata.split(',')[2]
    nr_app_run = jsdata.split('"nr_app_run"')[1]
    nr_app_run = re.sub('[":{}]', '', nr_app_run)
    logger.info('nr_run {}'.format(nr_app_run))
    if nr_app_run == '""':
        nr_app_run = 0
    else:
        nr_app_run = int(nr_app_run)
    logger.info('PRE APP RUN {}'.format(nr_app_run))
    nr_app_run += 1
    logger.info('POST APP RUN {}'.format(nr_app_run))
    occ = file.count(".")
    if occ >= 2:
        file = ".".join(file.split(".", occ)[:occ]).replace('"', '')
    else:
        file = file.split('.')[0].replace('"', '')
    unique_id = create_csv(file)
    params = {"unique_id": unique_id, 'nr_app_run': nr_app_run}
    return jsonify(params)


def underscore_split(file):
    occ = file.count("_")
    if occ >= 2:
        file = "_".join(file.split("_", 2)[:2])

    if re.search('\W', file):
        file = re.sub('\W', '', file)

    return file


def process_image(curr_file):
    filepath = os.path.join(IMAGE_PATH_BULK, curr_file)
    newfilepath = os.path.join(PNG_IMAGE_PATH_APP, curr_file)
    try:
        # logger.info('CERCO FILE {0}.png IN {1}'.format(curr_file, PNG_IMAGE_PATH_APP))
        img = Image.open(newfilepath + '.png')
    except FileNotFoundError:
        # ACCETTO DIVERSI FORMATI DI IMMAGINE DA CONVERTIRE IN PDF (PER AUMENTARE DPI)
        for ext in ['jpg', 'jpeg', 'tiff']:
            try:
                # logger.info("Cerco file {0} in formato {1} su {2}".format(curr_file, ext, filepath))
                img = Image.open(filepath + '.' + ext)
            except FileNotFoundError:
                # logger.info("Formato immagine {0} per file {1} non trovato".format(ext, curr_file))
                continue

            img.save(filepath + '.pdf')
            # os.remove(filepath + '.' + ext)
            break

        # CONVERTO IN PNG (NECESSARIO PER OCR)
        DPI = 200
        # logger.info("Salvataggio file {0} in formato png su {1}".format(curr_file, PNG_IMAGE_PATH_APP))
        pages = convert_from_path(filepath + '.pdf', DPI)
        for page in pages:
            page.save('{0}.png'.format(newfilepath), 'png')
            # SALVATAGGIO ANCHE SU DIRECTORY NEW_OCR
            logger.info('SALVATAGGIO FIR SU DIRECTORY IMAGES NEW_OCR')
            newfilepath_ocr = os.path.join(conf_OCR.BASEPATH, 'images', curr_file)
            logger.info('NEW FILE PATH OCR {}'.format(newfilepath_ocr))
            page.save('{0}.png'.format(newfilepath_ocr), 'png')

        os.remove(filepath + '.pdf')
        img = Image.open(newfilepath + '.png')

    return img


def get_ritaglio_fir(file, file_orig_path):
    ImageFile.LOAD_TRUNCATED_IMAGES = True
    img = Image.open(file_orig_path)
    img_copy = img.copy()
    img.close()
    # img_crop = img_copy.crop((0, 0, right, bottom))
    filepath = os.path.join(STATIC_PATH, 'screenshots', file + '_PRODUTTORE.png')
    img_copy.save(filepath, 'png')


def crop_image(file, img):
    width, height = img.size
    # Creazione ritagli
    npart = 12
    left = 0
    right = width

    screen_dir = os.path.join(STATIC_PATH, 'screenshots')

    if not os.path.exists(screen_dir):
        os.makedirs(screen_dir)

    crop_list = []
    for i, ip in enumerate([npart - 2, npart - 1, npart]):
        top = (ip - 1) * height / npart
        bottom = ip * height / npart
        img_crop = img.crop((left, top, right, bottom))
        cfilepath = os.path.join(screen_dir, 'CROP_{0}_{1}.png'.format(file, i + 1))
        img_crop.save("{}".format(cfilepath), 'png')
        crop_list.append((i + 1, 'CROP_{0}_{1}.png'.format(file, i + 1)))
    return crop_list


def remove_id_files():
    for id_files in os.listdir(ARCH_PATH):
        if '-' in id_files and id_files.endswith('.csv'):
            path_id = os.path.join(ARCH_PATH, id_files)
            os.remove(path_id)


def create_csv(filename):
    unique_id = str(uuid.uuid4())
    with open(get_filename(unique_id), "a") as file:
        file.write(filename)
    return unique_id


# def get_info_csv(info_d):
#     text = "FILE : {0} \nPRODUTTORE : {1} \nTRASPORTATORE : {2} \nDESTINATARIO : {3}"\
#         .format(info_d['FILE'], info_d['PROD'][0], info_d['TRASP'][0], info_d['DEST'][0])
#     with open(get_filename(info_d['FILE']), "w") as file:
#         file.write(text)


def get_filename(filename, folder='archive', ext='csv'):
    return f"{folder}/{filename}.{ext}"


def get_file_content(filename):
    with open(filename, "r") as file:
        return file.read()


if __name__ == '__main__':
    app.run()
