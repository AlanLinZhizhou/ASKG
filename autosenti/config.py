import os


FILE_DIR_PATH = os.path.dirname(os.path.abspath(__file__))

KGS = {
    'mr4_addsenti': os.path.join(FILE_DIR_PATH, 'kgs/mr4_addsenti.spo'),
    'AMAN': os.path.join(FILE_DIR_PATH, 'kgs/AMAN.spo'),
    'alm': os.path.join(FILE_DIR_PATH, 'kgs/alm.spo'),
    'alml3': os.path.join(FILE_DIR_PATH, 'kgs/alm.l3.spo'),
    'sst3_addsenti': os.path.join(FILE_DIR_PATH, 'kgs/sst3_addsenti.spo'),
    'sst5_addsenti': os.path.join(FILE_DIR_PATH, 'kgs/sst5_addsenti.spo')

}

MAX_ENTITIES = 2

# Special token words.
PAD_TOKEN = '[PAD]'
UNK_TOKEN = '[UNK]'
CLS_TOKEN = '[CLS]'
SEP_TOKEN = '[SEP]'
MASK_TOKEN = '[MASK]'
ENT_TOKEN = '[ENT]'
SUB_TOKEN = '[SUB]'
PRE_TOKEN = '[PRE]'
OBJ_TOKEN = '[OBJ]'

NEVER_SPLIT_TAG = [
    PAD_TOKEN, UNK_TOKEN, CLS_TOKEN, SEP_TOKEN, MASK_TOKEN,
    ENT_TOKEN, SUB_TOKEN, PRE_TOKEN, OBJ_TOKEN
]
