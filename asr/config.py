HOST = "0.0.0.0"
PORT = 2701

DEVICE = "cpu"
SAMPLE_RATE = 16000
FRAME_LEN = 40.0
CHUNK_SIZE = int(FRAME_LEN*SAMPLE_RATE)
TIME_STEP = 0.04
TIME_PAD = 1
NUM_WORKERS = 4
DELAY_WORKERS = 0.001
CUT_NOISE = False

MODEL_PATH="ru_quartznet.nemo"
LM_PATH="ru_lm.bin"
VOCAB = [' ', 'а', 'б', 'в', 'г', 'д', 'е', 'ж', 'з', 'и', 'й', 'к', 'л', 'м', 'н', 'о', 'п', 'р', 'с', 'т', 'у', 'ф', 'х', 'ц', 'ч', 'ш', 'щ', 'ъ', 'ы', 'ь', 'э', 'ю', 'я', '_']
ALPHA = 0.5
BETA = 0.5
BEAM_WIDTH = 128
BLANK_ID = 33
CALC_CONF = False

MODEL_PUNCT_PATH="v2_4lang_q.pt"
