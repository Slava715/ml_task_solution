from nemo.core.neural_types import NeuralType, AudioSignal, LengthsType
from nemo.core.classes import IterableDataset
from ctcdecode import CTCBeamDecoder
import nemo.collections.asr as nemo_asr

import torch
from torch import package
from torch.utils.data import DataLoader
import numpy as np

from torch.multiprocessing import Process, Queue, Array, Semaphore
from time import sleep
import math
import os

import config

from extractor import NumberExtractor
EXTRACTOR = NumberExtractor()

MODEL_PUNCT = package.PackageImporter(config.MODEL_PUNCT_PATH).load_pickle("te_model", "model")

class AudioDataLayer(IterableDataset):
    @property
    def output_types(self):
        return {
            'audio_signal': NeuralType(('B', 'T'), AudioSignal(freq=self._sample_rate)),
            'a_sig_length': NeuralType(tuple('B'), LengthsType()),
        }

    def __init__(self, sample_rate):
        super().__init__()
        self._sample_rate = sample_rate
        self.output = True
        
    def __iter__(self):
        return self
    
    def __next__(self):
        if not self.output:
            raise StopIteration
        self.output = False
        return self.signal, self.signal_shape 
        
    def set_signal(self, signal):
        self.signal = signal
        self.signal_shape = torch.as_tensor(len(signal), dtype=torch.int64)
        self.output = True

    def __len__(self):
        return 1
        
        
class FrameASR:
    
    def __init__(self, frame_len=config.FRAME_LEN, model_path=config.MODEL_PATH, lm_path=config.LM_PATH):
    
        self.device = config.DEVICE
        self.asr_model = nemo_asr.models.EncDecCTCModel.restore_from(model_path, map_location=self.device)
        self.asr_model.eval();
        
        self.data_layer = AudioDataLayer(sample_rate=self.asr_model.cfg.preprocessor.sample_rate)    
        self.data_loader = DataLoader(self.data_layer, batch_size=1, collate_fn=self.data_layer.collate_fn)
        
    def _decode(self, frame):
        logits = self.infer_signal(frame).cpu().numpy()[0]
        logits = logits[:-1]
        return logits
    
    @torch.no_grad()
    def transcribe(self, frame=None):
        return self._decode(frame)
        
    def infer_signal(self, signal):
        self.data_layer.set_signal(signal)
        batch = next(iter(self.data_loader))
        audio_signal, audio_signal_len = batch
        
        audio_signal, audio_signal_len = audio_signal.to(self.device), audio_signal_len.to(self.device)
        log_probs, encoded_len, predictions = self.asr_model.forward(input_signal=audio_signal, input_signal_length=audio_signal_len)
        return log_probs
        
        
def asr_process(num_worker):   
    asr = FrameASR()
    
    while True:       
        sleep(config.DELAY_WORKERS)
        
        if WORKER_IN[num_worker].empty() == False:
            file_np = WORKER_IN[num_worker].get()
            file_tensor = torch.FloatTensor(file_np)
            
            sum_frame = []
            sum_empty = True
            
            if len(file_tensor) > config.CHUNK_SIZE:
                get_chunks = torch.tensor_split(file_tensor, math.ceil(len(file_tensor)/config.CHUNK_SIZE))
                for chunk in get_chunks:
                    if sum_empty:
                        sum_frame = asr.transcribe(frame = chunk)
                        sum_empty = False
                    else:
                        sum_frame = np.concatenate((sum_frame, asr.transcribe(frame = chunk)), axis=0)
                    
            elif len(file_tensor) > 256:
                if sum_empty:
                    sum_frame = asr.transcribe(frame = file_tensor)
                    sum_empty = False
                else:
                    sum_frame = np.concatenate((sum_frame, sr.transcribe(frame = file_tensor)), axis=0)
                
            WORKER_OUT[num_worker].put(sum_frame)
            
            
MANAGE_LOCK = Semaphore()

WORKER_IN = []
WORKER_OUT = []
WORKER_STATE = Array('i', range(config.NUM_WORKERS))
WORKER_PROCESS = Array('i', range(config.NUM_WORKERS))

for n in range(config.NUM_WORKERS):
    print(str(n))
    
    WORKER_IN.append(Queue())
    WORKER_OUT.append(Queue())
    WORKER_STATE[n] = 0
    
    p = Process(target=asr_process, args=(n,))
    p.start()
    WORKER_PROCESS[n] = p.pid
    print(p.pid)
    
def manage_worker(num_worker=None):
    global WORKER_STATE
    global WORKER_PROCESS
    
    MANAGE_LOCK.acquire()
    try:
        if num_worker == None:
            for n in range(config.NUM_WORKERS):
                if WORKER_STATE[n] == 0:
                    WORKER_STATE[n] = 1
                    num_worker = n
                    break
        else:
            if WORKER_STATE[num_worker] == 1:
                WORKER_STATE[num_worker] = 0
                num_worker = None
                
    finally:
        MANAGE_LOCK.release()
        
    return num_worker


BEAM_SEARCH_LM = CTCBeamDecoder(
    config.VOCAB,
    model_path=config.LM_PATH,
    alpha=config.ALPHA,
    beta=config.BETA,
    beam_width=config.BEAM_WIDTH,
    num_processes=max(os.cpu_count(), 1),
    blank_id=config.BLANK_ID,
    cutoff_prob=1.0,
    cutoff_top_n=40,
    log_probs_input=True)

def beam_decoder(logits):
    probs_seq = torch.FloatTensor(logits)
    
    beam_results, beam_scores, timesteps, out_lens = BEAM_SEARCH_LM.decode(probs_seq.unsqueeze(0))
    
    if config.CALC_CONF:
        conf = sum(1/np.exp(beam_scores[0].numpy()))
    else:
        conf = 0
    
    beam_res = beam_results[0][0][:out_lens[0][0]].cpu().numpy()
    times = timesteps[0][0][:out_lens[0][0]].cpu().numpy()
    lens = out_lens[0][0]
    
    transcript = ""
    
    if len(times) > 0:
        if times[0] < config.TIME_PAD:
            start = 0
            end = (times[0] + config.TIME_PAD*2) * config.TIME_STEP
        else:
            start = (times[0] - config.TIME_PAD) * config.TIME_STEP
            end = (times[0] + config.TIME_PAD) * config.TIME_STEP
            
        tocken_prev = config.VOCAB[int(beam_res[0])]
        word = tocken_prev
        
        result = []
        transcript = ''
        
        for n in range(1,lens):
            tocken = config.VOCAB[int(beam_res[n])]
            
            if tocken[0] != " ":
                word = word + tocken
                
            else:
                if start > end:
                    result_word = { 'start': round(end, 3), 'end': round(start, 3), 'word': word}
                else:
                    result_word = { 'start': round(start, 3), 'end': round(end, 3), 'word': word}
                    
                result.append(result_word)
                transcript = transcript + result_word["word"]
                
                if times[n] < config.TIME_PAD:
                    start = 0
                else:
                    start = (times[n] - config.TIME_PAD) * config.TIME_STEP

                word = tocken
                
                
            if times[n] < config.TIME_PAD:
                end = (times[n] + config.TIME_PAD*2) * config.TIME_STEP
            else:
                end = (times[n] + config.TIME_PAD) * config.TIME_STEP
            
            tocken_prev = tocken
            
            
        if start > end:
            result_word = { 'start': round(end, 3), 'end': round(start, 3), 'word': word}
        else:
            result_word = { 'start': round(start, 3), 'end': round(end, 3), 'word': word}
            
        result.append(result_word)
        transcript = transcript + result_word["word"]
        
    else:
        print(transcript)
        result = []

    if transcript != "" and transcript[0] == " ":
        transcript = transcript[1:]
        
    return transcript, result, conf

def preproces_data(raw):
    
    data = np.array(np.frombuffer(raw, dtype=np.int16))
    len_dt = len(data)
    
    if config.CUT_NOISE:
        max_lv = max(max(data)*0.02, 328)
        min_lv = min(min(data)*0.02, -328)
        cnt = 0
        for itm in range(800, len_dt, 800):
            if max(data[cnt:itm]) < max_lv and min(data[cnt:itm]) > min_lv:
                data[cnt:itm] = 0
            cnt = itm
        if len_dt > cnt:
            if max(data[cnt:len_dt]) < max_lv and min(data[cnt:len_dt]) > min_lv:
                data[cnt:len_dt] = 0
                
    return data/32768.0, math.ceil(len_dt/config.SAMPLE_RATE)
    