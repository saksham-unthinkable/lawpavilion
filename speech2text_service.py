import numpy as np
import tensorflow as tf
from deepspeech import Model
import os
import wave

MODEL_PATH = 'ft_model_140.pbmm'
lm_alpha = 0.9190937309595953
lm_beta = 2.6800533813869043
beam_width = 500

class _Speech2Text_Service :
    model = None
    _instance = None
    def read_wav_file(self, filename):
        with wave.open(filename, 'rb') as f:
            rate = f.getframerate()
            frames = f.getnframes()
            buffer = f.readframes(frames)
        return buffer, rate

    def transcribe(self, audio_file):
        buffer, rate = self.read_wav_file(audio_file)
        data = np.frombuffer(buffer, dtype=np.int16)
        return self.model.stt(data)

def Speech2Text_Service():
    if _Speech2Text_Service._instance is None:
        _Speech2Text_Service._instance = _Speech2Text_Service()
        _Speech2Text_Service.model = Model(MODEL_PATH)
        _Speech2Text_Service.model.enableExternalScorer('kenlm-nigerian1.scorer')
        _Speech2Text_Service.model.setScorerAlphaBeta(lm_alpha, lm_beta)
        _Speech2Text_Service.model.setBeamWidth(beam_width)

    return _Speech2Text_Service._instance

# if __name__ == "__main__":
#     s2s = Speech2Text_Service()
#     transcript = s2s.transcribe('ngf_00295_01314275188.wav')
#     print(f"prediction :- {transcript}")