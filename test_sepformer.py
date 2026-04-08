import logging
import sys
logging.basicConfig(level=logging.DEBUG, stream=sys.stdout)

from app.diarization.separator_engine import SeparatorEngine
import numpy as np

print("Testando SepFormer instantiation...")
engine = SeparatorEngine(use_gpu=True, max_sources=2, model_type="SepFormer (Studio)")
print("Carregando modelo...")
engine.load_model()
print("Modelo carregado. is_ready =", engine.is_ready)

if engine.is_ready:
    print("Testando separação fake...")
    dummy_audio = np.random.randn(16000 * 3).astype(np.float32)
    tracks = engine.separate(dummy_audio)
    print("Tracks:", len(tracks))
else:
    print("Falha ao carregar")
