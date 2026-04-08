"""Separator engine — source separation using Conv-TasNet via Asteroid.

Abordagem: Source Separation antes da transcrição.
    O VAD (Voice Activity Detection) inicial gera um chunk que pode ter
    fala sobreposta (2+ pessoas falando ao mesmo tempo).
    
    Para evitar que o Groq/Whisper alucine, este engine recebe o array PCM inteiro,
    isola as vozes e retorna N trilhas limpas com o exato mesmo tamanho (zero-padding
    mantido onde não há fala na trilha respectiva, para preservar a minutagem do Buffer).
"""

from __future__ import annotations

import logging
import os
import threading
import time
from pathlib import Path
from typing import Optional

import numpy as np

logger = logging.getLogger(__name__)


class SeparatorEngine:
    """Isola fontes de áudio usando Conv-TasNet (mpariente/ConvTasNet_WHAM_sepclean).
    
    Diferencial: Executado no WorkerConsumer, não bloqueia o AudioCapture ou o VAD.
    """

    def __init__(self, use_gpu: bool = True, max_sources: int = 2, model_type: str = "Conv-TasNet (Fast)"):
        self._use_gpu = use_gpu
        self._max_sources = max_sources
        self._model_type = model_type
        self._model = None
        self._is_ready = False
        self._lock = threading.Lock()
        
        # Carregaremos o modelo preguiçosamente para não afogar o boot da UI.
        # Asteroid baixa o modelo do Hugging Face.
    
    @property
    def is_ready(self) -> bool:
        return self._is_ready

    def load_model(self) -> None:
        """Carrega o modelo selecionado e isola em contexto CUDA."""
        with self._lock:
            if self._is_ready:
                return

            import torch
            device = "cuda" if self._use_gpu and torch.cuda.is_available() else "cpu"
            logger.info(f"Carregando {self._model_type} no {device.upper()}...")
            
            t0 = time.monotonic()
            try:
                if "SepFormer" in self._model_type:
                    # Monkey-patch para torchaudio >= 2.1 e SpeechBrain
                    import torchaudio
                    if not hasattr(torchaudio, "list_audio_backends"):
                        torchaudio.list_audio_backends = lambda: ["soundfile"]
                        
                    # Monkey-patch para huggingface_hub >= 0.22.0 e retornos 404
                    import huggingface_hub
                    if hasattr(huggingface_hub, "hf_hub_download") and not hasattr(huggingface_hub.hf_hub_download, "_is_patched"):
                        _old_hf_download = huggingface_hub.hf_hub_download
                        def _patched_hf_download(*args, **kwargs):
                            if "use_auth_token" in kwargs:
                                val = kwargs.pop("use_auth_token")
                                if "token" not in kwargs:
                                    kwargs["token"] = val
                            try:
                                return _old_hf_download(*args, **kwargs)
                            except Exception as e:
                                if "NotFoundError" in e.__class__.__name__:
                                    from requests.exceptions import HTTPError
                                    raise HTTPError("404 Client Error: Not Found") from e
                                raise
                        _patched_hf_download._is_patched = True
                        huggingface_hub.hf_hub_download = _patched_hf_download
                        
                    from speechbrain.inference.separation import SepformerSeparation
                    # Definimos pasta local limpa igual ao ECAPA
                    savedir = os.path.join(os.getcwd(), ".models/speechbrain", "sepformer-libri2mix")
                    self._model = SepformerSeparation.from_hparams(
                        source="speechbrain/sepformer-libri2mix",
                        savedir=savedir,
                        run_opts={"device": device}
                    )
                else:
                    from asteroid.models import ConvTasNet
                    # Conv-TasNet: Libri2Mix 16kHz preserva alta frequência nítida vital pro Whisper.
                    self._model = ConvTasNet.from_pretrained("JorisCos/ConvTasNet_Libri2Mix_sepclean_16k")
                    if device == "cuda":
                        self._model.to(device)
                    self._model.eval()

                self._is_ready = True
                logger.info(f"Separation Engine ({self._model_type}) pronto em {time.monotonic() - t0:.1f}s")
            except RuntimeError as e:
                if "out of memory" in str(e).lower():
                    logger.error(f"Erro Crítico de Memória (OOM) ao carregar {self._model_type}! Sua placa de vídeo excedeu o limite. Use o modo Conv-TasNet.")
                else:
                    logger.error("Erro ao carregar modelo de separação: %s", e, exc_info=True)
                self._model = None
            except Exception as e:
                logger.error("Erro ao carregar modelo de separação: %s", e, exc_info=True)
                self._model = None

    def separate(self, audio_np: np.ndarray, sample_rate: int = 16000) -> list[np.ndarray]:
        """Processa a separação de vozes (Source Separation) no chunk.
        
        Args:
            audio_np: Array 1D mono `float32`.
            sample_rate: Taxa de amostragem padrão 16000hz.
        
        Returns:
            Lista de `np.ndarray` contendo N trilhas extraídas do áudio.
            O tamanho de cada array é sempre idêntico ao `audio_np` da entrada.
        """
        if not self._is_ready or self._model is None:
            logger.warning("Separator engine chamado, mas o modelo não foi carregado.")
            return [audio_np]

        import torch
        import torchaudio.functional as F

        # O SepformerSeparation wrapper gerencia o model parameter `.device` internamente 
        # acessível melhor por `self._model.device`.
        if hasattr(self._model, "parameters"):
            device = next(self._model.parameters()).device
        else:
            device = self._model.device # SpeechBrain Inference wrapper
            
        t0 = time.monotonic()
        try:
            with torch.no_grad():
                # Converter para Tensor PyTorch em Float32 nativo
                tensor = torch.from_numpy(audio_np).to(torch.float32).to(device)
                
                if "SepFormer" in self._model_type:
                    # SepFormer aceita input [batch, time]
                    if tensor.ndim == 1:
                        tensor = tensor.unsqueeze(0)
                        
                    # SpeechBrain nativamente faz o resampling se necessário mas é lento.
                    # Vamos ignorar porque Libri2Mix já é 16000Hz (mesmo rate que passamos).
                    est_sources = self._model.separate_batch(tensor)
                    # SepFormer devolve batch de trilhas [batch, time, n_src] -> permute para [batch, n_src, time]
                    # O Asteroid devolve [batch, n_src, time]
                    if est_sources.ndim == 3:
                        est_sources = est_sources.transpose(1, 2)
                        
                else:
                    # Conv-TasNet via Asteroid
                    model_sr = getattr(self._model, "sample_rate", 16000)
                    if isinstance(model_sr, float):
                        model_sr = int(model_sr)

                    if tensor.ndim == 1:
                        tensor = tensor.unsqueeze(0).unsqueeze(0)
                    elif tensor.ndim == 2:
                        tensor = tensor.unsqueeze(0)
                        
                    if sample_rate != model_sr:
                        tensor = F.resample(tensor, sample_rate, model_sr)
                        
                    est_sources = self._model(tensor)
                    
                    if sample_rate != model_sr:
                        est_sources = F.resample(est_sources, model_sr, sample_rate)
                
                # Cleanup Tensor
                est_sources = est_sources.squeeze(0).cpu().to(torch.float32).numpy()
                
            # Correção de Anomalia Acústica (Estouro de PCM-16):
            # Equalizamos os volumes para não clipar.
            overall_max = np.max(np.abs(est_sources))
            input_max = np.max(np.abs(audio_np))
            if overall_max > 1e-6:
                # Usamos um boost base de 0.8 para o Whisper ouvir claramente, mas respeitamos max=1.0
                target_peak = max(input_max, 0.8) 
                est_sources = (est_sources / overall_max) * target_peak
                
            # Garantia final contra distorções
            est_sources = np.clip(est_sources, -1.0, 1.0)
                
            # Converter de volta para lista de numpy
            tracks = []
            for i in range(min(est_sources.shape[0], self._max_sources)):
                tracks.append(est_sources[i])
            
            # Limpeza de VRAM proativa após inferência
            if device.type == "cuda":
                torch.cuda.empty_cache()
                
            elapsed = time.monotonic() - t0
            logger.debug("%s separou chunk de %.2fs em %.2fs (%s trilhas limit: %s)", 
                         self._model_type, len(audio_np)/sample_rate, elapsed, est_sources.shape[0], self._max_sources)
            
            return tracks

        except RuntimeError as e:
            if "out of memory" in str(e).lower():
                logger.error("CUDA OOM em Separação! A GPU fritou. O áudio vazou vazio.")
            else:
                logger.error("Separação de Áudio Falhou: %s", e)
            if device.type == "cuda":
                torch.cuda.empty_cache()
            return [audio_np]
        except Exception as e:
            logger.error("Separação de Áudio Falhou: %s", e)
            if device.type == "cuda":
                torch.cuda.empty_cache()
            
            # Fail-Safe: Se o separador falhar (ex: OOM), devolve o áudio mixed
            # para o Groq tentar transcrever pelo menos alguma coisa.
            return [audio_np]
