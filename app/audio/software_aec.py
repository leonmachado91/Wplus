"""Software Acoustic Echo Cancellation (AEC) using NLMS algorithm."""

import numpy as np
import logging

logger = logging.getLogger(__name__)

class SoftwareAEC:
    """
    Filtro Adaptativo de Cancelamento de Eco usando NLMS (Normalized Least Mean Squares).
    Projetado para rodar em tempo real no Python sem dependências C++ problemáticas.
    
    A janela de filtro precisa cobrir o delay do som da caixa até o microfone.
    Num ambiente de áudio comum no Windows via WASAPI, o delay é entre 50ms a 200ms.
    Em 16kHz, 256ms = 4096 samples.
    """
    def __init__(self, filter_taps: int = 4096, mu: float = 0.5) -> None:
        self.taps = filter_taps
        self.mu = mu
        # w são os pesos do filtro da onda acústica do eco
        self.w = np.zeros(self.taps, dtype=np.float32)
        # buffer circular dobrado para não gastar processador reconstruindo arrays
        self.x_buf = np.zeros(self.taps * 2, dtype=np.float32)
        self.idx = self.taps
        self.eps = 1e-2

    def process_frame(self, mic_frame: np.ndarray, ref_frame: np.ndarray) -> np.ndarray:
        """
        Subtrai o sinal de referência (Loopback do Desktop) do sinal do microfone.
        
        Args:
            mic_frame: np.ndarray 1D float32 com o áudio gravado no ambiente.
            ref_frame: np.ndarray 1D float32 com o áudio do Windows (Discord).
            
        Returns:
            np.ndarray 1D float32 com o áudio limpo.
        """
        if len(mic_frame) != len(ref_frame):
            # Se desalinharem por algum erro bizarro do SoundDevice vs Pyaudio, bypass
            return mic_frame

        mic = mic_frame.astype(np.float32).flatten()
        ref = ref_frame.astype(np.float32).flatten()
        out = np.zeros_like(mic)
        n = len(mic)
        
        # Referências locais para otimizar velocidade no loop
        w = self.w
        x_buf = self.x_buf
        mu = self.mu
        eps = self.eps
        taps = self.taps
        
        for i in range(n):
            self.idx -= 1
            if self.idx < 0:
                # O buffer rodou até o final, reset sem criar novos objetos pesados
                x_buf[taps : 2*taps] = x_buf[0 : taps]
                self.idx = taps - 1
                
            x_buf[self.idx] = ref[i]
            
            # View instantânea da janela (ação O(1) no NumPy)
            x_view = x_buf[self.idx : self.idx + taps]
            
            # Prediz o eco na amostra atual
            y = np.dot(w, x_view)
            e = mic[i] - y
            
            # Atualiza o filtro apenas se houver áudio forte de referência
            # NLMS equation
            norm = np.dot(x_view, x_view)
            if norm > 1e-4:
                # factor = (mu * e) / (norm + eps)
                w += ((mu * e) / (norm + eps)) * x_view
                
            out[i] = e
            
        return out
