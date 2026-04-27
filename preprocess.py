import numpy as np
import pandas as pd
import pywt
from scipy import signal
from scipy.stats import entropy

class VirtualChannelExpander:
    """
    Virtual Channel Expander using Wavelet Packet Decomposition (WPD).
    Expands 4 physical channels to 16 virtual channels by selecting Top-K frequency bands.
    """
    def __init__(self, wavelet='db4', level=3, top_k=3):
        self.wavelet = wavelet
        self.level = level
        self.top_k = top_k

    def wpd_decompose(self, signal_1d):
        wp = pywt.WaveletPacket(data=signal_1d, wavelet=self.wavelet, mode='symmetric', maxlevel=self.level)
        nodes = wp.get_level(self.level, 'freq')
        subbands = [n.data for n in nodes]
        return subbands

    def select_top_k(self, subbands):
        scores = []
        for sb in subbands:
            energy = np.sum(sb**2)
            # Normalize for entropy calculation
            prob = np.abs(sb) / (np.sum(np.abs(sb)) + 1e-10)
            ent = entropy(prob + 1e-10)
            # Score: Energy / Entropy (Higher energy, lower entropy -> more impulsive/informative)
            score = energy / (ent + 1e-10)
            scores.append(score)
            
        top_indices = np.argsort(scores)[::-1][:self.top_k]
        return [subbands[i] for i in top_indices]

    def expand(self, x_4ch):
        """
        x_4ch: (T, 4) array
        Returns: (T, 16) array
        """
        T, C = x_4ch.shape
        x_16ch = []
        
        for c in range(C):
            # 1. Keep original physical channel
            x_16ch.append(x_4ch[:, c])
            
            # 2. WPD Decomposition
            subbands = self.wpd_decompose(x_4ch[:, c])
            
            # 3. Select Top-K
            selected = self.select_top_k(subbands)
            
            # Interpolate back to original length if needed (pywt decimation)
            for sb in selected:
                sb_resampled = signal.resample(sb, T)
                x_16ch.append(sb_resampled)
                
        return np.stack(x_16ch, axis=1)

class CSDMatrixBuilder:
    """
    Cross-Spectral Density (CSD) Matrix Builder using Welch's method.
    """
    def __init__(self, fs=100, nperseg=256):
        self.fs = fs
        self.nperseg = nperseg

    def build(self, x_16ch):
        """
        x_16ch: (T, 16) array
        Returns: (32, 32) real-valued matrix representing the 16x16 complex CSD matrix
        """
        T, C = x_16ch.shape
        csd_complex = np.zeros((C, C), dtype=np.complex128)
        
        for i in range(C):
            for j in range(C):
                f, Pxy = signal.csd(x_16ch[:, i], x_16ch[:, j], 
                                    fs=self.fs, nperseg=min(self.nperseg, T), 
                                    noverlap=min(self.nperseg, T)//2)
                # Aggregate frequency bins (mean)
                csd_complex[i, j] = np.mean(Pxy)
                
        # Convert to 32x32 real matrix: [[Re, -Im], [Im, Re]]
        csd_real = np.zeros((2*C, 2*C), dtype=np.float32)
        csd_real[:C, :C] = np.real(csd_complex)
        csd_real[C:, C:] = np.real(csd_complex)
        csd_real[:C, C:] = -np.imag(csd_complex)
        csd_real[C:, :C] = np.imag(csd_complex)
        
        return csd_real

def main():
    print("Loading sample data...")
    df = pd.read_csv('github_release/sample_data.csv')
    
    # Extract physical channels: P1, V1, P2, V2
    # The sample_data.csv has V1, V2, P1, P2
    x_4ch = df[['P1', 'V1', 'P2', 'V2']].values
    print(f"Original shape: {x_4ch.shape}")
    
    # 1. Virtual Channel Expansion
    print("Performing WPD Virtual Channel Expansion (L=3, K=3)...")
    expander = VirtualChannelExpander(level=3, top_k=3)
    x_16ch = expander.expand(x_4ch)
    print(f"Expanded shape: {x_16ch.shape}")
    
    # 2. CSD Matrix Construction
    print("Computing CSD Matrix (Welch's method)...")
    builder = CSDMatrixBuilder(fs=100, nperseg=256)
    csd_matrix = builder.build(x_16ch)
    print(f"CSD Matrix shape: {csd_matrix.shape}")
    
    # Save output
    np.save('github_release/sample_csd.npy', csd_matrix)
    print("Saved sample_csd.npy successfully.")

if __name__ == '__main__':
    main()
