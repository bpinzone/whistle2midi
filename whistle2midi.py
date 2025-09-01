#!/usr/bin/python3

import time
import mido
import sounddevice as sd
import numpy as np
import matplotlib.pyplot as plt
import threading
import librosa


def main():
    # send_simple_note()
    # demo_mic_with_visualization()  # Original waveform demo
    demo_mic_with_fft_visualization()  # New FFT demo


class FFTVisualizer:
    """Real-time FFT spectrum visualizer using librosa"""
    
    def __init__(self, buffer_size=2048, update_rate_ms=50, sample_rate=44100, n_fft=2048):
        self.buffer_size = buffer_size
        self.update_rate_ms = update_rate_ms
        self.sample_rate = sample_rate
        self.n_fft = n_fft
        self.audio_buffer = np.zeros(buffer_size)
        self.buffer_lock = threading.Lock()
        self.running = False
        
        # Setup matplotlib for frequency domain
        self.fig = plt.figure(figsize=(12, 8))
        plt.ion()
        
        # Create frequency bins for x-axis
        self.freqs = librosa.fft_frequencies(sr=sample_rate, n_fft=n_fft)
        # Limit to human hearing range (20 Hz - 20 kHz)
        self.freq_mask = (self.freqs >= 20) & (self.freqs <= 20000)
        self.display_freqs = self.freqs[self.freq_mask]
    
    def update_audio_data(self, audio_samples):
        """Update the audio buffer with new samples (thread-safe)"""
        with self.buffer_lock:
            shift_amount = len(audio_samples)
            self.audio_buffer[:-shift_amount] = self.audio_buffer[shift_amount:]
            self.audio_buffer[-shift_amount:] = audio_samples
    
    def _compute_fft_spectrum(self, audio_data):
        """Compute FFT spectrum using librosa"""
        if len(audio_data) < self.n_fft:
            # Pad with zeros if not enough data
            padded_data = np.pad(audio_data, (0, self.n_fft - len(audio_data)), 'constant')
        else:
            # Use the most recent n_fft samples
            padded_data = audio_data[-self.n_fft:]
        
        # Compute STFT using librosa
        stft = librosa.stft(padded_data, n_fft=self.n_fft, hop_length=self.n_fft//4)
        
        # Get magnitude spectrum
        magnitude = np.abs(stft)
        
        # Take the mean across time frames (if multiple frames)
        if magnitude.shape[1] > 1:
            magnitude = np.mean(magnitude, axis=1)
        else:
            magnitude = magnitude[:, 0]
        
        # Convert to dB scale
        magnitude_db = librosa.amplitude_to_db(magnitude, ref=np.max)
        
        return magnitude_db[self.freq_mask]
    
    def start_visualization(self):
        """Start the real-time FFT visualization loop"""
        self.running = True
        print("FFT Visualization started. Close the plot window to stop.")
        
        try:
            while self.running and plt.get_fignums():
                # Get current buffer data (thread-safe)
                with self.buffer_lock:
                    current_buffer = self.audio_buffer.copy()
                
                # Compute FFT spectrum
                spectrum = self._compute_fft_spectrum(current_buffer)
                
                # Update plot
                plt.clf()
                
                # Create subplot layout
                plt.subplot(2, 1, 1)
                # Plot waveform
                plt.plot(current_buffer, 'b-', linewidth=1)
                plt.ylim(-1, 1)
                plt.title('Real-time Audio Waveform')
                plt.xlabel('Sample Index')
                plt.ylabel('Amplitude')
                plt.grid(True, alpha=0.3)
                
                plt.subplot(2, 1, 2)
                # Plot FFT spectrum
                plt.plot(self.display_freqs, spectrum, 'r-', linewidth=1)
                plt.xlim(20, 20000)
                plt.ylim(-80, 0)  # dB range
                plt.xscale('log')  # Log scale for frequency
                plt.title('Real-time FFT Spectrum')
                plt.xlabel('Frequency (Hz)')
                plt.ylabel('Magnitude (dB)')
                plt.grid(True, alpha=0.3)
                
                # Add frequency labels for musical notes
                note_freqs = [82.4, 110, 146.8, 196, 261.6, 329.6, 392, 523.3, 659.3, 783.9, 1047, 1319, 1568]
                note_names = ['E2', 'A2', 'D3', 'G3', 'C4', 'E4', 'G4', 'C5', 'E5', 'G5', 'C6', 'E6', 'G6']
                
                for freq, name in zip(note_freqs, note_names):
                    if 20 <= freq <= 20000:
                        plt.axvline(x=freq, color='gray', linestyle='--', alpha=0.5)
                        plt.text(freq, -5, name, rotation=45, fontsize=8, alpha=0.7)
                
                plt.tight_layout()
                
                # Refresh the plot
                plt.pause(self.update_rate_ms / 1000.0)
                
        except KeyboardInterrupt:
            print("\nFFT Visualization stopped by user.")
        finally:
            self.stop_visualization()
    
    def stop_visualization(self):
        """Stop the visualization"""
        self.running = False
        plt.ioff()
        plt.close('all')


def demo_mic_with_fft_visualization():
    """Demo function combining microphone input with real-time FFT visualization"""
    
    # Create FFT visualizer and microphone input
    fft_visualizer = FFTVisualizer(buffer_size=4096, update_rate_ms=50, sample_rate=44100, n_fft=2048)
    mic_input = MicrophoneInput(samplerate=44100, channels=1, blocksize=1024)
    
    # Connect microphone to FFT visualizer
    mic_input.set_audio_callback(fft_visualizer.update_audio_data)
    
    print("Starting microphone input with real-time FFT spectrum visualization...")
    print("Close the plot window or press Ctrl+C to stop")
    
    try:
        with mic_input:
            # Start FFT visualization (this will block until stopped)
            fft_visualizer.start_visualization()
            
    except KeyboardInterrupt:
        print("\nFFT Demo stopped by user.")
    except Exception as e:
        print(f"Error: {e}")
    finally:
        fft_visualizer.stop_visualization()


class AudioVisualizer:
    """Real-time audio waveform visualizer"""
    
    def __init__(self, buffer_size=2048, update_rate_ms=50):
        self.buffer_size = buffer_size
        self.update_rate_ms = update_rate_ms
        self.audio_buffer = np.zeros(buffer_size)
        self.buffer_lock = threading.Lock()
        self.running = False
        
        # Setup matplotlib
        self.fig = plt.figure(figsize=(10, 6))
        plt.ion()
    
    def update_audio_data(self, audio_samples):
        """Update the audio buffer with new samples (thread-safe)"""
        with self.buffer_lock:
            shift_amount = len(audio_samples)
            self.audio_buffer[:-shift_amount] = self.audio_buffer[shift_amount:]
            self.audio_buffer[-shift_amount:] = audio_samples
    
    def start_visualization(self):
        """Start the real-time visualization loop"""
        self.running = True
        print("Visualization started. Close the plot window to stop.")
        
        try:
            while self.running and plt.get_fignums():
                # Get current buffer data (thread-safe)
                with self.buffer_lock:
                    current_buffer = self.audio_buffer.copy()
                
                # Update plot
                plt.clf()
                plt.plot(current_buffer, 'b-', linewidth=1)
                plt.ylim(-1, 1)
                plt.title('Real-time Audio Waveform')
                plt.xlabel('Sample Index')
                plt.ylabel('Amplitude')
                plt.grid(True, alpha=0.3)
                
                # Refresh the plot
                plt.pause(self.update_rate_ms / 1000.0)
                
        except KeyboardInterrupt:
            print("\nVisualization stopped by user.")
        finally:
            self.stop_visualization()
    
    def stop_visualization(self):
        """Stop the visualization"""
        self.running = False
        plt.ioff()
        plt.close('all')


class MicrophoneInput:
    """Microphone input handler"""
    
    def __init__(self, samplerate=44100, channels=1, blocksize=1024):
        self.samplerate = samplerate
        self.channels = channels
        self.blocksize = blocksize
        self.stream = None
        self.callback_func = None
    
    def set_audio_callback(self, callback_func):
        """Set the function to call when new audio data is available"""
        self.callback_func = callback_func
    
    def _audio_callback(self, indata, frames, time, status):
        """Internal audio callback"""
        if status:
            print(f"Audio status: {status}")
        
        # Get mono audio samples
        audio_samples = indata[:, 0] if self.channels == 1 else indata
        
        # Call the user-defined callback if set
        if self.callback_func:
            self.callback_func(audio_samples)
    
    def start_stream(self):
        """Start the audio input stream"""
        if self.stream is None:
            self.stream = sd.InputStream(
                callback=self._audio_callback,
                channels=self.channels,
                samplerate=self.samplerate,
                blocksize=self.blocksize
            )
            self.stream.start()
            print(f"Audio stream started: {self.samplerate}Hz, {self.channels} channel(s)")
    
    def stop_stream(self):
        """Stop the audio input stream"""
        if self.stream:
            self.stream.stop()
            self.stream.close()
            self.stream = None
            print("Audio stream stopped.")
    
    def __enter__(self):
        """Context manager entry"""
        self.start_stream()
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit"""
        self.stop_stream()


def demo_mic_with_visualization():
    """Demo function combining microphone input with real-time visualization"""
    
    # Create visualizer and microphone input with faster refresh rate
    visualizer = AudioVisualizer(buffer_size=2048, update_rate_ms=20)  # Faster refresh: 50 FPS
    mic_input = MicrophoneInput(samplerate=44100, channels=1, blocksize=1024)
    
    # Connect microphone to visualizer
    mic_input.set_audio_callback(visualizer.update_audio_data)
    
    print("Starting microphone input with real-time waveform visualization...")
    print("Close the plot window or press Ctrl+C to stop")
    
    try:
        with mic_input:
            # Start visualization (this will block until stopped)
            visualizer.start_visualization()
            
    except KeyboardInterrupt:
        print("\nDemo stopped by user.")
    except Exception as e:
        print(f"Error: {e}")
    finally:
        visualizer.stop_visualization()


def send_simple_note():
    out = mido.open_output("Whistle2MIDI Out", virtual=True)

    for i in range(100):
        out.send(mido.Message('note_on', note=60, velocity=100))
        time.sleep(0.5)
        out.send(mido.Message('note_off', note=60))
        time.sleep(1.0)  # 1 second between iterations

    out.close()


if __name__ == "__main__":
    main()