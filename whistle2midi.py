#!/usr/bin/python3

import time
import mido
import sounddevice as sd
import numpy as np
import matplotlib.pyplot as plt
import threading


def main():
    demo_mic_with_chromatic_note_visualization()


class ChromaticNoteVisualizer:
    """Real-time chromatic note detection from microphone input"""
    
    def __init__(self, buffer_size=4096, update_rate_ms=4, sample_rate=44100, n_fft=4096):
        self.buffer_size = buffer_size
        self.update_rate_ms = update_rate_ms
        self.sample_rate = sample_rate
        self.n_fft = n_fft
        self.audio_buffer = np.zeros(buffer_size, dtype=np.float32)
        self.buffer_lock = threading.Lock()
        self.running = False
        
        # Pre-allocate arrays for performance
        self.windowed_data = np.zeros(n_fft, dtype=np.float32)
        self.fft_result = np.zeros(n_fft//2 + 1, dtype=np.complex64)
        self.magnitude = np.zeros(n_fft//2 + 1, dtype=np.float32)
        self.hanning_window = np.hanning(n_fft).astype(np.float32)
        
        # Create frequency bins
        self.freqs = np.fft.rfftfreq(n_fft, 1/sample_rate).astype(np.float32)
        self.freq_mask = (self.freqs >= 392) & (self.freqs <= 20000)  # G4 to 20kHz
        self.display_freqs = self.freqs[self.freq_mask]
        
        # Pre-allocate spectrum arrays
        self.spectrum_full = np.zeros(len(self.freqs), dtype=np.float32)
        self.spectrum_display = np.zeros(len(self.display_freqs), dtype=np.float32)
        
        # Setup chromatic scale
        self._setup_chromatic_scale()
    
    def _setup_chromatic_scale(self):
        """Pre-compute all chromatic scale frequencies and note names"""
        note_names = ['G', 'G#', 'A', 'A#', 'B', 'C', 'C#', 'D', 'D#', 'E', 'F', 'F#']
        semitone_ratio = 2**(1/12)
        
        frequencies = []
        names = []
        
        # Generate all chromatic notes from G4 to 20kHz
        current_freq = 392.0  # G4
        octave = 4
        note_index = 0
        
        while current_freq <= 20000:
            frequencies.append(current_freq)
            names.append(f"{note_names[note_index]}{octave}")
            
            current_freq *= semitone_ratio
            note_index += 1
            if note_index >= len(note_names):
                note_index = 0
                octave += 1
        
        self.chromatic_freqs = np.array(frequencies, dtype=np.float32)
        self.chromatic_names = names
        
        # Separate natural notes for labeling
        self.natural_freqs = []
        self.natural_names = []
        for freq, name in zip(self.chromatic_freqs, self.chromatic_names):
            if '#' not in name:
                self.natural_freqs.append(freq)
                self.natural_names.append(name)
    
    def _find_closest_note(self, frequency):
        """Find the closest chromatic note to a frequency"""
        if frequency < self.chromatic_freqs[0] or frequency > self.chromatic_freqs[-1]:
            return None, None
        
        distances = np.abs(self.chromatic_freqs - frequency)
        closest_index = np.argmin(distances)
        return self.chromatic_freqs[closest_index], self.chromatic_names[closest_index]
    
    def _get_peak_frequency(self):
        """Get the frequency of the highest peak"""
        with self.buffer_lock:
            if len(self.audio_buffer) >= self.n_fft:
                self.windowed_data[:] = self.audio_buffer[-self.n_fft:] * self.hanning_window
            else:
                self.windowed_data[:len(self.audio_buffer)] = self.audio_buffer * self.hanning_window[:len(self.audio_buffer)]
                self.windowed_data[len(self.audio_buffer):] = 0
        
        # Compute FFT
        np.fft.rfft(self.windowed_data, n=self.n_fft, out=self.fft_result)
        np.abs(self.fft_result, out=self.magnitude)
        np.maximum(self.magnitude, 1e-10, out=self.magnitude)
        np.log10(self.magnitude, out=self.spectrum_full)
        
        # Extract frequency range of interest
        self.spectrum_display[:] = self.spectrum_full[self.freq_mask]
        
        # Find highest peak
        if len(self.spectrum_display) < 1:
            return None, None
        
        peak_index = np.argmax(self.spectrum_display)
        peak_freq = self.display_freqs[peak_index]
        peak_mag = self.spectrum_display[peak_index]
        
        # Only return if peak is significant
        if peak_mag < 0:
            return None, None
        
        return peak_freq, peak_mag
    
    def get_current_note(self):
        """Get the currently detected note (public API)"""
        peak_freq, peak_mag = self._get_peak_frequency()
        if peak_freq is not None:
            closest_freq, closest_name = self._find_closest_note(peak_freq)
            return {
                'note': closest_name,
                'frequency': float(peak_freq),
                'magnitude': float(peak_mag),
                'closest_note_freq': float(closest_freq) if closest_freq else None
            }
        return None
    
    def update_audio_data(self, audio_samples):
        """Update the audio buffer with new samples"""
        with self.buffer_lock:
            shift_amount = len(audio_samples)
            self.audio_buffer[:-shift_amount] = self.audio_buffer[shift_amount:]
            self.audio_buffer[-shift_amount:] = audio_samples.astype(np.float32)
    
    def start_visualization(self):
        """Start the real-time visualization"""
        self.running = True
        
        # Setup matplotlib
        fig = plt.figure(figsize=(12, 6))
        plt.ion()
        
        print("Chromatic Note Visualizer started.")
        print("Close the plot window or press Ctrl+C to stop")
        
        try:
            while self.running and plt.get_fignums():
                peak_freq, peak_mag = self._get_peak_frequency()
                
                # Clear and setup plot
                plt.clf()
                ax = plt.gca()
                ax.set_xlim(392, 20000)
                ax.set_ylim(-1, 1)
                ax.set_xscale('log')
                ax.set_title('Detected Musical Note', fontsize=16)
                ax.set_xlabel('Frequency (Hz)', fontsize=12)
                ax.grid(True, alpha=0.3)
                ax.set_yticks([])
                
                # Draw chromatic scale reference lines
                for freq in self.chromatic_freqs:
                    ax.axvline(x=freq, color='lightgray', linestyle='-', alpha=0.2, linewidth=0.5)
                
                # Draw and label natural notes
                for freq, name in zip(self.natural_freqs, self.natural_names):
                    ax.axvline(x=freq, color='gray', linestyle='--', alpha=0.4, linewidth=1)
                    ax.text(freq, 0.8, name, rotation=45, fontsize=10, alpha=0.6, ha='center')
                
                # Show detected note
                if peak_freq is not None:
                    closest_freq, closest_name = self._find_closest_note(peak_freq)
                    if closest_freq is not None:
                        # Blue line for actual frequency
                        ax.axvline(x=peak_freq, color='blue', linestyle='-', alpha=0.4, linewidth=2)
                        
                        # Red line and label for closest note
                        ax.axvline(x=closest_freq, color='red', linestyle='-', alpha=1.0, linewidth=5)
                        ax.text(closest_freq, 0, closest_name, rotation=0, fontsize=24, 
                               color='red', weight='bold', ha='center', va='center',
                               bbox=dict(boxstyle="round,pad=0.3", facecolor="white", edgecolor="red"))
                        
                        # Show frequency and magnitude
                        ax.text(closest_freq, -0.6, f'{peak_freq:.1f} Hz (peak: {peak_mag:.2f})', 
                               rotation=0, fontsize=14, color='red', alpha=0.8, ha='center')
                        
                        print(f"Detected: {closest_name} ({peak_freq:.1f} Hz, peak: {peak_mag:.2f})")
                else:
                    ax.text(1000, 0, "No Note Detected", fontsize=20, color='gray', 
                           ha='center', va='center', alpha=0.7)
                
                plt.pause(self.update_rate_ms / 1000.0)
                
        except KeyboardInterrupt:
            print("\nVisualization stopped.")
        finally:
            self.stop_visualization()
    
    def stop_visualization(self):
        """Stop the visualization"""
        self.running = False
        plt.ioff()
        plt.close('all')


class MicrophoneInput:
    """Microphone input handler"""
    
    def __init__(self, samplerate=44100, channels=1, blocksize=128):
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
        
        audio_samples = indata[:, 0] if self.channels == 1 else indata
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
            print(f"Audio stream started: {self.samplerate}Hz")
    
    def stop_stream(self):
        """Stop the audio input stream"""
        if self.stream:
            self.stream.stop()
            self.stream.close()
            self.stream = None
            print("Audio stream stopped.")
    
    def __enter__(self):
        self.start_stream()
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        self.stop_stream()


def demo_mic_with_chromatic_note_visualization():
    """Demo function for chromatic note detection"""
    
    # Create components
    visualizer = ChromaticNoteVisualizer()
    mic_input = MicrophoneInput()
    
    # Connect them
    mic_input.set_audio_callback(visualizer.update_audio_data)
    
    print("Starting real-time chromatic note detection...")
    print("The detected note will be highlighted in red")
    print("Close the plot window or press Ctrl+C to stop")
    
    try:
        with mic_input:
            visualizer.start_visualization()
    except KeyboardInterrupt:
        print("\nDemo stopped.")
    except Exception as e:
        print(f"Error: {e}")


def send_simple_note():
    """Send simple MIDI notes"""
    out = mido.open_output("Whistle2MIDI Out", virtual=True)

    for i in range(100):
        out.send(mido.Message('note_on', note=60, velocity=100))
        time.sleep(0.5)
        out.send(mido.Message('note_off', note=60))
        time.sleep(1.0)

    out.close()


if __name__ == "__main__":
    main()