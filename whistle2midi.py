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
    """Real-time FFT visualizer that highlights the closest chromatic note based on the highest single peak"""
    
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
        
        # Setup matplotlib
        self.fig, (self.ax1, self.ax2) = plt.subplots(2, 1, figsize=(12, 8))
        plt.ion()
        
        # Create frequency bins
        self.freqs = np.fft.rfftfreq(n_fft, 1/sample_rate).astype(np.float32)
        self.freq_mask = (self.freqs >= 392) & (self.freqs <= 20000)
        self.display_freqs = self.freqs[self.freq_mask]
        self.n_display_freqs = len(self.display_freqs)
        
        # Pre-allocate spectrum arrays
        self.spectrum_full = np.zeros(len(self.freqs), dtype=np.float32)
        self.spectrum_display = np.zeros(self.n_display_freqs, dtype=np.float32)
        
        # Pre-create plot elements
        self.waveform_line, = self.ax1.plot([], [], 'b-', linewidth=1)
        self.x_waveform = np.arange(buffer_size, dtype=np.float32)
        
        # Setup chromatic scale
        self._setup_chromatic_scale()
        self._setup_axes()
    
    def _setup_chromatic_scale(self):
        """Pre-compute all chromatic scale frequencies and note names"""
        note_names_chromatic = ['G', 'G#', 'A', 'A#', 'B', 'C', 'C#', 'D', 'D#', 'E', 'F', 'F#']
        semitone_ratio = 2**(1/12)
        base_freq = 392.0  # G4
        base_octave = 4
        
        self.chromatic_freqs = []
        self.chromatic_names = []
        
        current_freq = base_freq
        octave = base_octave
        note_index = 0
        
        while current_freq <= 20000:
            self.chromatic_freqs.append(current_freq)
            note_name = note_names_chromatic[note_index]
            self.chromatic_names.append(f"{note_name}{octave}")
            
            current_freq *= semitone_ratio
            note_index += 1
            if note_index >= len(note_names_chromatic):
                note_index = 0
                octave += 1
        
        # Convert to numpy arrays for efficiency
        self.chromatic_freqs = np.array(self.chromatic_freqs, dtype=np.float32)
        
        # Separate natural notes for regular display
        self.natural_freqs = []
        self.natural_names = []
        for freq, name in zip(self.chromatic_freqs, self.chromatic_names):
            if '#' not in name:
                self.natural_freqs.append(freq)
                self.natural_names.append(name)
        
        self.natural_freqs = np.array(self.natural_freqs, dtype=np.float32)
    
    def _setup_axes(self):
        """Setup plot axes"""
        # Waveform plot
        self.ax1.set_xlim(0, self.buffer_size)
        self.ax1.set_ylim(-1, 1)
        self.ax1.set_title('Real-time Audio Waveform')
        self.ax1.set_xlabel('Sample Index')
        self.ax1.set_ylabel('Amplitude')
        self.ax1.grid(True, alpha=0.3)
        
        # Spectrum plot
        self.ax2.set_xlim(392, 20000)
        self.ax2.set_ylim(-10, 5)
        self.ax2.set_xscale('log')
        self.ax2.set_title('FFT Spectrum with Detected Note')
        self.ax2.set_xlabel('Frequency (Hz)')
        self.ax2.set_ylabel('Magnitude (log10)')
        self.ax2.grid(True, alpha=0.3)
        
        plt.tight_layout()
    
    def _find_closest_chromatic_note(self, frequency):
        """Find the closest chromatic note to a given frequency"""
        if frequency < self.chromatic_freqs[0] or frequency > self.chromatic_freqs[-1]:
            return None, None
        
        # Find the closest frequency
        distances = np.abs(self.chromatic_freqs - frequency)
        closest_index = np.argmin(distances)
        
        return self.chromatic_freqs[closest_index], self.chromatic_names[closest_index]
    
    def _get_top_1_peak(self):
        """Get the frequency of the single highest peak"""
        # Compute spectrum
        with self.buffer_lock:
            if len(self.audio_buffer) >= self.n_fft:
                self.windowed_data[:] = self.audio_buffer[-self.n_fft:] * self.hanning_window
            else:
                self.windowed_data[:len(self.audio_buffer)] = self.audio_buffer * self.hanning_window[:len(self.audio_buffer)]
                self.windowed_data[len(self.audio_buffer):] = 0
        
        # FFT computation
        np.fft.rfft(self.windowed_data, n=self.n_fft, out=self.fft_result)
        np.abs(self.fft_result, out=self.magnitude)
        np.maximum(self.magnitude, 1e-10, out=self.magnitude)
        np.log10(self.magnitude, out=self.spectrum_full)
        
        # Extract display range
        self.spectrum_display[:] = self.spectrum_full[self.freq_mask]
        
        # Find the single highest peak
        if len(self.spectrum_display) < 1:
            return None, None, self.spectrum_display
        
        top_1_index = np.argmax(self.spectrum_display)
        top_1_freq = self.display_freqs[top_1_index]
        top_1_mag = self.spectrum_display[top_1_index]
        
        # Check if the peak is below threshold (0)
        if top_1_mag < 0:
            return None, None, self.spectrum_display
        
        return top_1_freq, top_1_mag, self.spectrum_display
    
    def update_audio_data(self, audio_samples):
        """Update the audio buffer with new samples (thread-safe)"""
        with self.buffer_lock:
            shift_amount = len(audio_samples)
            self.audio_buffer[:-shift_amount] = self.audio_buffer[shift_amount:]
            self.audio_buffer[-shift_amount:] = audio_samples.astype(np.float32)
    
    def start_visualization(self):
        """Start the real-time chromatic note detection visualization"""
        self.running = True
        print("Chromatic Note Visualizer started. Detecting closest note from highest peak.")
        print("Close the plot window or press Ctrl+C to stop")
        
        try:
            while self.running and plt.get_fignums():
                # Get current buffer and compute peak frequency
                with self.buffer_lock:
                    current_buffer = self.audio_buffer.copy()
                
                peak_freq, peak_mag, spectrum = self._get_top_1_peak()
                
                # Clear the entire figure
                self.fig.clear()
                
                # Create a single large plot for note detection
                ax = self.fig.add_subplot(1, 1, 1)
                ax.set_xlim(392, 20000)
                ax.set_ylim(-1, 1)
                ax.set_xscale('log')
                ax.set_title('Detected Musical Note', fontsize=16)
                ax.set_xlabel('Frequency (Hz)', fontsize=12)
                ax.set_ylabel('Note Detection', fontsize=12)
                ax.grid(True, alpha=0.3)
                
                # Draw all chromatic scale lines (very light)
                for freq in self.chromatic_freqs:
                    ax.axvline(x=freq, color='lightgray', linestyle='-', alpha=0.2, linewidth=0.5)
                
                # Draw natural notes (light)
                for freq, name in zip(self.natural_freqs, self.natural_names):
                    ax.axvline(x=freq, color='gray', linestyle='--', alpha=0.4, linewidth=1)
                    ax.text(freq, 0.8, name, rotation=45, fontsize=10, alpha=0.6, ha='center')
                
                # Highlight detected note in red (if any)
                if peak_freq is not None:
                    closest_freq, closest_name = self._find_closest_chromatic_note(peak_freq)
                    if closest_freq is not None:
                        # Faint blue line for the actual detected frequency
                        ax.axvline(x=peak_freq, color='blue', linestyle='-', alpha=0.4, linewidth=2)
                        
                        # Large red highlight for the closest chromatic note
                        ax.axvline(x=closest_freq, color='red', linestyle='-', alpha=1.0, linewidth=5)
                        
                        # Large red text for the detected note
                        ax.text(closest_freq, 0, closest_name, rotation=0, fontsize=24, 
                               color='red', weight='bold', alpha=1.0, ha='center', va='center',
                               bbox=dict(boxstyle="round,pad=0.3", facecolor="white", edgecolor="red", alpha=0.9))
                        
                        # Show exact frequency and peak magnitude below
                        ax.text(closest_freq, -0.6, f'{peak_freq:.1f} Hz (peak: {peak_mag:.2f})', rotation=0, 
                               fontsize=14, color='red', alpha=0.8, ha='center')
                        
                        # Print to console as well
                        print(f"Detected: {closest_name} ({peak_freq:.1f} Hz, peak: {peak_mag:.2f})")
                else:
                    # Show "No Note Detected" message
                    ax.text(1000, 0, "No Note Detected", fontsize=20, color='gray', 
                           ha='center', va='center', alpha=0.7,
                           bbox=dict(boxstyle="round,pad=0.5", facecolor="lightgray", alpha=0.5))
                
                # Remove y-axis ticks since we're only showing note detection
                ax.set_yticks([])
                
                # Redraw
                self.fig.canvas.draw_idle()
                self.fig.canvas.flush_events()
                
                plt.pause(self.update_rate_ms / 1000.0)
                
        except KeyboardInterrupt:
            print("\nChromatic Note Visualizer stopped by user.")
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


def demo_mic_with_chromatic_note_visualization():
    """Demo function for chromatic note detection visualization"""
    
    global chromatic_visualizer_instance
    
    # Create chromatic note visualizer and microphone input
    chromatic_visualizer_instance = ChromaticNoteVisualizer(buffer_size=4096, update_rate_ms=4, sample_rate=44100, n_fft=4096)
    mic_input = MicrophoneInput(samplerate=44100, channels=1, blocksize=128)
    
    # Connect microphone to visualizer
    mic_input.set_audio_callback(chromatic_visualizer_instance.update_audio_data)
    
    print("Starting microphone input with real-time chromatic note detection...")
    print("The closest chromatic note to the highest peak will be highlighted in red")
    print("Close the plot window or press Ctrl+C to stop")
    
    try:
        with mic_input:
            chromatic_visualizer_instance.start_visualization()
            
    except KeyboardInterrupt:
        print("\nChromatic Note Demo stopped by user.")
    except Exception as e:
        print(f"Error: {e}")
    finally:
        chromatic_visualizer_instance.stop_visualization()


# Initialize global variable
chromatic_visualizer_instance = None


def get_top_peaks(n_peaks=1):
    """Convenience function to get the current top peak from the running visualizer"""
    try:
        if 'chromatic_visualizer_instance' in globals() and chromatic_visualizer_instance is not None:
            peak_freq, peak_mag, spectrum = chromatic_visualizer_instance._get_top_1_peak()
            if peak_freq is not None:
                return np.array([peak_freq]), np.array([peak_mag])
            else:
                return None, None
        else:
            print("No visualizer is currently running.")
            return None, None
    except Exception as e:
        print(f"Error getting peaks: {e}")
        return None, None


def send_simple_note():
    """Send simple MIDI notes (kept for potential future use)"""
    out = mido.open_output("Whistle2MIDI Out", virtual=True)

    for i in range(100):
        out.send(mido.Message('note_on', note=60, velocity=100))
        time.sleep(0.5)
        out.send(mido.Message('note_off', note=60))
        time.sleep(1.0)

    out.close()


if __name__ == "__main__":
    main()