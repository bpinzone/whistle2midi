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
    # demo_mic_with_fft_visualization()  # Full FFT demo
    demo_mic_with_peak_fft_visualization(n_peaks=3)  # New peak FFT demo - show top 5 peaks


class FFTVisualizer:
    """Real-time FFT spectrum visualizer using librosa"""
    
    def __init__(self, buffer_size=4096, update_rate_ms=4, sample_rate=44100, n_fft=4096):
        self.buffer_size = buffer_size
        self.update_rate_ms = update_rate_ms
        self.sample_rate = sample_rate
        self.n_fft = n_fft
        self.audio_buffer = np.zeros(buffer_size)
        self.buffer_lock = threading.Lock()
        self.running = False
        
        # Setup matplotlib for frequency domain
        self.fig, (self.ax1, self.ax2) = plt.subplots(2, 1, figsize=(12, 8))
        plt.ion()
        
        # Create frequency bins for x-axis
        self.freqs = np.fft.rfftfreq(n_fft, 1/sample_rate)  # Use numpy for efficiency
        # Limit to frequencies above G4 (392 Hz) up to 20 kHz
        self.freq_mask = (self.freqs >= 392) & (self.freqs <= 20000)
        self.display_freqs = self.freqs[self.freq_mask]
        
        # Pre-create plot lines for efficiency
        self.waveform_line, = self.ax1.plot([], [], 'b-', linewidth=1)
        self.spectrum_line, = self.ax2.plot([], [], 'r-', linewidth=1)
        
        # Setup axes once
        self._setup_axes()
    
    def _setup_axes(self):
        """Setup plot axes once for efficiency"""
        # Waveform plot setup
        self.ax1.set_xlim(0, self.buffer_size)
        self.ax1.set_ylim(-1, 1)
        self.ax1.set_title('Real-time Audio Waveform')
        self.ax1.set_xlabel('Sample Index')
        self.ax1.set_ylabel('Amplitude')
        self.ax1.grid(True, alpha=0.3)
        
        # Spectrum plot setup
        self.ax2.set_xlim(392, 20000)  # Start from G4 (392 Hz)
        self.ax2.set_ylim(-10, 5)  # Log scale range
        self.ax2.set_xscale('log')
        self.ax2.set_title('Real-time FFT Spectrum')
        self.ax2.set_xlabel('Frequency (Hz)')
        self.ax2.set_ylabel('Magnitude (log10)')
        self.ax2.grid(True, alpha=0.3)
        
        # Generate full chromatic scale markers (G4 and above)
        note_names_chromatic = ['G', 'G#', 'A', 'A#', 'B', 'C', 'C#', 'D', 'D#', 'E', 'F', 'F#']
        semitone_ratio = 2**(1/12)
        base_freq = 392.0  # G4
        base_octave = 4
        
        # Generate all chromatic notes
        all_freqs = []
        major_freqs = []
        major_names = []
        
        current_freq = base_freq
        octave = base_octave
        note_index = 0
        
        while current_freq <= 20000:
            all_freqs.append(current_freq)
            note_name = note_names_chromatic[note_index]
            full_name = f"{note_name}{octave}"
            
            # Add to major notes if it's a natural note (no sharp)
            if '#' not in note_name:
                major_freqs.append(current_freq)
                major_names.append(full_name)
            
            current_freq *= semitone_ratio
            note_index += 1
            if note_index >= len(note_names_chromatic):
                note_index = 0
                octave += 1
        
        # Draw all chromatic scale lines (light)
        for freq in all_freqs:
            self.ax2.axvline(x=freq, color='lightgray', linestyle='-', alpha=0.3, linewidth=0.5)
        
        # Draw and label major notes (darker)
        for freq, name in zip(major_freqs, major_names):
            self.ax2.axvline(x=freq, color='gray', linestyle='--', alpha=0.7, linewidth=1)
            self.ax2.text(freq, 4, name, rotation=45, fontsize=8, alpha=0.8)
        
        plt.tight_layout()
    
    def update_audio_data(self, audio_samples):
        """Update the audio buffer with new samples (thread-safe)"""
        with self.buffer_lock:
            shift_amount = len(audio_samples)
            self.audio_buffer[:-shift_amount] = self.audio_buffer[shift_amount:]
            self.audio_buffer[-shift_amount:] = audio_samples
    
    def _compute_fft_spectrum(self, audio_data):
        """Compute FFT spectrum efficiently using numpy"""
        # Use windowing for better frequency resolution
        windowed_data = audio_data * np.hanning(len(audio_data))
        
        # Compute FFT using numpy (faster than librosa STFT for single frame)
        fft = np.fft.rfft(windowed_data, n=self.n_fft)
        
        # Get magnitude spectrum
        magnitude = np.abs(fft)
        
        # Use log scaling instead of dB scaling
        magnitude_log = np.log10(np.maximum(magnitude, 1e-10))  # Avoid log(0)
        
        return magnitude_log[self.freq_mask]
    
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
                
                # Update plots efficiently (no plt.clf())
                x_waveform = np.arange(len(current_buffer))
                self.waveform_line.set_data(x_waveform, current_buffer)
                self.spectrum_line.set_data(self.display_freqs, spectrum)
                
                # Redraw only what changed
                self.ax1.draw_artist(self.waveform_line)
                self.ax2.draw_artist(self.spectrum_line)
                self.fig.canvas.flush_events()
                
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


class PeakFFTVisualizer:
    """Real-time FFT spectrum visualizer showing only the N highest peaks"""
    
    def __init__(self, buffer_size=4096, update_rate_ms=4, sample_rate=44100, n_fft=4096, n_peaks=10):
        self.buffer_size = buffer_size
        self.update_rate_ms = update_rate_ms
        self.sample_rate = sample_rate
        self.n_fft = n_fft
        self.n_peaks = n_peaks  # Number of peaks to show
        self.audio_buffer = np.zeros(buffer_size, dtype=np.float32)  # Use float32 for speed
        self.buffer_lock = threading.Lock()
        self.running = False
        
        # Pre-allocate arrays for performance
        self.windowed_data = np.zeros(n_fft, dtype=np.float32)
        self.fft_result = np.zeros(n_fft//2 + 1, dtype=np.complex64)  # Pre-allocated FFT result
        self.magnitude = np.zeros(n_fft//2 + 1, dtype=np.float32)
        self.hanning_window = np.hanning(n_fft).astype(np.float32)  # Pre-computed window
        
        # Setup matplotlib for frequency domain
        self.fig, (self.ax1, self.ax2) = plt.subplots(2, 1, figsize=(12, 8))
        plt.ion()
        
        # Create frequency bins for x-axis (pre-computed)
        self.freqs = np.fft.rfftfreq(n_fft, 1/sample_rate).astype(np.float32)
        # Limit to frequencies above G4 (392 Hz) up to 20 kHz
        self.freq_mask = (self.freqs >= 392) & (self.freqs <= 20000)
        self.display_freqs = self.freqs[self.freq_mask]
        self.n_display_freqs = len(self.display_freqs)
        
        # Pre-allocate spectrum arrays
        self.spectrum_full = np.zeros(len(self.freqs), dtype=np.float32)
        self.spectrum_display = np.zeros(self.n_display_freqs, dtype=np.float32)
        
        # Pre-create plot elements
        self.waveform_line, = self.ax1.plot([], [], 'b-', linewidth=1)
        self.x_waveform = np.arange(buffer_size, dtype=np.float32)  # Pre-computed x-axis
        
        # Setup axes once
        self._setup_axes()
        
        # Pre-compute musical note data for faster plotting
        self._setup_note_markers()
    
    def _setup_note_markers(self):
        """Pre-compute musical note marker data for full chromatic scale"""
        # Generate chromatic scale frequencies starting from G4 (392 Hz)
        # Each octave multiplies frequency by 2, each semitone by 2^(1/12)
        
        note_names_chromatic = ['G', 'G#', 'A', 'A#', 'B', 'C', 'C#', 'D', 'D#', 'E', 'F', 'F#']
        semitone_ratio = 2**(1/12)  # Frequency ratio between semitones
        
        # Start with G4 = 392 Hz
        base_freq = 392.0  # G4
        base_octave = 4
        
        frequencies = []
        names = []
        
        # Generate notes up to 20 kHz
        current_freq = base_freq
        octave = base_octave
        note_index = 0  # Start with G
        
        while current_freq <= 20000:
            # Add current note
            frequencies.append(current_freq)
            note_name = note_names_chromatic[note_index]
            names.append(f"{note_name}{octave}")
            
            # Move to next semitone
            current_freq *= semitone_ratio
            note_index += 1
            
            # Handle octave change (after B, next is C of next octave)
            if note_index >= len(note_names_chromatic):
                note_index = 0
                octave += 1
        
        # Convert to numpy arrays for efficiency
        self.note_freqs = np.array(frequencies, dtype=np.float32)
        self.note_names = names
        
        # Create two sets: major notes (for labels) and all notes (for lines)
        major_note_indices = []
        for i, name in enumerate(names):
            # Show labels only for natural notes (no sharps/flats)
            if '#' not in name:
                major_note_indices.append(i)
        
        self.major_note_freqs = self.note_freqs[major_note_indices]
        self.major_note_names = [names[i] for i in major_note_indices]
    
    def _setup_axes(self):
        """Setup plot axes once for efficiency"""
        # Waveform plot setup
        self.ax1.set_xlim(0, self.buffer_size)
        self.ax1.set_ylim(-1, 1)
        self.ax1.set_title('Real-time Audio Waveform')
        self.ax1.set_xlabel('Sample Index')
        self.ax1.set_ylabel('Amplitude')
        self.ax1.grid(True, alpha=0.3)
        
        # Spectrum peaks plot setup
        self.ax2.set_xlim(392, 20000)  # Start from G4 (392 Hz)
        self.ax2.set_ylim(-10, 5)  # Log scale range
        self.ax2.set_xscale('log')
        self.ax2.set_title(f'Top {self.n_peaks} FFT Peaks')
        self.ax2.set_xlabel('Frequency (Hz)')
        self.ax2.set_ylabel('Magnitude (log10)')
        self.ax2.grid(True, alpha=0.3)
        
        plt.tight_layout()
    
    def set_n_peaks(self, n_peaks):
        """Change the number of peaks to display"""
        self.n_peaks = n_peaks
        self.ax2.set_title(f'Top {self.n_peaks} FFT Peaks')
    
    def get_current_peaks(self, n_peaks=3):
        """Get the current top N peaks from the audio buffer - optimized version"""
        with self.buffer_lock:
            # Copy only what we need
            if len(self.audio_buffer) >= self.n_fft:
                self.windowed_data[:] = self.audio_buffer[-self.n_fft:] * self.hanning_window
            else:
                self.windowed_data[:len(self.audio_buffer)] = self.audio_buffer * self.hanning_window[:len(self.audio_buffer)]
                self.windowed_data[len(self.audio_buffer):] = 0
        
        # Optimized FFT computation
        np.fft.rfft(self.windowed_data, n=self.n_fft, out=self.fft_result)
        np.abs(self.fft_result, out=self.magnitude)
        
        # Use log scaling with pre-allocated arrays
        np.maximum(self.magnitude, 1e-10, out=self.magnitude)
        np.log10(self.magnitude, out=self.spectrum_full)
        
        # Extract display frequencies
        self.spectrum_display[:] = self.spectrum_full[self.freq_mask]
        
        # Get top N peaks efficiently
        if n_peaks >= self.n_display_freqs:
            peak_indices = np.arange(self.n_display_freqs)
        else:
            peak_indices = np.argpartition(self.spectrum_display, -n_peaks)[-n_peaks:]
        
        peak_freqs = self.display_freqs[peak_indices]
        peak_mags = self.spectrum_display[peak_indices]
        
        # Sort by magnitude (highest first)
        sorted_indices = np.argsort(peak_mags)[::-1]
        return peak_freqs[sorted_indices], peak_mags[sorted_indices]
    
    def update_audio_data(self, audio_samples):
        """Update the audio buffer with new samples (thread-safe) - optimized"""
        with self.buffer_lock:
            shift_amount = len(audio_samples)
            # Use efficient array operations
            self.audio_buffer[:-shift_amount] = self.audio_buffer[shift_amount:]
            self.audio_buffer[-shift_amount:] = audio_samples.astype(np.float32)
    
    def _compute_fft_spectrum_optimized(self):
        """Optimized FFT spectrum computation using pre-allocated arrays"""
        # Use pre-allocated windowed data
        if len(self.audio_buffer) >= self.n_fft:
            self.windowed_data[:] = self.audio_buffer[-self.n_fft:] * self.hanning_window
        else:
            self.windowed_data[:len(self.audio_buffer)] = self.audio_buffer * self.hanning_window[:len(self.audio_buffer)]
            self.windowed_data[len(self.audio_buffer):] = 0
        
        # Optimized FFT with pre-allocated output
        np.fft.rfft(self.windowed_data, n=self.n_fft, out=self.fft_result)
        np.abs(self.fft_result, out=self.magnitude)
        
        # Optimized log scaling
        np.maximum(self.magnitude, 1e-10, out=self.magnitude)
        np.log10(self.magnitude, out=self.spectrum_full)
        
        # Extract display range efficiently
        self.spectrum_display[:] = self.spectrum_full[self.freq_mask]
        return self.spectrum_display
    
    def _find_top_peaks_optimized(self, spectrum, n_peaks):
        """Optimized peak finding using argpartition"""
        if n_peaks >= len(spectrum):
            peak_indices = np.arange(len(spectrum))
        else:
            # argpartition is faster than argsort for finding top N
            peak_indices = np.argpartition(spectrum, -n_peaks)[-n_peaks:]
        
        peak_freqs = self.display_freqs[peak_indices]
        peak_mags = spectrum[peak_indices]
        
        # Sort by magnitude (highest first)
        sorted_indices = np.argsort(peak_mags)[::-1]
        return peak_freqs[sorted_indices], peak_mags[sorted_indices]
    
    def start_visualization(self):
        """Start the real-time peak FFT visualization loop - optimized"""
        self.running = True
        print(f"Optimized Peak FFT Visualization started (showing top {self.n_peaks} peaks). Close the plot window to stop.")
        
        try:
            while self.running and plt.get_fignums():
                # Get current buffer data (thread-safe)
                with self.buffer_lock:
                    current_buffer = self.audio_buffer.copy()
                
                # Optimized spectrum computation
                spectrum = self._compute_fft_spectrum_optimized()
                
                # Optimized peak finding
                peak_freqs, peak_mags = self._find_top_peaks_optimized(spectrum, self.n_peaks)
                
                # Update waveform plot efficiently
                self.waveform_line.set_data(self.x_waveform, current_buffer)
                
                # Optimized peaks plot update - minimize clear operations
                self.ax2.clear()
                self.ax2.scatter(peak_freqs, peak_mags, c='red', s=10, alpha=0.8, zorder=5)
                
                # Batch set axis properties
                self.ax2.set_xlim(392, 20000)
                self.ax2.set_ylim(-10, 5)
                self.ax2.set_xscale('log')
                self.ax2.set_title(f'Top {self.n_peaks} FFT Peaks')
                self.ax2.set_xlabel('Frequency (Hz)')
                self.ax2.set_ylabel('Magnitude (log10)')
                self.ax2.grid(True, alpha=0.3)
                
                # Add all chromatic scale vertical lines
                for freq in self.note_freqs:
                    self.ax2.axvline(x=freq, color='lightgray', linestyle='-', alpha=0.3, linewidth=0.5)
                
                # Add labels only for major notes (natural notes)
                for freq, name in zip(self.major_note_freqs, self.major_note_names):
                    self.ax2.axvline(x=freq, color='gray', linestyle='--', alpha=0.7, linewidth=1)
                    self.ax2.text(freq, 4, name, rotation=45, fontsize=8, alpha=0.8)
                
                # Optimized redraw
                self.fig.canvas.draw_idle()  # Use draw_idle for better performance
                self.fig.canvas.flush_events()
                
                # Refresh the plot
                plt.pause(self.update_rate_ms / 1000.0)
                
        except KeyboardInterrupt:
            print("\nOptimized Peak FFT Visualization stopped by user.")
        finally:
            self.stop_visualization()
    
    def stop_visualization(self):
        """Stop the visualization"""
        self.running = False
        plt.ioff()
        plt.close('all')


def demo_mic_with_fft_visualization():
    """Demo function combining microphone input with real-time FFT visualization"""
    
    # Create FFT visualizer and microphone input with ultra-fast updates
    fft_visualizer = FFTVisualizer(buffer_size=4096, update_rate_ms=4, sample_rate=44100, n_fft=4096)  # 250 FPS!
    mic_input = MicrophoneInput(samplerate=44100, channels=1, blocksize=128)  # Ultra-low latency
    
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


def demo_mic_with_peak_fft_visualization(n_peaks=10):
    """Demo function combining microphone input with real-time peak FFT visualization"""
    
    global peak_visualizer_instance  # Make it globally accessible
    
    # Create peak FFT visualizer and microphone input with ultra-fast updates
    peak_visualizer_instance = PeakFFTVisualizer(buffer_size=4096, update_rate_ms=4, sample_rate=44100, n_fft=4096, n_peaks=n_peaks)  # 250 FPS!
    mic_input = MicrophoneInput(samplerate=44100, channels=1, blocksize=128)  # Ultra-low latency
    
    # Connect microphone to peak visualizer
    mic_input.set_audio_callback(peak_visualizer_instance.update_audio_data)
    
    print(f"Starting microphone input with real-time peak FFT visualization (showing top {n_peaks} peaks)...")
    print("Close the plot window or press Ctrl+C to stop")
    print("You can now call get_top_peaks() from anywhere to get the current top 3 peaks")
    
    try:
        with mic_input:
            # Start peak visualization (this will block until stopped)
            peak_visualizer_instance.start_visualization()
            
    except KeyboardInterrupt:
        print("\nPeak FFT Demo stopped by user.")
    except Exception as e:
        print(f"Error: {e}")
    finally:
        peak_visualizer_instance.stop_visualization()


# Global convenience function to get current peaks
def get_top_peaks(n_peaks=3):
    """Convenience function to get the current top N peaks from the running visualizer
    
    Args:
        n_peaks (int): Number of top peaks to return (default: 3)
        
    Returns:
        tuple: (frequencies, magnitudes) - arrays of the top N peaks
               frequencies in Hz, magnitudes in log10 scale
               Returns (None, None) if no visualizer is running
    """
    try:
        if 'peak_visualizer_instance' in globals() and peak_visualizer_instance is not None:
            return peak_visualizer_instance.get_current_peaks(n_peaks)
        else:
            print("No peak visualizer is currently running. Start it with demo_mic_with_peak_fft_visualization()")
            return None, None
    except Exception as e:
        print(f"Error getting peaks: {e}")
        return None, None


# Initialize global variable
peak_visualizer_instance = None


class AudioVisualizer:
    """Real-time audio waveform visualizer"""
    
    def __init__(self, buffer_size=2048, update_rate_ms=4):
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
    
    # Create visualizer and microphone input with ultra-fast refresh rate
    visualizer = AudioVisualizer(buffer_size=2048, update_rate_ms=4)  # Ultra-fast refresh: 250 FPS!
    mic_input = MicrophoneInput(samplerate=44100, channels=1, blocksize=128)  # Ultra-low latency
    
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