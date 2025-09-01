#!/usr/bin/python3

import time
import mido
import sounddevice as sd
import numpy as np
import matplotlib.pyplot as plt
import threading


def main():
    demo_mic_with_visualization()

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