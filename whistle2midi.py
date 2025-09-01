#!/usr/bin/python3

import time
import mido
import sounddevice as sd
import numpy as np
import matplotlib.pyplot as plt
import threading


def main():
    # send_simple_note()
    sound_device_demo()

def sound_device_demo():
    """Demo function showing real-time microphone input with simple waveform plotting"""
    
    # Shared audio buffer between threads
    buffer_size = 2048  # Show about 0.05 seconds at 44100 Hz
    audio_buffer = np.zeros(buffer_size)
    buffer_lock = threading.Lock()
    
    def audio_callback(indata, frames, time, status):
        nonlocal audio_buffer
        if status:
            print(f"Audio status: {status}")
        
        # Get mono audio samples
        audio_samples = indata[:, 0]
        
        # Update the shared buffer (thread-safe)
        with buffer_lock:
            shift_amount = len(audio_samples)
            audio_buffer[:-shift_amount] = audio_buffer[shift_amount:]
            audio_buffer[-shift_amount:] = audio_samples
    
    print("Starting simple audio waveform plotting...")
    print("Close the plot window or press Ctrl+C to stop")
    
    # Setup matplotlib in main thread
    plt.figure(figsize=(10, 6))
    plt.ion()
    
    try:
        # Start audio stream
        with sd.InputStream(callback=audio_callback, 
                          channels=1,
                          samplerate=44100,
                          blocksize=1024):
            
            # Main plotting loop in main thread
            while plt.get_fignums():
                # Get current buffer data (thread-safe)
                with buffer_lock:
                    current_buffer = audio_buffer.copy()
                
                # Update plot
                plt.clf()
                plt.plot(current_buffer, 'b-', linewidth=1)
                plt.ylim(-1, 1)
                plt.title('Real-time Audio Waveform')
                plt.xlabel('Sample Index')
                plt.ylabel('Amplitude')
                plt.grid(True, alpha=0.3)
                
                # Refresh the plot
                plt.pause(0.05)  # Update every 50ms
                
    except KeyboardInterrupt:
        print("\nAudio stream stopped.")
    except Exception as e:
        print(f"Error: {e}")
    finally:
        plt.ioff()
        plt.close('all')


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