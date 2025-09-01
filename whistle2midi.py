#!/usr/bin/python3

import time
import mido
import sounddevice as sd
import numpy as np


def main():
    # send_simple_note()
    sound_device_demo()

def sound_device_demo():
    """Demo function showing real-time microphone input using SoundDevice"""
    
    def audio_callback(indata, frames, time, status):
        if status:
            print(f"Audio status: {status}")
        
        # indata is already a numpy array with shape (frames, channels)
        audio_samples = indata[:, 0]  # Get mono channel (first channel)
        
        # Print some stats about the audio
        volume = np.sqrt(np.mean(audio_samples**2))  # RMS volume
        max_val = np.max(np.abs(audio_samples))
        print(f"Volume: {volume:.4f}, Max: {max_val:.4f}, Samples: {len(audio_samples)}")
    
    print("Starting audio stream... Speak into your microphone!")
    print("Press Ctrl+C to stop")
    
    try:
        # Start audio stream
        with sd.InputStream(callback=audio_callback, 
                          channels=1,           # Mono
                          samplerate=44100,     # Sample rate
                          blocksize=1024):      # Buffer size
            sd.sleep(10000)  # Keep running for 10 seconds
    except KeyboardInterrupt:
        print("\nAudio stream stopped.")


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