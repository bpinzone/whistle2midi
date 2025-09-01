#!/usr/bin/env python3
"""
Global key listener that detects 'J' key presses regardless of window focus.
Uses pynput library for cross-platform global key detection.
"""

import time
from pynput import keyboard

# Track the last known state of the J key to avoid key repeat
j_key_state = "released"  # Can be "pressed" or "released"


def on_key_press(key):
    """Handle key press events"""
    global j_key_state
    
    try:
        # Check if the pressed key is 'j' or 'J'
        if hasattr(key, 'char') and key.char and key.char.lower() == 'j':
            # Only print if the key was previously released (avoid key repeat)
            if j_key_state == "released":
                timestamp = time.strftime('%H:%M:%S') + f".{int(time.time() * 1000) % 1000:04d}"
                print(f"J key PRESSED at {timestamp}")
                j_key_state = "pressed"
            return True  # Continue listening
            
    except AttributeError:
        # Special keys (ctrl, alt, etc.) don't have char attribute
        pass
    
    return True  # Continue listening


def on_key_release(key):
    """Handle key release events"""
    global j_key_state
    
    try:
        # Check if the released key is 'j' or 'J'
        if hasattr(key, 'char') and key.char and key.char.lower() == 'j':
            # Only print if the key was previously pressed (avoid duplicate releases)
            if j_key_state == "pressed":
                timestamp = time.strftime('%H:%M:%S') + f".{int(time.time() * 1000) % 1000:04d}"
                print(f"J key RELEASED at {timestamp}")
                j_key_state = "released"
            return True  # Continue listening
    except AttributeError:
        # Special keys (ctrl, alt, etc.) don't have char attribute
        pass
    
    # Stop listener if ESC is pressed
    if key == keyboard.Key.esc:
        print("ESC pressed - stopping key listener")
        return False  # Stop listener
    
    return True  # Continue listening


def main():
    """Main function to start the global key listener"""
    print("Global Key Listener Started")
    print("Press 'J' anywhere to trigger detection")
    print("Press ESC to stop the listener")
    print("=" * 50)
    
    try:
        # Create and start the global key listener
        with keyboard.Listener(
            on_press=on_key_press,
            on_release=on_key_release
        ) as listener:
            # Keep the script running
            listener.join()
            
    except KeyboardInterrupt:
        print("\nKey listener stopped by user")
    except Exception as e:
        print(f"Error: {e}")
        print("Make sure you have the 'pynput' library installed:")
        print("pip install pynput")


if __name__ == "__main__":
    main() 