#!/usr/bin/env python3
"""
Global key listener that detects 'J' key presses regardless of window focus.
Uses pynput library for cross-platform global key detection.
Runs in a separate thread and publishes key state to a shared variable.
"""

import time
import threading
from pynput import keyboard

# Track the last known state of the J key to avoid key repeat
j_key_state = "released"  # Can be "pressed" or "released"


class GlobalKeyListener:
    """Thread-safe global key listener for J key detection"""
    
    def __init__(self):
        self.j_key_state = "released"  # Current state: "pressed" or "released"
        self.state_lock = threading.Lock()  # Thread-safe access to state
        self.listener_thread = None
        self.listener = None
        self.running = False
    
    def get_j_key_state(self):
        """Get the current J key state (thread-safe)"""
        with self.state_lock:
            return self.j_key_state
    
    def _set_j_key_state(self, new_state):
        """Set the J key state (thread-safe)"""
        with self.state_lock:
            self.j_key_state = new_state
    
    def _on_key_press(self, key):
        """Handle key press events"""
        try:
            # Check if the pressed key is 'j' or 'J'
            if hasattr(key, 'char') and key.char and key.char.lower() == 'j':
                # Only update if the key was previously released (avoid key repeat)
                if self.get_j_key_state() == "released":
                    timestamp = time.strftime('%H:%M:%S') + f".{int(time.time() * 1000) % 1000:04d}"
                    print(f"J key PRESSED at {timestamp}")
                    self._set_j_key_state("pressed")
                return True  # Continue listening
                
        except AttributeError:
            # Special keys (ctrl, alt, etc.) don't have char attribute
            pass
        
        return True  # Continue listening
    
    def _on_key_release(self, key):
        """Handle key release events"""
        try:
            # Check if the released key is 'j' or 'J'
            if hasattr(key, 'char') and key.char and key.char.lower() == 'j':
                # Only update if the key was previously pressed (avoid duplicate releases)
                if self.get_j_key_state() == "pressed":
                    timestamp = time.strftime('%H:%M:%S') + f".{int(time.time() * 1000) % 1000:04d}"
                    print(f"J key RELEASED at {timestamp}")
                    self._set_j_key_state("released")
                return True  # Continue listening
        except AttributeError:
            # Special keys (ctrl, alt, etc.) don't have char attribute
            pass
        
        # Stop listener if ESC is pressed
        if key == keyboard.Key.esc:
            print("ESC pressed - stopping key listener")
            self.stop()
            return False  # Stop listener
        
        return True  # Continue listening
    
    def start(self):
        """Start the key listener in a separate thread"""
        if self.running:
            return
        
        self.running = True
        
        def run_listener():
            try:
                with keyboard.Listener(
                    on_press=self._on_key_press,
                    on_release=self._on_key_release
                ) as self.listener:
                    self.listener.join()
            except Exception as e:
                print(f"Key listener error: {e}")
            finally:
                self.running = False
        
        self.listener_thread = threading.Thread(target=run_listener, daemon=True)
        self.listener_thread.start()
        print("Global key listener started in background thread")
    
    def stop(self):
        """Stop the key listener"""
        self.running = False
        if self.listener:
            self.listener.stop()
        print("Key listener stopped")
    
    def is_running(self):
        """Check if the listener is currently running"""
        return self.running


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


def demo_threaded_key_listener():
    """Demo showing how to use the threaded key listener with state access"""
    print("Threaded Global Key Listener Demo")
    print("Press 'J' anywhere to trigger detection")
    print("Press ESC to stop the listener")
    print("=" * 50)
    
    # Create the key listener
    key_listener = GlobalKeyListener()
    
    try:
        # Start the key listener in its own thread
        key_listener.start()
        
        # Main thread can do other work and check key state
        last_printed_state = None
        while key_listener.is_running():
            current_state = key_listener.get_j_key_state()
            
            # Only print when state changes (for demo purposes)
            if current_state != last_printed_state:
                timestamp = time.strftime('%H:%M:%S') + f".{int(time.time() * 1000) % 1000:04d}"
                print(f"Main thread sees J key state: {current_state} at {timestamp}")
                last_printed_state = current_state
            
            # Small sleep to avoid busy waiting
            time.sleep(0.01)
            
    except KeyboardInterrupt:
        print("\nDemo stopped by user")
    except Exception as e:
        print(f"Error: {e}")
        print("Make sure you have the 'pynput' library installed:")
        print("pip install pynput")
    finally:
        key_listener.stop()


def main():
    """Main function - choose between simple or threaded demo"""
    print("Choose demo mode:")
    print("1. Simple key listener (original)")
    print("2. Threaded key listener with state access")
    
    choice = input("Enter choice (1 or 2, default=2): ").strip()
    
    if choice == "1":
        main_simple()
    else:
        demo_threaded_key_listener()


def main_simple():
    """Original main function for simple key listener"""
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