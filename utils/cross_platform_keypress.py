#!/usr/bin/env python3
"""
Cross-platform keypress handling utility.

This module provides a unified interface for keypress detection across
Windows, macOS, and Linux platforms.
"""

import os
import sys
import time
import threading
from typing import Optional, Callable


class CrossPlatformKeypress:
    """
    Cross-platform keypress handler that works on Windows, macOS, and Linux.
    """
    
    def __init__(self):
        self.platform = sys.platform.lower()
        self._setup_platform_specific()
    
    def _setup_platform_specific(self):
        """Setup platform-specific keypress handling."""
        if self.platform == "win32":
            self._setup_windows()
        elif self.platform in ["darwin", "linux"]:
            self._setup_unix()
        else:
            raise RuntimeError(f"Unsupported platform: {self.platform}")
    
    def _setup_windows(self):
        """Setup Windows-specific keypress handling."""
        try:
            import msvcrt
            self._msvcrt = msvcrt
            self._has_key = msvcrt.kbhit
            self._get_key = msvcrt.getch
        except ImportError:
            raise RuntimeError("Windows keypress handling requires msvcrt module")
    
    def _setup_unix(self):
        """Setup Unix-like systems (macOS, Linux) keypress handling."""
        try:
            import termios
            import tty
            self._termios = termios
            self._tty = tty
            self._stdin_fd = sys.stdin.fileno()
            self._old_settings = None
        except ImportError:
            raise RuntimeError("Unix keypress handling requires termios module")
    
    def wait_for_key(self, key: str, prompt: str = "", timeout: Optional[float] = None) -> bool:
        """
        Wait for a specific key press.
        
        Args:
            key: Key to wait for (case-insensitive)
            prompt: Optional prompt message to display
            timeout: Optional timeout in seconds
            
        Returns:
            True if the key was pressed, False if timeout occurred
        """
        if prompt:
            print(prompt, flush=True)
        
        start_time = time.time()
        
        while True:
            if timeout and (time.time() - start_time) > timeout:
                return False
            
            if self._check_key_pressed():
                pressed_key = self._get_pressed_key()
                if pressed_key and pressed_key.lower() == key.lower():
                    return True
            
            time.sleep(0.02)  # Small delay to prevent high CPU usage
    
    def _check_key_pressed(self) -> bool:
        """Check if any key is pressed."""
        if self.platform == "win32":
            return self._has_key()
        else:
            # For Unix systems, we need to check if input is available
            try:
                import select
                return select.select([sys.stdin], [], [], 0) == ([sys.stdin], [], [])
            except ImportError:
                # Fallback: try to read without blocking
                return self._check_unix_key_fallback()
    
    def _check_unix_key_fallback(self) -> bool:
        """Fallback method for Unix key checking."""
        try:
            # Set terminal to raw mode temporarily
            old_settings = self._termios.tcgetattr(self._stdin_fd)
            self._tty.setraw(self._stdin_fd)
            
            # Check if input is available
            import fcntl
            flags = fcntl.fcntl(self._stdin_fd, fcntl.F_GETFL)
            fcntl.fcntl(self._stdin_fd, fcntl.F_SETFL, flags | os.O_NONBLOCK)
            
            try:
                ch = sys.stdin.read(1)
                return len(ch) > 0
            except (IOError, OSError):
                return False
            finally:
                # Restore terminal settings
                fcntl.fcntl(self._stdin_fd, fcntl.F_SETFL, flags)
                self._termios.tcsetattr(self._stdin_fd, self._termios.TCSADRAIN, old_settings)
        except (ImportError, OSError):
            return False
    
    def _get_pressed_key(self) -> Optional[str]:
        """Get the currently pressed key."""
        if self.platform == "win32":
            try:
                ch = self._get_key()
                if isinstance(ch, bytes):
                    return ch.decode('utf-8', errors='ignore')
                return ch
            except (UnicodeDecodeError, OSError):
                return None
        else:
            # For Unix systems, read the key
            try:
                # Set terminal to raw mode
                old_settings = self._termios.tcgetattr(self._stdin_fd)
                self._tty.setraw(self._stdin_fd)
                
                # Read one character
                ch = sys.stdin.read(1)
                
                # Restore terminal settings
                self._termios.tcsetattr(self._stdin_fd, self._termios.TCSADRAIN, old_settings)
                
                return ch
            except (OSError, UnicodeDecodeError):
                return None
    
    def start_key_monitoring(self, callback: Callable[[str], None], keys: list = None):
        """
        Start monitoring for key presses in a background thread.
        
        Args:
            callback: Function to call when a key is pressed
            keys: List of keys to monitor (None for all keys)
        """
        def monitor():
            while True:
                if self._check_key_pressed():
                    key = self._get_pressed_key()
                    if key and (keys is None or key.lower() in [k.lower() for k in keys]):
                        callback(key)
                time.sleep(0.02)
        
        thread = threading.Thread(target=monitor, daemon=True)
        thread.start()
        return thread


# Convenience functions for common use cases
def wait_for_r_key(prompt: str = "Press 'r' to continue...", timeout: Optional[float] = None) -> bool:
    """
    Wait for 'r' key press (case-insensitive).
    
    Args:
        prompt: Prompt message to display
        timeout: Optional timeout in seconds
        
    Returns:
        True if 'r' was pressed, False if timeout occurred
    """
    handler = CrossPlatformKeypress()
    return handler.wait_for_key('r', prompt, timeout)


def wait_for_any_key(prompt: str = "Press any key to continue...", timeout: Optional[float] = None) -> Optional[str]:
    """
    Wait for any key press.
    
    Args:
        prompt: Prompt message to display
        timeout: Optional timeout in seconds
        
    Returns:
        The pressed key or None if timeout occurred
    """
    handler = CrossPlatformKeypress()
    if prompt:
        print(prompt, flush=True)
    
    start_time = time.time()
    
    while True:
        if timeout and (time.time() - start_time) > timeout:
            return None
        
        if handler._check_key_pressed():
            key = handler._get_pressed_key()
            if key:
                return key
        
        time.sleep(0.02)


if __name__ == "__main__":
    # Test the keypress functionality
    print("Testing cross-platform keypress handling...")
    print("Press 'r' to test, or 'q' to quit")
    
    handler = CrossPlatformKeypress()
    
    while True:
        if handler.wait_for_key('r', "Press 'r' to test (or 'q' to quit): "):
            print("✅ 'r' key detected!")
        elif handler.wait_for_key('q', "Press 'q' to quit: "):
            print("👋 Goodbye!")
            break
