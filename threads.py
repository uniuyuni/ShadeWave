
import threading

# Numbaの並行アクセスを防ぐためのロック
numba_lock = threading.Lock()

def lock_numba(func):
    def wrapper(*args, **kwargs):
        with numba_lock:
            return func(*args, **kwargs)
    return wrapper

editor_lock = threading.Lock()

def lock_editor(func):
    def wrapper(*args, **kwargs):
        with editor_lock:
            return func(*args, **kwargs)
    return wrapper
