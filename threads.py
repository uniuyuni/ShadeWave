
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

# MaskEditor2 の tcg_info['matrix'] / texture_size は描画スレッドと UI/overlay 更新で共有される。
# セグメントマスク用の一時 matrix swap が再入するため RLock にする。
mask_editor_matrix_lock = threading.RLock()

# メインスレッドの primary_param / imgset と描画スレッドの競合を防ぐ（on_select→empty_image など同一スレッド再入のため RLock）
primary_param_lock = threading.RLock()

def lock_primary_param(func):
    def wrapper(*args, **kwargs):
        with primary_param_lock:
            return func(*args, **kwargs)
    return wrapper
