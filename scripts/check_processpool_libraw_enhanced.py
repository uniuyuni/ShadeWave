import multiprocessing as mp
from concurrent.futures import ProcessPoolExecutor
import importlib


def child():
    import libraw_enhanced as lre
    try:
        inner = importlib.import_module("libraw_enhanced.libraw_enhanced")
    except ModuleNotFoundError:
        inner = lre
    return {
        "top_file": getattr(lre, "__file__", None),
        "top_has_imread": hasattr(lre, "imread"),
        "inner_core_available": getattr(inner, "_CORE_AVAILABLE", None),
    }


def main():
    print("start_method", mp.get_start_method())
    with ProcessPoolExecutor(max_workers=1) as ex:
        print(ex.submit(child).result())


if __name__ == "__main__":
    main()
