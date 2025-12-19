
from multiprocessing import Process, Queue, Event, shared_memory
import numpy as np
from queue import Empty
import time

import config
import effects
import pipeline
import splitimage

def worker_function(copy_config, worker_id, task_queue, result_queue, stop_event):
    """ワーカー関数（グローバルスコープに定義）"""
    config._config = copy_config
    
    current_effects = effects.create_effects()
    effects.reeffect_all(current_effects, 1)

    while not stop_event.is_set():
        try:
            task = task_queue.get(timeout=0.1)
            if task is None:
                break
            
            # 共有メモリの情報を受け取る
            tile_id, shm_name, shape, dtype_str, params, crop, version, lv1reset = task

            if tile_id[0] != worker_id:
                task_queue.put(task)  # 再度キューに戻す
                continue

            # 共有メモリから配列を取得（コピー不要）
            shm = shared_memory.SharedMemory(name=shm_name)
            tile_array = np.ndarray(shape, dtype=dtype_str, buffer=shm.buf)
            
            # 処理を実行
            result = process_tile(tile_array, current_effects, crop, params, lv1reset)
            
            # 結果も共有メモリに書き込む
            tile_array[:] = result[:]
            result_queue.put((tile_id, shm.name, result.shape, str(result.dtype), version))

            # 共有メモリを閉じる
            shm.close()
            
        except Empty:
            continue
        except Exception as e:
            print(f"Worker {worker_id} error: {e}")

def process_tile(tile_data, current_effects, crop, params, lv1reset):
    """タイル処理関数（グローバルスコープに定義）"""
    effects.reeffect_all(current_effects, 1)
    img2 = pipeline.pipeline2(tile_data, crop, current_effects, params["param"], None, params["efconfig"], lv1reset)
    
    return img2

class DynamicImageProcessor:
    def __init__(self, num_workers=4):
        self.num_workers = num_workers
        self.task_queue = Queue(maxsize=100)
        self.result_queue = Queue()
        self.stop_event = Event()
        self.workers = []
        self.expected_results = 0  # 期待する結果の数
        
    def ensure_started(self):
        """ワーカーが起動していなければ起動する"""
        if not self.workers:
            self.start()

    def start(self):
        """ワーカープロセスを起動"""
        for i in range(self.num_workers):
            p = Process(target=worker_function,
                        args=(config._config, i, self.task_queue, self.result_queue, self.stop_event),
                        daemon=True )
            p.start()
            self.workers.append(p)
    
    def submit_tiles(self, image, params, version, lv1reset):
        """画像をタイルに分割して処理キューに投入"""
        h, w = image.shape[:2]
        
        # ワーカーが起動しているか確認
        self.ensure_started()

        # 古いタスクをクリア（オプション：最新のパラメータのみ処理）
        self._clear_queue(self.task_queue)

        # 画像分割
        blocks, crops, split_info = splitimage.split_image_with_overlap(
            image, block_height=((h + 7) // 8 * 8) // 2 + 32, block_width=((w + 7) // 8 * 8) // 2 + 32, overlap=32, crops_out=True)
        
        tile_id = 0
        for block in blocks:
            # 共有メモリに書き込み
            shm = shared_memory.SharedMemory(create=True, size=block.nbytes)
            shm_array = np.ndarray(block.shape, dtype=block.dtype, buffer=shm.buf)
            shm_array[:] = block[:]
            shm.close()
            
            # タスクをキューに追加
            self.task_queue.put((
                (tile_id, split_info),
                shm.name,  # 共有メモリの名前だけ
                block.shape,
                str(block.dtype),
                params.copy(),
                crops[tile_id],
                version,
                lv1reset,
            ))
            tile_id += 1

        self.expected_results = tile_id  # 期待する結果数を記録
        print(f"Submitted {tile_id} tiles")
    
    def _clear_queue(self, q):
        """キューをクリア"""
        try:
            while True:
                q.get_nowait()
        except Empty:
            pass
    
    def collect_results(self, current_version, timeout=1):
        """結果を収集（最新バージョンのみ）"""
        results = []
        while len(results) < self.expected_results:
            try:
                tile_id, shm_name, shape, dtype_str, version = self.result_queue.get(timeout=timeout)

                # 共有メモリから読み取り
                shm = shared_memory.SharedMemory(name=shm_name)

                # 最新バージョンの結果のみ採用
                if version == current_version:
                    result = np.ndarray(shape, dtype=dtype_str, buffer=shm.buf).copy()
                    results.append((tile_id, result))
                    print(f"Collected tile {tile_id} for version {version}")

                shm.close()
                shm.unlink()

            except Empty:
                pass

        return results
    
    def stop(self):
        """ワーカーを停止"""
        self.stop_event.set()
        # 終了シグナルを送信
        for _ in range(self.num_workers):
            self.task_queue.put(None)
        
        for p in self.workers:
            p.join(timeout=1)
            if p.is_alive():
                p.terminate()
