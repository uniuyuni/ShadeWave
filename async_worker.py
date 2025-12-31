
import multiprocessing
from multiprocessing import Process, Queue, Event, shared_memory
import numpy as np
import time
from queue import Empty
import logging
import traceback
import sys
import copy

import effects
import config

def worker_process(input_queue, result_queue, stop_event, config_dict, latest_tasks):    
    """
    Background worker process.
    Continuously pulls tasks from input_queue and processes them.
    """    
    # 子プロセスで読み込まれているモジュールを確認
    loaded_modules = list(sys.modules.keys())
    print(f"子プロセス {multiprocessing.current_process().name} で読み込まれているモジュール:")
    
    # 特定のモジュールをチェック
    check_modules = ['numpy', 'matplotlib', 'kivy', 'kivymd']
    for module in check_modules:
        if any(module in m for m in loaded_modules):
            print(f"  ⚠️ {module} が読み込まれています")
            # 具体的にどのサブモジュールか表示
            related = [m for m in loaded_modules if module in m]
            print(f"     {related[:5]}")  # 最初の5つを表示

    # Restore configuration in the worker process
    config._config = config_dict
    
    # Create independent effect instances for the worker
    worker_effects = effects.create_effects()
    
    while not stop_event.is_set():
        try:
            # Get task with timeout to allow checking stop_event
            task = input_queue.get(timeout=0.1)
            
            if task is None:
                continue
                
            task_id, effect_name, shm_name, shape, dtype_str, params, efconfig = task
            
            # Check if this task is already cancelled
            if effect_name in latest_tasks:
                if task_id < latest_tasks[effect_name]:
                    # Cancelled
                    # Need to cleanup input SHM? 
                    # Main process cleans up when it receives result. 
                    # If we silently drop, Main process waits forever? 
                    # No, Main tracks SHM. We should probably signal "Cancelled" 
                    # or Main should rely on "Latest Task ID" logic too?
                    # Main polls. If Main sees new task submitted, it knows old one is dead.
                    # But SHM cleanup needs to happen.
                    
                    # Simplest: Send "cancelled" result so Main can cleanup SHM
                    result_queue.put({'task_id': task_id, 'status': 'cancelled'})
                    
                    # Close input SHM
                    try:
                        shm = shared_memory.SharedMemory(name=shm_name)
                        shm.close()
                    except:
                        pass
                    continue
            
            try:
                # Access shared memory
                existing_shm = shared_memory.SharedMemory(name=shm_name)
                # Create numpy array from shared memory (no copy yet)
                input_image = np.ndarray(shape, dtype=dtype_str, buffer=existing_shm.buf)
                
                # We need a copy to process because we shouldn't modify the source in place 
                # if it might be used by others, but here strictly 1-to-1.
                # However, many effects return a new array.
                
                # Find the effect instance
                # Currently we assume effects are in the first group (lv0~lv4 flattened? No, effects structure is complex)
                # We need a way to locate the effect. 
                # For simplicity, let's assume we pass enough info to reconstruct or find it.
                # Actually, `effects.create_effects()` returns a structure.
                # We need to find the specific effect by name.
                
                target_effect = None
                target_effect = None
                for layer in worker_effects:
                    # Search by key match first (fast)
                    if effect_name in layer:
                         target_effect = layer[effect_name]
                         break
                    # Fallback: Search by Class Name
                    for inst in layer.values():
                        if inst.__class__.__name__ == effect_name:
                            target_effect = inst
                            break
                    if target_effect:
                        break
                
                if target_effect:
                    # Sync parameters
                    # We assume params is a dictionary of parameters for the effect
                    # But wait, `make_diff` takes `param` (global param) and `efconfig`.
                    # The `params` passed in task should be the global param dictionary.
                    
                    # Execute heavy processing
                    # Note: We are not calling make_diff/apply_diff logic directly because we want to force the heavy logic.
                    # But usually `make_diff` IS the heavy logic.
                    
                    # Update effect internal state if needed (some effects might need set2param logic equivalent?)
                    # Generally parameters are passed to make_diff.
                    
                    result_image = None
                    diff = target_effect.make_diff(input_image, params, efconfig)
                    if diff is not None:
                        result_image = target_effect.apply_diff(input_image)
                    else:
                        result_image = input_image # No change
                    
                    # Write result to NEW shared memory
                    # (We cannot reuse input SHM as it might be read by others or main process?)
                    # Actually, for efficiency, can we? 
                    # Use a new SHM for the result to avoid race conditions.
                    
                    result_shm = shared_memory.SharedMemory(create=True, size=result_image.nbytes)
                    result_array = np.ndarray(result_image.shape, dtype=result_image.dtype, buffer=result_shm.buf)
                    result_array[:] = result_image[:]
                    
                    # Send result back
                    result_queue.put({
                        'task_id': task_id,
                        'status': 'success',
                        'shm_name': result_shm.name,
                        'shape': result_image.shape,
                        'dtype': str(result_image.dtype)
                    })
                    
                    # Close result_shm in worker (it's still open in main via name)
                    result_shm.close()
                    
                else:
                    logging.error(f"Worker: Effect {effect_name} not found.")
                    result_queue.put({'task_id': task_id, 'status': 'error', 'message': 'Effect not found'})

                # Close input shm
                existing_shm.close()

            except Exception as e:
                logging.error(f"Worker processing error: {e}")
                traceback.print_exc()
                result_queue.put({'task_id': task_id, 'status': 'error', 'message': str(e)})
                
        except Empty:
            continue
        except Exception as e:
            logging.error(f"Worker loop error: {e}")

class AsyncWorker:
    def __init__(self):
        self.process = None
        self.input_queue = Queue()
        self.result_queue = Queue()
        self.stop_event = Event()
        self.active_shms = set() # Track SHMs to unlink them if needed
        self.task_counter = 0
        with multiprocessing.Manager() as mp_manager:
            self.latest_tasks = mp_manager.dict() # effect_name -> task_id

    def start(self):
        if self.process is None or not self.process.is_alive():
            self.stop_event.clear()
            # Pass current config to worker
            self.process = Process(
                target=worker_process,
                name="ASyncWorker", 
                args=(self.input_queue, self.result_queue, self.stop_event, config._config, self.latest_tasks)
            )
            self.process.daemon = True
            self.process.start()
            logging.info("AsyncWorker started.")

    def stop(self):
        if self.process:
            self.stop_event.set()
            self.process.join(timeout=0.1) # Short wait
            if self.process.is_alive():
                logging.warning(f"Terminating worker process {self.process.pid}...")
                self.process.terminate()
                self.process.join(timeout=0.1)
                
            if self.process.is_alive():
                logging.warning(f"Killing worker process {self.process.pid}...")
                try:
                    self.process.kill() # Force kill
                    self.process.join()
                except AttributeError:
                    # In case python version is old or process object issues
                    import os
                    import signal
                    try:
                        os.kill(self.process.pid, signal.SIGKILL)
                    except:
                        pass
                        
            self.process = None
            logging.info("AsyncWorker stopped.")
            
        # Clean up queues (drain)
        try:
            while not self.input_queue.empty():
                self.input_queue.get_nowait()
        except:
            pass
            
    def restart(self):
        """
        Forcefully restart the worker.
        Useful for cancelling running heavy tasks or recovering from errors.
        SAFELY recreates queues to avoid corruption from terminated process.
        """
        self.stop()
        
        # Recreate queues to avoid corruption
        self.input_queue = Queue()
        self.result_queue = Queue()
        self.stop_event = Event()
        
        self.start()
        
        # We might need to handle leftover SHMs here, but it's tricky.

    def submit_task(self, effect_name, image, params, efconfig):
        """
        Submit a task to the background worker.
        Returns task_id.
        """
        self.start() # Ensure started
        
        self.task_counter += 1
        task_id = self.task_counter
        
        # Create SharedMemory for input image
        shm = shared_memory.SharedMemory(create=True, size=image.nbytes)
        shm_array = np.ndarray(image.shape, dtype=image.dtype, buffer=shm.buf)
        shm_array[:] = image[:]
        
        # We keep track of this SHM to unlink it later if needed (or rely on worker to close and we unlink?)
        # Strategy: Main creates, Worker opens & closes. Main unlinks after Worker is done?
        # Actually, simpler: Main creates, puts name in queue. Worker opens, reads, closes.
        # Main MUST unlink it. But when? 
        # If we just put it in queue, we lose reference. 
        # Ideally: Main creates -> Worker reads -> Worker signals done -> Main unlinks.
        # But for input, we can probably unlink immediately after putting to queue IF we trust the OS keeps it alive 
        # as long as it's open? No, in POSIX shm_unlink removes the name but keeps the segment if open. 
        # But in Python shared_memory, strict lifecycle is better.
        # Let's let the worker be responsible for closing its handle, but unlinking must happen 
        # after worker is done reading. 
        
        # To simplify: We won't strictly optimize SHM lifecycle perfectly in this V1.
        # We will let the "inputs" be managed by a simple mechanism: 
        # The worker copies the data immediately then closes. 
        # So multiple heavy tasks might consume RAM. 
        # For now, let's just assume we unlink in `poll_results` or similar? 
        # No, `input` SHM is transient. 
        # Use a strategy: Main creates SHM, sends to Worker. Worker reads. 
        # Main can unlink after some time? No.
        # Correct approach: Worker sends back an acknowledgement? Too complex.
        # Alternative: We use the same SHM for return? No, shape might change.
        
        # Let's track input SHMs associated with tasks.
        
        task = (task_id, effect_name, shm.name, image.shape, str(image.dtype), params, efconfig)
        
        # Safely copy efconfig to remove non-picklable objects (processor)
        # Assuming efconfig is a simple object, we can use copy.copy or create new.
        # But efconfig doesn't have clone method.
        # We can just manually copy key attributes we need.
        safe_efconfig = copy.copy(efconfig)
        safe_efconfig.processor = None
        
        task = (task_id, effect_name, shm.name, image.shape, str(image.dtype), params, safe_efconfig)
        self.input_queue.put(task)
        
        # Update latest task ID for this effect
        self.latest_tasks[effect_name] = task_id
        
        # We must keep shm open in this process until worker picks it up?
        # Actually, if we close it here, and unlink, it might disappear before worker opens it.
        # If we don't unlink, it leaks.
        # Hack: Unlink immediately? 
        # "On Windows, looking up the name... fails... if it has been marked for deletion."
        # On Mac/Linux, unlink removes the name. New opens fail.
        # So we must NOT unlink until worker has opened it.
        # We will add it to a list and check in poll logic? 
        # Or cleaner: Worker sends an event "Input Read Complete"?
        # Let's stick to a simpler approach: 
        # Input SHMs are tracked in `self.pending_inputs`. 
        # When result comes back (or error), we unlink the input SHM.
        
        self.active_shms.add((task_id, shm))
        
        return task_id

    def cancel_effect(self, effect_name):
        """
        Cancel pending tasks for a specific effect.
        """
        # Set latest task to a future ID (or current max + 1) to invalidate pending
        # But we don't know pending IDs easily.
        # Just ensure any task currently in queue with id <= current is skipped.
        # We can just increment counter? No, counter is global.
        # We set latest_tasks[effect_name] = self.task_counter + 1
        # So any task (id <= task_counter) will be seen as old.
        self.latest_tasks[effect_name] = self.task_counter + 1
        
        # Also force restart to stop CPU usage immediately
        self.restart()

    def cancel_all(self):
        """
        Cancel all pending tasks.
        Ideally we purge the queue.
        """
        while not self.input_queue.empty():
            try:
                task = self.input_queue.get_nowait()
                # If we pulled a task, we need to clean up its SHM if we were tracking it
                # But it's hard to correlate without parsing.
                # Simplified: Just increment a "cancellation_token" or "min_task_id"
                # and ignore results from older tasks.
                pass
            except Empty:
                break
        
        # Clean up active SHMs that will never return?
        # This is a potential leak source if we just drop tasks.
        # Better strategy: Do not drop tasks, let them process (fast fail) or just consume inputs.
        # Or, track task_id and cleaning up.
        pass

    def poll_results(self):
        """
        Yields (task_id, result_image_or_None, error_msg).
        """
        results = []
        while not self.result_queue.empty():
            try:
                res = self.result_queue.get_nowait()
                task_id = res['task_id']
                
                # Cleanup Input SHM for this task
                to_remove = None
                for tid, shm in self.active_shms:
                    if tid == task_id:
                        shm.close()
                        shm.unlink()
                        to_remove = (tid, shm)
                        break
                if to_remove:
                    self.active_shms.remove(to_remove)

                if res['status'] == 'success':
                    # Get result from SHM
                    shm_name = res['shm_name']
                    shape = res['shape']
                    dtype_str = res['dtype']
                    
                    try:
                        r_shm = shared_memory.SharedMemory(name=shm_name)
                        r_array = np.ndarray(shape, dtype=dtype_str, buffer=r_shm.buf)
                        # Copy to local
                        img = r_array.copy()
                        r_shm.close()
                        r_shm.unlink() # We own the result SHM now
                        
                        results.append((task_id, img, None))
                    except Exception as e:
                        logging.error(f"Failed to read result SHM: {e}")
                        results.append((task_id, None, str(e)))
                else:
                    results.append((task_id, None, res.get('message', 'Unknown error')))
                    
            except Empty:
                break
        
        return results

