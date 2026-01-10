
import multiprocessing

_msg_queue = None

def init(queue):
    global _msg_queue

    _msg_queue = queue

def set_text(tag, text, main_widget=None):
    global _msg_queue

    if _msg_queue is not None:
        try:
            _msg_queue.put({'type': 'waitinfo', 'tag': tag, 'text': text})
        except:
            pass
        return

    # Fallback to direct UI update (Main Process only)
    try:
        widget = main_widget.ids["waitinfo_" + tag]

        widget.text = text + " "
        if text is None or text == "":
            widget.disabled = True
        else:
            widget.disabled = False
    except:
        pass
