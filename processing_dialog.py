
import tkinter as tk
from PIL import Image, ImageTk
import time
from concurrent.futures import ThreadPoolExecutor


class _ProcessingDialog():

    def __init__(self, gif_path):
        self.parent = tk.Tk()
        self.hide()
        self.parent.overrideredirect(True)  # フレームを非表示

        # ダイアログをモーダルに設定（ユーザーが閉じられない）
        self.parent.resizable(False, False)
        
        # 閉じるボタンを無効化
        self.parent.protocol("WM_DELETE_WINDOW", lambda: None)
        
        # GIFアニメーションの設定
        self.gif_path = gif_path
        self.gif_frames = self._load_gif_frames()
        self.current_frame = 0
        self.animation_label = tk.Label(self.parent)
        self.animation_label.pack(side=tk.LEFT, padx=10, pady=10)
        
        # 処理中テキスト
        self.text_label = tk.Label(
            self.parent, 
            text="Processing...", 
            font=("Arial", 14, "bold"),
            padx=20
        )
        self.text_label.pack(side=tk.RIGHT, padx=10, pady=10)
                
        # ダイアログを画面中央に表示
        self._center_dialog()

        # 属性設定
        self.parent.attributes('-topmost', True)
        self.parent.attributes('-alpha', 0.8)

        self._animate()
        self._maintain_front()
    
    def _load_gif_frames(self):
        """GIFファイルをフレームに分割して読み込む"""
        frames = []
        try:
            # GIFをフレームごとに読み込み
            gif_image = Image.open(self.gif_path)
            for frame in range(0, gif_image.n_frames):
                gif_image.seek(frame)
                frames.append(ImageTk.PhotoImage(master=self.parent, image=gif_image.resize((100, 100))))
            return frames

        except Exception as e:
            print(f"GIF読み込みエラー: {e}")
            # エラー時は代替画像を表示
            return [ImageTk.PhotoImage(Image.new('RGBA', (100, 100), (240, 240, 240)))]
    
    def _animate(self):
        """GIFアニメーションを更新"""
        if self.gif_frames:
            self.animation_label.configure(image=self.gif_frames[self.current_frame])
            self.current_frame = (self.current_frame + 1) % len(self.gif_frames)

    def _center_dialog(self):
        """ダイアログを画面中央に配置"""
        self.parent.update_idletasks()
        width = 250
        height = 100
        x = (self.parent.winfo_screenwidth() // 2) - (width // 2)
        y = (self.parent.winfo_screenheight() // 2) - (height // 2)
        self.parent.geometry(f'{width}x{height}+{x}+{y}')

    def _maintain_front(self):
        """Tkinterレベルでの最前面維持"""
        try:
            self.parent.lift()
            #self.parent.attributes('-topmost', True)
            #self.parent.focus_force()
            self.parent.after(50, self._maintain_front)
        except:
            pass

    def show(self):
        """ダイアログを表示"""
        self.parent.deiconify()
        self.parent.grab_set()

    def update(self, sleep_time=0.05):
        self._animate()
        self.parent.update()
        time.sleep(sleep_time)
    
    def hide(self):
        """ダイアログを閉じる"""
        self.parent.withdraw()
        self.parent.grab_release()
        #self.parent.destroy()

__dialog = None

def create_processing_dialog():
    global __dialog
    __dialog = _ProcessingDialog("spinner.gif")
    
def show_processing_dialog():
    global __dialog
    __dialog.show()

def update_processing_dialog(sleep_time=0.05):
    global __dialog
    __dialog.update(sleep_time)

def hide_processing_dialog():
    global __dialog
    __dialog.hide()

def wait_prosessing(process, arg):
    show_processing_dialog()
    with ThreadPoolExecutor(max_workers=1) as executor:
        future = executor.submit(process, arg)
        while not future.done():
            update_processing_dialog(sleep_time=0.05)
        result = future.result()
    hide_processing_dialog()

    return result