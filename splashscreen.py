
import tkinter as tk
import sys
import time

_root = None

def display_splash_screen(image_path):
    """
    スプラッシュスクリーンの表示
    """
    global _root
        
    # Tkinterでフレームなしウィンドウを作成
    _root = tk.Tk()
    _root.overrideredirect(True)  # フレームを非表示

    # 画像をTkinter形式に変換
    img_tk = tk.PhotoImage(file=image_path).subsample(2)
    
    # 画面中央に配置
    width = img_tk.width()
    height = img_tk.height()
    x = (_root.winfo_screenwidth() // 2) - (width // 2)
    y = (_root.winfo_screenheight() // 2) - (height // 2)
    _root.geometry(f'{width}x{height}+{x}+{y}')

    _root.update_idletasks()   

    # ラベルに画像を設定
    label = tk.Label(_root, image=img_tk)
    label.pack()

    # 最前面に表示
    _root.attributes('-topmost', True)
        
    # メインループ
    _root.update()
    _root.mainloop(1)

def close_splash_screen():
    global tk

    if _root is not None:
        _root.destroy()

    del sys.modules['tkinter']
    del tk

# 使用例
if __name__ == "__main__":
    # 画像パスを指定してください
    image_path = "escargot.jpg"  # ここに画像パスを入力
        
    display_splash_screen(image_path)

    for i in range(4):
        print("loop")
        time.sleep(1)

    close_splash_screen()
