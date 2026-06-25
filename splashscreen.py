"""起動スプラッシュスクリーン（macOS / PyObjC・Cocoa 実装）。

旧実装は tkinter を使っていたが、PyInstaller + Kivy 既定フックは tkinter を
除外するため .app では import 段階で失敗して表示されなかった。処理中 HUD
（macos.MacOSProcessingOverlay）と同じく Cocoa(AppKit) で実装し、ソース実行・
.app の双方で表示できるようにする。PyObjC が無い環境では ImportError を送出し、
呼び出し側（main._display_startup_splash）の try/except で無表示にフォールバックする。
"""

import os

_win = None


def display_splash_screen(image_path):
    """borderless な最前面ウィンドウに画像を中央表示する。"""
    global _win

    import AppKit
    from AppKit import NSApplication, NSColor, NSImage, NSImageView, NSWindow, NSMakeRect
    from Foundation import NSDate, NSRunLoop

    abs_path = os.path.abspath(image_path)
    image = NSImage.alloc().initWithContentsOfFile_(abs_path)
    if image is None:
        raise FileNotFoundError(f"splash image not found: {abs_path}")

    # 旧 tkinter 実装の subsample(2) 相当（半分のサイズで表示）。
    size = image.size()
    w = max(1.0, float(size.width) / 2.0)
    h = max(1.0, float(size.height) / 2.0)

    # NSApplication を用意（policy が Prohibited だとウィンドウが出ないため Accessory に）。
    app = NSApplication.sharedApplication()
    try:
        if app.activationPolicy() == 2:  # NSApplicationActivationPolicyProhibited
            app.setActivationPolicy_(getattr(AppKit, "NSApplicationActivationPolicyAccessory", 1))
    except Exception:
        pass

    style = getattr(AppKit, "NSWindowStyleMaskBorderless", 0)
    backing = getattr(AppKit, "NSBackingStoreBuffered", 2)
    _win = NSWindow.alloc().initWithContentRect_styleMask_backing_defer_(
        NSMakeRect(0.0, 0.0, w, h), style, backing, False
    )
    _win.setOpaque_(False)
    _win.setBackgroundColor_(NSColor.clearColor())
    _win.setLevel_(getattr(AppKit, "NSStatusWindowLevel", 25))
    _win.setIgnoresMouseEvents_(True)
    _win.setReleasedWhenClosed_(False)
    cb = getattr(AppKit, "NSWindowCollectionBehaviorTransient", 0)
    if cb:
        try:
            _win.setCollectionBehavior_(cb)
        except Exception:
            pass

    view = NSImageView.alloc().initWithFrame_(NSMakeRect(0.0, 0.0, w, h))
    view.setImage_(image)
    view.setImageScaling_(getattr(AppKit, "NSImageScaleProportionallyUpOrDown", 3))
    view.setImageFrameStyle_(getattr(AppKit, "NSImageFrameNone", 0))
    _win.setContentView_(view)

    _win.center()
    _win.orderFrontRegardless()

    # Kivy のイベントループ開始前なので、一度ランループを回して確実に描画させる。
    try:
        NSRunLoop.currentRunLoop().runUntilDate_(NSDate.dateWithTimeIntervalSinceNow_(0.05))
    except Exception:
        pass


def close_splash_screen():
    """スプラッシュウィンドウを閉じる。"""
    global _win
    if _win is not None:
        try:
            _win.orderOut_(None)
            _win.close()
        except Exception:
            pass
        _win = None


# 使用例
if __name__ == "__main__":
    import time

    display_splash_screen("assets/Shade Wave.png")
    for i in range(4):
        print("loop")
        time.sleep(1)
    close_splash_screen()
