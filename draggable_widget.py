
import os
from kivy.app import App
from kivy.uix.widget import Widget
from kivy.core.window import Window
from PIL import Image as PILImage
import io

from AppKit import (
    NSApp, NSDragOperationCopy,
    NSImage, NSURL, NSDraggingItem,
    NSBezierPath, NSColor, NSMakeRect,
    NSEvent, NSLog, NSBitmapImageRep, NSDragOperationNone
)
from objc import objc_method, informal_protocol, selector

NSPasteboardTypeDrag = 'NSDragPboard'

informal_protocol('NSDraggingSource', [
    selector(None, b'draggingSession:sourceOperationMaskForDraggingContext:',
                  signature=b'v@:@@'),
    selector(None, b'draggingSession:endedAtPoint:operation:',
                  signature=b'v@:@@L')
])

def ndarray_to_nsimage(arr):
    """NumPy配列をNSImageに変換"""
    pil_img = PILImage.fromarray(arr)
    png_data = io.BytesIO()
    pil_img.save(png_data, format='PNG')
    ns_image = NSImage.alloc().initWithData_(png_data.getvalue())
    rep = NSBitmapImageRep.alloc().initWithData_(png_data.getvalue())
    ns_image.addRepresentation_(rep)
    return ns_image

class DraggableWidget(Widget):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        """
        self.file_path = os.path.join(os.getcwd(), 'your_image.jpg')
        self.size = (100, 100)
        self.pos = (Window.width/2 - 50, Window.height/2 - 50)
        
        with self.canvas:
            Color(0.2, 0.6, 1, 1)
            self.rect = Rectangle(pos=self.pos, size=self.size)
        """
        self.dragging = False
    
    def on_touch_move(self, touch):
        if self.collide_point(*touch.pos) and self.dragging == False:
            self.dragging = True
            self.start_drag(touch)
            return True
        
        return super().on_touch_move(touch)
    
    def on_touch_up(self, touch):
        self.dragging = False
        return super().on_touch_up(touch)
    
    # テスト用（残すこと）
    def get_drag_files(self):
        file_paths = []
        file_paths.append((os.path.join(os.getcwd(), 'your_image.jpg'), None))
        file_paths.append((os.path.join(os.getcwd(), 'escargot.jpg'), None))
        return file_paths

    def start_drag(self, touch):
        
        file_paths = self.get_drag_files()
        if len(file_paths) <= 0:
            return

        dragging_items = []
        for i, file in enumerate(file_paths):
            file_path, image = file

            # ファイルURLの準備
            file_url = NSURL.fileURLWithPath_(file_path)
            
            # ドラッグアイテムの作成
            dragging_item = NSDraggingItem.alloc().initWithPasteboardWriter_(file_url)
            
            # ドラッグ画像の設定（サイズ: 64x64）
            if image is None:
                drag_image = NSImage.alloc().initWithSize_((64, 64))
                drag_image.lockFocus()
                NSColor.systemBlueColor().set()
                path = NSBezierPath.bezierPathWithRoundedRect_xRadius_yRadius_(
                    NSMakeRect(0, 0, 64, 64), 10, 10)
                path.fill()
                drag_image.unlockFocus()
            else:
                drag_image = ndarray_to_nsimage(image)
            
            # 座標変換（Kivy → Cocoa）
            mouse_pos = self.convert_kivy_to_macos_pos(touch.pos, i*80)
            
            # ドラッグフレームの設定
            dragging_item.setDraggingFrame_contents_(
                (mouse_pos, (64, 64)),
                drag_image
            )

            dragging_items.append(dragging_item)
        
        # メインウィンドウの取得
        main_window = NSApp().keyWindow()
        if not main_window:
            NSLog("メインウィンドウが見つかりません")
            return
        
        # ドラッグセッションの開始
        session = main_window.contentView().beginDraggingSessionWithItems_event_source_(
            dragging_items,
            NSEvent.mouseEventWithType_location_modifierFlags_timestamp_windowNumber_context_eventNumber_clickCount_pressure_(
                6,  # NSLeftMouseDragged
                mouse_pos,
                0,  # modifierFlags
                0,  # timestamp
                main_window.windowNumber(),
                None,  # context
                0,  # eventNumber
                1,  # clickCount
                0.0  # pressure
            ),
            main_window
        )
        NSLog(f"ドラッグセッション開始: {session}")

    def convert_kivy_to_macos_pos(self, pos, offset):
        # Kivyの座標（左下原点）→ macOS座標（左上原点）
        return (pos[0] + offset, Window.height - pos[1])

    @objc_method
    def draggingSession_sourceOperationMaskForDraggingContext_(self, session, context):
        NSLog("ドラッグ操作検出")
        return NSDragOperationCopy

    @objc_method
    def draggingSession_endedAtPoint_operation_(self, session, end_point, operation):    
        if operation == NSDragOperationNone:
            print("ドロップ失敗")
        else:
            print("ドロップ成功")


class DragDropApp(App):
    def build(self):
        Window.size = (400, 400)
        return DraggableWidget()

if __name__ == '__main__':
    DragDropApp().run()