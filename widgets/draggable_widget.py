
import os
import logging
from kivy.app import App as KVApp
from kivy.uix.widget import Widget as KVWidget
from kivy.core.window import Window as KVWindow
from PIL import Image as PILImage
import io

from AppKit import (
    NSApp, NSDragOperationCopy,
    NSImage, NSURL, NSDraggingItem,
    NSBezierPath, NSColor, NSMakeRect,
    NSEvent, NSLog, NSBitmapImageRep, NSDragOperationNone,
    NSDragOperationDelete, NSWorkspace, NSObject,
    NSPasteboardItem, NSPasteboardTypeFileURL, NSURLPboardType,
    NSFilenamesPboardType,
)
from objc import objc_method

NSPasteboardTypeDrag = 'NSDragPboard'

def ndarray_to_nsimage(arr):
    """NumPy配列をNSImageに変換"""
    pil_img = PILImage.fromarray(arr)
    png_data = io.BytesIO()
    pil_img.save(png_data, format='PNG')
    ns_image = NSImage.alloc().initWithData_(png_data.getvalue())
    rep = NSBitmapImageRep.alloc().initWithData_(png_data.getvalue())
    ns_image.addRepresentation_(rep)
    return ns_image


class _DraggingSource(NSObject):
    @objc_method
    def draggingSession_sourceOperationMaskForDraggingContext_(self, session, context):
        NSLog("ドラッグ操作検出")
        return NSDragOperationCopy | NSDragOperationDelete

    @objc_method
    def draggingSession_endedAtPoint_operation_(self, session, end_point, operation):
        owner = getattr(self, "owner", None)
        if owner is not None:
            owner._on_dragging_session_ended(operation)


def _make_file_pasteboard_item(file_path):
    path = os.path.abspath(file_path)
    url = NSURL.fileURLWithPath_(path)
    url_string = str(url.absoluteString())
    item = NSPasteboardItem.alloc().init()
    item.setString_forType_(url_string, NSPasteboardTypeFileURL)
    item.setString_forType_(url_string, NSURLPboardType)
    item.setPropertyList_forType_([path], NSFilenamesPboardType)
    return item


class DraggableWidget(KVWidget):
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
        self._current_drag_file_paths = []
        self._drag_source = None
    
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
        
        files = self.get_drag_files()
        if len(files) <= 0:
            return
        self._current_drag_file_paths = [file_path for file_path, _image in files]

        dragging_items = []
        for i, file in enumerate(files):
            file_path, image = file

            # Finder 互換寄りの複数 pasteboard type を載せる。
            # Chromium/Arc の Web drop target は NSURL だけだと一度受けてから
            # キャンセルすることがあるため、legacy filenames も明示する。
            dragging_item = NSDraggingItem.alloc().initWithPasteboardWriter_(
                _make_file_pasteboard_item(file_path)
            )
            
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
        self._drag_source = _DraggingSource.alloc().init()
        self._drag_source.owner = self
        
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
            self._drag_source
        )
        NSLog(f"ドラッグセッション開始: {session}")

    def convert_kivy_to_macos_pos(self, pos, offset):
        # Kivyの座標（左下原点）→ macOS座標（左上原点）
        return (pos[0] + offset, KVWindow.height - pos[1])

    def _on_dragging_session_ended(self, operation):
        if operation == NSDragOperationNone:
            logging.debug("ドロップ失敗")
        elif self._drag_operation_requests_delete(operation):
            self._recycle_drag_files(self._current_drag_file_paths)
        else:
            logging.debug("ドロップ成功")
        self._current_drag_file_paths = []
        self._drag_source = None

    def _drag_operation_requests_delete(self, operation):
        return bool(int(operation) & int(NSDragOperationDelete))

    def _recycle_drag_files(self, file_paths):
        paths = [
            os.path.abspath(path)
            for path in file_paths
            if path and os.path.isfile(path)
        ]
        if not paths:
            logging.debug("trash drop ignored: no existing files")
            return

        urls = [NSURL.fileURLWithPath_(path) for path in paths]

        def _completion(_new_urls, error):
            if error is not None:
                logging.warning("failed to move dragged files to Trash: %s", error)
            else:
                logging.info("moved dragged files to Trash: %s", paths)

        NSWorkspace.sharedWorkspace().recycleURLs_completionHandler_(urls, _completion)


class DragDropApp(KVApp):
    def build(self):
        KVWindow.size = (400, 400)
        return DraggableWidget()

if __name__ == '__main__':
    DragDropApp().run()
