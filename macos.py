
import AppKit
import fcntl
from AppKit import NSScreen, NSApplication, NSEvent, NSBundle
from Quartz import CGDisplayScreenSize, CGDisplayPixelsWide, CGDisplayPixelsHigh

import define

class FileChooser:
    '''A native implementation of file chooser dialogs using Apple's API
    through pyobjus.

    Not implemented features:
    * filters (partial, wildcards are converted to extensions if possible.
        Pass the Mac-specific "use_extensions" if you can provide
        Mac OS X-compatible to avoid automatic conversion)
    * multiple (only for save dialog. Available in open dialog)
    * icon
    * preview
    '''

    mode = "open"
    path = None
    multiple = False
    filters = []
    preview = False
    title = None
    icon = None
    show_hidden = False
    use_extensions = False

    def __init__(self, *args, **kwargs):
        self._handle_selection = kwargs.pop(
            'on_selection', self._handle_selection
        )

        # Simulate Kivy's behavior
        for i in kwargs:
            setattr(self, i, kwargs[i])

    @staticmethod
    def _handle_selection(selection):
        '''
        Dummy placeholder for returning selection from chooser.
        '''
        return selection

    def run(self):
        panel = None
        if self.mode in ("open", "dir", "dir_and_files"):
            panel = AppKit.NSOpenPanel.openPanel()

            panel.setCanChooseDirectories_(self.mode != "open")
            panel.setCanChooseFiles_(self.mode != "dir")

            if self.multiple:
                panel.setAllowsMultipleSelection_(True)
        elif self.mode == "save":
            panel = AppKit.NSSavePanel.savePanel()
        else:
            assert False, self.mode

        panel.setCanCreateDirectories_(True)
        panel.setShowsHiddenFiles_(self.show_hidden)

        if self.title:
            panel.setTitle_(AppKit.NSString.alloc().initWithString_(self.title))

        # Mac OS X does not support wildcards unlike the other platforms.
        # This tries to convert wildcards to "extensions" when possible,
        # ans sets the panel to also allow other file types, just to be safe.
        if self.filters:
            filthies = []
            for f in self.filters:
                if isinstance(f, str):
                    f = (None, f)
                for s in f[1:]:
                    if not self.use_extensions:
                        if s.strip().endswith("*"):
                            continue
                    pystr = s.strip().split("*")[-1].split(".")[-1]
                    filthies.append(AppKit.NSString.alloc().initWithString_(pystr))

            ftypes_arr = AppKit.NSArray.alloc().initWithArray_(filthies)
            # todo: switch to allowedContentTypes
            panel.setAllowedFileTypes_(ftypes_arr)
            panel.setAllowsOtherFileTypes_(not self.use_extensions)

        if self.path:
            url = AppKit.NSURL.fileURLWithPath_(self.path)
            panel.setDirectoryURL_(url)

        selection = None

        if panel.runModal():
            if self.mode == "save" or not self.multiple:
                selection = [panel.filename().UTF8String()]
            else:
                filename = panel.filenames()
                selection = [
                    filename.objectAtIndex_(x).UTF8String()
                    for x in range(filename.count())]

        self._handle_selection(selection)

        return selection

def fadvice(file_path, use_cache=True):
    with open(file_path, "rb") as fd:
        # キャッシュの有効/無効を設定
        fcntl.fcntl(fd, fcntl.F_NOCACHE, 0 if use_cache else 1)
        
        # シーケンシャルアクセスの場合、先読みを有効化
        fcntl.fcntl(fd, 45, 1 if use_cache else 0)  # F_RDAHEAD の値は macOS でのみ有効


def get_screens_info():
    """
    NSScreenを使用して全ディスプレイ情報を取得
    スクリーンID、解像度（ポイント/ピクセル）、表示位置、スケールファクタ、物理サイズ(mm)を含む
    """
    screens = NSScreen.screens()
    results = []
    
    for i, screen in enumerate(screens):
        # スクリーンID
        screen_number = screen.deviceDescription().get('NSScreenNumber', 0)
        display_id = int(screen_number)
        
        # 表示位置とサイズ（ポイント）
        frame = screen.frame()
        
        # 実際のピクセル解像度
        backing_frame = screen.convertRectToBacking_(frame)
        
        # スケールファクタ
        scale_factor = screen.backingScaleFactor()
        
        # 物理サイズ（mm）をQuartzから取得
        try:
            screen_size = CGDisplayScreenSize(display_id)
            width_mm = screen_size.width
            height_mm = screen_size.height
        except:
            # 取得できない場合は0
            width_mm = 0
            height_mm = 0
        
        # 物理ピクセル解像度をQuartzから取得（より正確）
        try:
            phys_pixels_wide = CGDisplayPixelsWide(display_id)
            phys_pixels_high = CGDisplayPixelsHigh(display_id)
        except:
            # 取得できない場合はバッキングフレームから計算
            phys_pixels_wide = backing_frame.size.width
            phys_pixels_high = backing_frame.size.height
        
        results.append({
            # 基本情報
            'id': display_id,
            'index': i,
            'is_primary': (i == 0),
            
            # 表示位置
            'x': int(frame.origin.x),
            'y': int(frame.origin.y),
                        
            # 物理解像度（ピクセル）
            'width_pixels': int(phys_pixels_wide),
            'height_pixels': int(phys_pixels_high),
            
            # 表示解像度（ポイント）
            'width_points': int(frame.size.width),
            'height_points': int(frame.size.height),

            # スケールファクタ
            'scale': float(scale_factor),
            
            # 物理サイズ（mm）
            'width_mm': float(width_mm),
            'height_mm': float(height_mm)
        })
    
    return results

def print_screens_info(screens):
    """ディスプレイ情報を見やすく表示"""
    print("=" * 100)
    print(f"{'No.':<3} {'ID':<10} {'位置':<10} {'解像度（ポイント）':<9} {'解像度（ピクセル）':<9} {'Scale':<7} {'物理サイズ(mm)':<16}")
    print("-" * 100)
    
    for screen in screens:
        pos = f"({screen['x']},{screen['y']})"
        points_res = f"{screen['width_points']}×{screen['height_points']}"
        pixels_res = f"{screen['width_pixels']}×{screen['height_pixels']}"
        scale = f"{screen['scale']}x"
        phys_size = f"{screen['width_mm']:.0f}×{screen['height_mm']:.0f}"
        
        print(f"{screen['index']:<3} "
              f"{screen['id']:<10} "
              f"{pos:<12} "
              f"{points_res:<18} "
              f"{pixels_res:<18} "
              f"{scale:<7} "
              f"{phys_size:<16}")

def calculate_ppi(width_px, height_px, width_mm, height_mm):
    """PPIを計算"""
    import math
    
    # 対角線のピクセル数
    diag_px = math.sqrt(width_px**2 + height_px**2)
    
    # 対角線のインチ数（mmをインチに変換: 1インチ = 25.4mm）
    diag_inch = math.sqrt((width_mm/25.4)**2 + (height_mm/25.4)**2)
    
    # PPI (対角線ベース)
    return diag_px / diag_inch if diag_inch > 0 else 0

_screens = get_screens_info()

def get_self_window_position(app_name=None):
    """
    最もシンプルなバージョン - 座標のみ
    """
    if app_name:
        window = get_window(app_name)
    else:
        window = app.mainWindow()
    
    if window:
        frame = window.frame()
        screen_index = NSScreen.screens().index(window.screen())
        return (frame.origin.x, frame.origin.y, frame.size.width, frame.size.height, screen_index)

    # ウィンドウが見つからない場合はマウスカーソルの位置を返す
    mouse_location = NSEvent.mouseLocation()
    return (mouse_location.x, mouse_location.y, 0, 0, None)

def dpi_scale():
    display = get_current_display()
    return _screens[display['display']]['scale']

def get_current_display(win_x=0, win_y=0, win_display=None):
    global _screens

    # 現在のウィンドウの左下座標
    if win_x == 0 and win_y == 0:
        win_x, win_y, _, _, win_display = get_self_window_position(define.APPNAME)
    
    # 所属しているディスプレイが取得できている場合は、それを返す
    if win_display is not None:
        return {"display": win_display, "width": _screens[win_display]['width_points'], "height": _screens[win_display]['height_points'], "is_primary": _screens[win_display]['is_primary']}

    for m in _screens:
        if m['is_primary'] == True:
            primary = m
            break

    for i, m in enumerate(_screens):
        # Yの反転
        my = 0
        if m['y'] != 0:
            my = primary['height_points'] if m['y'] > 0 else -m['height_points']

        if m['x'] <= win_x < m['x'] + m['width_points'] and my <= win_y < my + m['height_points']:
            return {"display": i, "width": m['width_points'], "height": m['height_points'], "is_primary": m['is_primary']}
    
    return {"display": primary['index'], "width": primary['width_points'], "height": primary['height_points'], "is_primary": primary['is_primary']}

def get_window(app_name):
    app = NSApplication.sharedApplication()

    # mainWindow() ではなく、windows() リストから探す
    windows = app.windows()

    if windows and windows.count() > 0:
        # より確実に特定したい場合（例：特定のタイトルを持つウィンドウ）
        for i in range(windows.count()):
            win = windows.objectAtIndex_(i)
            if win.title() == app_name:
                return win
    
    return None

def set_window_autosave(app_name, window_name):

    target_window = get_window(app_name)
    if target_window:
        target_window.setFrameAutosaveName_(window_name)
        return True

    return False

# 使用例
if __name__ == "__main__":
    print("全ディスプレイ情報（詳細版）:")
    print_screens_info(_screens)
