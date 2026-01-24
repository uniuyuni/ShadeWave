
from kivy.core.window import Window as KVWindow
from kivy.uix.widget import Widget as KVWidget
from kivymd.uix.scrollview import MDScrollView
#from screeninfo import get_monitors
import macos as device

def get_root_widget(widget):
    while widget.parent is not widget:
        widget = widget.parent
    return widget.children[0]

"""
def get_current_display():
    # 現在のウィンドウの左上座標
    win_x, win_y = KVWindow.left, KVWindow.top

    # モニタ一覧を取得して、ウィンドウが属しているモニタを探す
    monitors = get_monitors()

    for i, m in enumerate(monitors):
        if m.is_primary == True:
            primary = m
            break

    for i, m in enumerate(monitors):
        if m.y != 0:
            m.y = -m.height if m.y > 0 else primary.height
        if m.x <= win_x < m.x + m.width and m.y <= win_y < m.y + m.height:
            return {"display": i, "width": m.width, "height": m.height, "is_primary": m.is_primary}
    
    return None
"""

def get_entire_widget_tree(root, delay=0.1):
    """全ウィジェット取得（未表示含む）"""
    results = []
    
    def _collect(w):
        if not isinstance(w, KVWidget):
            return
            
        results.append(w)
        
        # 特殊レイアウト対応
        if hasattr(w, 'tab_list'):  # TabbedPanel
            for tab in w.tab_list:
                _collect(tab.content)
                
        if hasattr(w, 'screens'):  # ScreenManager
            for screen in w.screens:
                _collect(screen)
                
        # 通常の子要素
        for child in w.children:
            _collect(child)
    
    # 遅延実行で未初期化要素に対応
    #KVClock.schedule_once(lambda dt: _collect(root), delay)
    _collect(root)

    return results

def find_widget(p, id):
    # 動的追加Widgetでも探し出す
    for child in get_entire_widget_tree(p):
        if hasattr(child, 'id') and child.id == id:
            return child
    return None

def traverse_widget(root):
    # すべてのスケールが必要なウィジェットを更新
    if root:
        for child in get_entire_widget_tree(root):
            if hasattr(child, 'ref_width') and child.ref_width:
                child.width = dpi_scale_width(child.ref_width)
            if hasattr(child, 'ref_height') and child.ref_height:
                child.height = dpi_scale_height(child.ref_height)
            if hasattr(child, 'ref_padding') and child.ref_padding:
                child.padding = dpi_scale_width(child.ref_padding)
            if hasattr(child, 'ref_spacing') and child.ref_spacing:
                child.spacing = dpi_scale_width(child.ref_spacing)
            if hasattr(child, 'ref_tab_width') and child.ref_tab_width:
                child.tab_width = dpi_scale_width(child.ref_tab_width)
            if hasattr(child, 'ref_tab_height') and child.ref_tab_height:
                child.tab_height = dpi_scale_height(child.ref_tab_height)
        
        for child in get_entire_widget_tree(root):
            if isinstance(child, MDScrollView):
                child.children[0].height = child.children[0].minimum_height


def dpi_scale_width(ref):
    return ref * device.dpi_scale()

def dpi_scale_height(ref):
    return ref * device.dpi_scale()