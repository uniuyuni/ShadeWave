
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
            #if hasattr(child, 'ref_size_hint_min') and child.ref_size_hint_min:
            #    child.size_hint_min = (dpi_scale_width(child.ref_size_hint_min[0]), dpi_scale_height(child.ref_size_hint_min[1]))
            #if hasattr(child, 'ref_size_hint_max') and child.ref_size_hint_max:
            #    child.size_hint_max = (dpi_scale_width(child.ref_size_hint_max[0]), dpi_scale_height(child.ref_size_hint_max[1]))
        
        for child in get_entire_widget_tree(root):
            if isinstance(child, MDScrollView):
                child.children[0].height = child.children[0].minimum_height


def dpi_scale_width(ref):
    if isinstance(ref, (list, tuple)):
        return [r * device.dpi_scale() for r in ref]
    return ref * device.dpi_scale()

def dpi_scale_height(ref):
    if isinstance(ref, (list, tuple)):
        return [r * device.dpi_scale() for r in ref]
    return ref * device.dpi_scale()


# main.kv: 中列 0.55, preview 下 ref_height:30, Viewer ref_height:160
_PREVIEW_COL_FRAC = 0.55
_PREVIEW_BAR_REF = 30
_VIEWER_REF = 160


def get_window_screen_size():
    """
    1 台分の表示枠 (w, h) ポイント。NSScreen.frame ベース。Kivy の Window 幅・minimum_* の単位
   （macOS では主に pt）に合わせ、最小窓 cap にそのまま使う。

    preview_min_edge では ref*dpi 系 (m0, bar) と同じ系に揃えるため、内部で sw,sh に
    device.dpi_scale() を掛けて「バッキング相当幅」にして m の上限を計算する（混在を避けるため）。
    """
    if hasattr(device, "get_app_window_screen_size_points"):
        try:
            p = device.get_app_window_screen_size_points()
        except Exception:
            p = None
        if p and p[0] > 0 and p[1] > 0:
            return int(p[0]), int(p[1])
    if hasattr(device, "get_primary_display_size_points"):
        try:
            p = device.get_primary_display_size_points()
        except Exception:
            p = None
        if p and p[0] > 0 and p[1] > 0:
            return int(p[0]), int(p[1])
    return 1920, 1080


def preview_min_edge_for_window(
        preview_size_ref,
        area_frac=0.92,
        preview_col_frac=_PREVIEW_COL_FRAC,
        viewer_ref=_VIEWER_REF):
    """
    m0=ref*dpi_scale に対し、画面 1 台のポイント (sw, sh) を *dpi して m0, bar, view のバッキング
    域で比較し、最小幅窓に収まる最大 m = min(m0, m_wmax, m_hmax)。get_window_screen_size() は
    ポイント (Kivy と minimum_*  cap に整合)。
    """
    m0 = int(dpi_scale_width(preview_size_ref))
    if m0 < 1:
        return 1
    sw_pt, sh_pt = get_window_screen_size()
    if sw_pt < 2 or sh_pt < 2:
        return max(1, m0)
    dps = float(device.dpi_scale())
    sw = float(sw_pt) * dps
    sh = float(sh_pt) * dps
    bar = int(dpi_scale_height(_PREVIEW_BAR_REF))
    view = int(dpi_scale_height(viewer_ref))
    # 最小幅 w: m/preview_col_frac <= sw*area  （sw = 表示ポイント*scale）
    m_w = int(float(preview_col_frac) * sw * area_frac)
    # 最低高さ: m + view + bar <= sh*area
    m_h = int(max(0.0, sh * area_frac - float(view) - float(bar)))
    m_max = min(m_w, m_h)
    if m_max < 1:
        return 1
    return min(m0, m_max)
