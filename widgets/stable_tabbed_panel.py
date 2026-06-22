from kivy.clock import Clock
from kivy.properties import NumericProperty, StringProperty
from kivy.uix.tabbedpanel import TabbedPanel, TabbedPanelHeader, TabbedPanelItem


class StableTabbedPanel(TabbedPanel):
    """TabbedPanel that keeps header widgets matched to tab_height."""

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.bind(tab_height=self._schedule_sync_tabs, tab_width=self._schedule_sync_tabs)
        Clock.schedule_once(self._sync_tabs, 0)

    def add_widget(self, widget, *args, **kwargs):
        result = super().add_widget(widget, *args, **kwargs)
        if isinstance(widget, TabbedPanelHeader):
            self._schedule_sync_tabs()
        return result

    def on_tab_height(self, *args):
        super().on_tab_height(*args)
        self._schedule_sync_tabs()

    def on_tab_width(self, *args):
        super().on_tab_width(*args)
        self._schedule_sync_tabs()

    def _schedule_sync_tabs(self, *_args):
        Clock.schedule_once(self._sync_tabs, 0)

    def _sync_tabs(self, *_args):
        tab_height = max(1, float(self.tab_height or 1))
        tab_width = float(self.tab_width or 0)

        strip = getattr(self, "_tab_strip", None)
        if strip is not None:
            strip.size_hint_y = None
            strip.height = tab_height
            strip.row_force_default = True
            strip.row_default_height = tab_height
            if tab_width > 0:
                strip.col_force_default = True
                strip.col_default_width = tab_width
            else:
                strip.col_force_default = False

        layout = getattr(self, "_tab_layout", None)
        if layout is not None:
            layout.height = tab_height + layout.padding[1] + layout.padding[3] + 2

        for tab in self.tab_list:
            tab.size_hint_y = None
            tab.height = tab_height
            if tab_width > 0:
                tab.width = tab_width

        self._reposition_tabs()


class IconTabbedPanelItem(TabbedPanelItem):
    icon_source = StringProperty("")
    icon_scale = NumericProperty(0.7)
    font_scale = NumericProperty(0.45)


class TextTabbedPanelItem(TabbedPanelItem):
    font_scale = NumericProperty(0.62)
