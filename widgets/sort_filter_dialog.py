"""
Viewer のソート/フィルタ設定ダイアログ。
Select... ボタン横の小ボタンから開き、変更は即座に Viewer へ反映される。
"""
from kivy.uix.modalview import ModalView as KVModalView
from kivy.properties import ObjectProperty as KVObjectProperty

import utils.dialogutils as dialogutils
from utils import viewer_query


class SortFilterDialog(KVModalView):
    viewer = KVObjectProperty(None, allownone=True)

    _SORT_LABEL_TO_KEY = {
        'Filename': 'filename',
        'Date': 'date',
        'Rating': 'rating',
        'Edited': 'edited',
    }
    _RATING_LABEL_TO_MIN = {'All': 0, '1+': 1, '2+': 2, '3+': 3, '4+': 4, '5': 5}
    _EDITED_LABEL_TO_VALUE = {'All': 'all', 'Edited': 'edited', 'Unedited': 'unedited'}
    _TYPE_LABEL_TO_VALUE = {'All': 'all', 'RAW': 'raw', 'RGB': 'rgb'}

    def __init__(self, viewer=None, **kwargs):
        super().__init__(**kwargs)
        self.viewer = viewer

    def on_kv_post(self, base_widget):
        dialogutils.install_ref_scaling(self)

    def on_pre_open(self):
        self._sync_widgets_from_settings()

    @staticmethod
    def _label_for(mapping, value, default):
        for label, mapped in mapping.items():
            if mapped == value:
                return label
        return default

    def _sync_widgets_from_settings(self):
        settings = (
            self.viewer.view_settings
            if self.viewer is not None
            else dict(viewer_query.DEFAULT_SETTINGS)
        )
        self.ids['sort_key_spinner'].text = self._label_for(
            self._SORT_LABEL_TO_KEY, settings.get('sort_key'), 'Filename')
        self.ids['sort_order_spinner'].text = (
            'Descending' if settings.get('sort_descending') else 'Ascending')
        self.ids['rating_spinner'].text = self._label_for(
            self._RATING_LABEL_TO_MIN, int(settings.get('filter_rating_min', 0) or 0), 'All')
        self.ids['edited_spinner'].text = self._label_for(
            self._EDITED_LABEL_TO_VALUE, settings.get('filter_edited'), 'All')
        self.ids['type_spinner'].text = self._label_for(
            self._TYPE_LABEL_TO_VALUE, settings.get('filter_type'), 'All')
        self.ids['filter_text_input'].text = settings.get('filter_text', '') or ''

    def _apply(self, **changes):
        if self.viewer is not None:
            self.viewer.set_view_settings(**changes)

    def on_sort_key_label(self, label):
        self._apply(sort_key=self._SORT_LABEL_TO_KEY.get(label, 'filename'))

    def on_sort_order_label(self, label):
        self._apply(sort_descending=(label == 'Descending'))

    def on_rating_label(self, label):
        self._apply(filter_rating_min=self._RATING_LABEL_TO_MIN.get(label, 0))

    def on_edited_label(self, label):
        self._apply(filter_edited=self._EDITED_LABEL_TO_VALUE.get(label, 'all'))

    def on_type_label(self, label):
        self._apply(filter_type=self._TYPE_LABEL_TO_VALUE.get(label, 'all'))

    def on_filter_text(self, text):
        self._apply(filter_text=(text or '').strip())

    def reset(self):
        if self.viewer is not None:
            self.viewer.set_view_settings(**viewer_query.DEFAULT_SETTINGS)
        self._sync_widgets_from_settings()

    def close(self):
        self.dismiss()
