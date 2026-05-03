
import uuid
from typing import List, Dict, Any
import numpy as np
import logging

import effects
import params
import utils.utils as utils

class LayerCtrl:
    def update_layer(self, op, type, index, op_type, param):
        pass

    def get_layer(self, index):
        pass

class Operation:

    def __init__(self, lv=0, effect_list=None, subname=None, mask_id=None, type="Effect"):
        self.id = str(uuid.uuid4())
        self.type = type
        self.name = ""
        self.lv = lv
        self.effect_list = effect_list
        self.effects = None
        self.effects_param = None
        self.subname = subname
        self.mask_id = mask_id
        self.update = {}    # 更新パラメータ
        self.backup = {}    # もとに戻す時のパラメータ
        self.diff = []      # 差分
    
    def set_backup_layer(self, layer_ctrl, op, index, op_type):
        self.layer_ctrl = layer_ctrl
        self.name = "Layer"
        self.backup['op'] = op
        self.backup['index'] = index
        self.backup['op_type'] = op_type
        self.backup['dict'] = None if op == "Delete" else layer_ctrl.get_layer(index).serialize()
        
    def set_update_layer(self, layer_ctrl, op, index):
        if self.layer_ctrl is not layer_ctrl:
            logging.error("Operation.set_update_layer layer_ctrl is not match error.")
            return None

        self.update['op'] = op
        self.update['index'] = index
        self.update['op_type'] = self.backup['op_type']
        self.update['dict'] = layer_ctrl.get_layer(index).serialize()

        self.diff.append(["Mask2 " + op, self.update['dict']['name'], str(index)])

        return self.update
    
    def set_backup(self, effects, param, subname=None):
        self.name = "Reset" if self.effect_list[0] is None else effects[self.lv][self.effect_list[0]].__class__.__name__
        self.effects = effects
        self.effects_param = param
        ef_dict = self._get_effect_param_dict(effects, param, subname)

        if ef_dict is None:
            return (self.lv, self.effect_list)

        # バックアップを作成
        for key in ef_dict.keys():
            val = param.get(key, ef_dict[key])
            if isinstance(val, list):
                self.backup[key] = val.copy()
            else:
                self.backup[key] = val
        
        return (self.lv, self.effect_list)

    def _get_effect_param_dict(self, effects, param, subname=None):
        if self.effect_list[0] is None:
            return param

        if subname is not None:
            return effects[self.lv][self.effect_list[0]].get_param_dict(param, subname)

        ef_dict = {}
        for effect in self.effect_list:
            part = effects[self.lv][effect].get_param_dict(param)
            if part:
                ef_dict.update(part)
        return ef_dict
    
    def set_update(self, _effects, param, subname=None):
        if self.effect_list is None:
            logging.warning("Operation.set_update effect_list is None.")
            return None
            
        if self.effects_param is not param:
            logging.warning("Operation.set_update param is not match.")
            return None

        ef_dict = self._get_effect_param_dict(_effects, param, subname)
        if ef_dict is None:
            return None

        # アップデートを作成
        for key in ef_dict.keys():
            val = param.get(key, ef_dict[key])
            if isinstance(val, list):
                self.update[key] = val.copy()
            else:
                self.update[key] = val

        # 差分を作成
        self.diff = [
            [key, self.backup[key], self.update.get(key, "Reset")]
            for key in self.backup.keys() | self.update.keys()
            if (not (self.backup.get(key, effects.get_default_param(self.effects, key, self.effects_param)) == self.update.get(key, effects.get_default_param(self.effects, key, self.effects_param))).all() if isinstance(self.backup.get(key, effects.get_default_param(self.effects, key, self.effects_param)), np.ndarray)
                else self.backup.get(key, effects.get_default_param(self.effects, key, self.effects_param)) != self.update.get(key, effects.get_default_param(self.effects, key, self.effects_param)))
        ]
        if len(self.diff) == 0:
            return None
        
        return (self.lv, self.effect_list)

    def set_backup_all(self, param, mask_editor):
        temp_param = params.serialize(param, mask_editor)
        params.copy_special_param(temp_param['primary_param'], param)
        self.name = "Reset"
        self.backup['dict'] = temp_param

    def check_backup_all(self, param, mask_editor):
        temp_param = params.serialize(param, mask_editor)
        params.copy_special_param(temp_param['primary_param'], param)
        return utils.dict_equal_with_ndarray(self.backup['dict'], temp_param)

    def undo(self, widget):
        if self.type == "Effect":
            self.effects_param.update(self.backup)
            if self.effect_list[0] is None:
                effects.set2widget_all(widget, self.effects, self.effects_param)
            else:
                for effect in self.effect_list:
                    self.effects[self.lv][effect].set2widget(widget, self.effects_param)

        elif self.type == "Layer":
            self.layer_ctrl.update_layer(self.backup['op'], self.backup['index'], self.backup['op_type'], self.backup['dict'])
        
        elif self.type == "All":
            widget.primary_param.clear()
            dict = self.backup['dict']
            if dict is not None:
                params.deserialize(dict, widget.primary_param, widget.ids['mask_editor2'])

    def redo(self, widget):
        if self.type == "Effect":
            diff = {}
            for d in self.diff:
                if d[2] == "Reset":
                    diff[d[0]] = effects.get_default_param(self.effects, d[0], self.effects_param)
                else:
                    diff[d[0]] = d[2]
            self.effects_param.update(diff)
            self.effects_param.update(self.update)
            if self.effect_list[0] is None:
                effects.set2widget_all(widget, self.effects, self.effects_param)
            else:
                for effect in self.effect_list:
                    self.effects[self.lv][effect].set2widget(widget, self.effects_param)

        elif self.type == "Layer":
            self.layer_ctrl.update_layer(self.update['op'], self.update['index'], self.update['op_type'], self.update['dict'])

        elif self.type == "All":
            widget.reset_all()

class History:
    """操作履歴マネージャー"""
    
    def __init__(self, max_history: int = 1000):
        self.operations: List[Operation] = []
        self.current_index: int = -1
        self.max_history = max_history
    
    def append(self, operation: Operation) -> None:
        """新しい操作を実行"""
        result = self.current_index

        # 現在位置より後ろの操作を削除（redoスタックをクリア）
        self.operations = self.operations[:self.current_index + 1]
        
        # 新しい操作を追加
        self.operations.append(operation)
        self.current_index += 1
        
        # 履歴上限チェック
        if len(self.operations) > self.max_history:
            self.operations.pop(0)
            self.current_index -= 1

        return result
    
    def undo(self, widget) -> bool:
        """1つ前の状態に戻す"""
        if self.can_undo():
            self.operations[self.current_index].undo(widget)
            self.current_index -= 1
            return True
        return False
    
    def redo(self, widget) -> bool:
        """1つ先の状態に進む"""
        if self.can_redo():
            self.operations[self.current_index+1].redo(widget)
            self.current_index += 1
            return True
        return False
    
    def can_undo(self) -> bool:
        return self.current_index >= 0
    
    def can_redo(self) -> bool:
        return self.current_index < len(self.operations) - 1
    
    def get_active_operations(self) -> List[Operation]:
        """現在有効な操作リストを取得"""
        return self.operations[:self.current_index + 1]
    
    def get_history_info(self) -> List[Dict[str, Any]]:
        """履歴情報を取得"""
        info = []
        for i, op in enumerate(self.operations):
            info.append({
                "index": i,
                "type": op.type,
                "name": op.name,
                "backup": op.backup,
                "update": op.update,
                "active": i < self.current_index
            })
        return info

def get_history_ctrl():
    from kivymd.app import MDApp
    return MDApp.get_running_app().main_widget
