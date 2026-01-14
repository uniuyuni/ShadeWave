
import uuid
from typing import List, Dict, Any
import numpy as np
import logging
from enum import Enum

from kivymd.app import MDApp

class LayerCtrl:
    def update_layer(self, op, type, index, op_type, param):
        pass

    def get_layer(self, index):
        pass

class Operation:

    def __init__(self, lv=0, effect=None, subname=None, mask_id=None, type="Effect"):
        self.id = str(uuid.uuid4())
        self.type = type
        self.name = ""
        self.lv = lv
        self.effect = effect
        self.effects = None
        self.effects_param = None
        self.subname = subname
        self.mask_id = mask_id
        self.update = {}    # 更新パラメータ
        self.backup = {}    # もとに戻す時のパラメータ
        self.diff = []      # 差分
    
    def set_backup_layer(self, layer_ctrl, op, index, op_type):
        self.layer_ctrl = layer_ctrl
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
        self.name = effects[self.lv][self.effect].__class__.__name__
        self.effects = effects
        self.effects_param = param

        # パラメータ辞書を取得
        if subname is not None:
            ef_dict = effects[self.lv][self.effect].get_param_dict(param, subname)
        else:
            ef_dict = effects[self.lv][self.effect].get_param_dict(param)

        # バックアップを作成
        for key in ef_dict.keys():
            val = param.get(key, ef_dict[key])
            if isinstance(val, list):
                self.backup[key] = val.copy()
            else:
                self.backup[key] = val
        
        return (self.lv, self.effect)
    
    def set_update(self, effects, param, subname=None):
        if self.effect is None:
            logging.warning("Operation.set_update effect is None.")
            return None
            
        if self.effects_param is not param:
            logging.warning("Operation.set_update param is not match.")
            return None

        # パラメータ辞書を取得
        if subname is not None:
            ef_dict = effects[self.lv][self.effect].get_param_dict(param, subname)
        else:
            ef_dict = effects[self.lv][self.effect].get_param_dict(param)
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
            [key, self.backup[key], self.update[key]]
            for key in self.backup.keys() & self.update.keys()
            if self.backup[key] != self.update[key]
            #if self.backup[key] is not self.update[key]
        ]
        if len(self.diff) == 0:
            return None
        
        return (self.lv, self.effect)

    def undo(self, widget):
        if self.type == "Effect":
            self.effects_param.update(self.backup)
            self.effects[self.lv][self.effect].set2widget(widget, self.effects_param)

        elif self.type == "Layer":
            self.layer_ctrl.update_layer(self.backup['op'], self.backup['index'], self.backup['op_type'], self.backup['dict'])

    def redo(self, widget):
        if self.type == "Effect":
            self.effects_param.update(self.update)
            self.effects[self.lv][self.effect].set2widget(widget, self.effects_param)

        elif self.type == "Layer":
            self.layer_ctrl.update_layer(self.update['op'], self.update['index'], self.update['op_type'], self.update['dict'])

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
    return MDApp.get_running_app().main_widget
