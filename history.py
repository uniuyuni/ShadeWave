
import uuid
from typing import List, Dict, Any
import numpy as np
import logging

class Operation:

    def __init__(self, lv, effect):
        self.id = str(uuid.uuid4())
        self.type = 0       # type effect
        self.name = ""
        self.lv = lv
        self.effect = effect
        self.effects = None
        self.effects_param = None
        self.update = {}    # 更新パラメータ
        self.backup = {}    # もとに戻す時のパラメータ
    
    def set_backup(self, effects, param):
        self.name = effects[self.lv][self.effect].__class__.__name__
        self.effects = effects
        self.effects_param = param

        # パラメータ辞書を取得
        ef_dict = effects[self.lv][self.effect].get_param_dict(param)

        # バックアップを作成
        for key in ef_dict.keys():
            self.backup[key] = param.get(key, ef_dict[key])
        
        return (self.lv, self.effect)
    
    def set_update(self, effects, param):
        if self.effects_param is not param:
            logging.error("Operation.set_update param is not match error.")

        # パラメータ辞書を取得
        ef_dict = effects[self.lv][self.effect].get_param_dict(param)

        # アップデートを作成
        for key in ef_dict.keys():
            self.update[key] = param.get(key, ef_dict[key])
        
        return (self.lv, self.effect)

    def undo(self, widget):
        self.effects_param.update(self.backup)
        self.effects[self.lv][self.effect].set2widget(widget, self.effects_param)

    def redo(self, widget):
        self.effects_param.update(self.update)
        self.effects[self.lv][self.effect].set2widget(widget, self.effects_param)

class History:
    """操作履歴マネージャー"""
    
    def __init__(self, max_history: int = 1000):
        self.operations: List[Operation] = []
        self.current_index: int = -1
        self.max_history = max_history
    
    def append(self, operation: Operation) -> None:
        """新しい操作を実行"""
        # 現在位置より後ろの操作を削除（redoスタックをクリア）
        self.operations = self.operations[:self.current_index + 1]
        
        # 新しい操作を追加
        self.operations.append(operation)
        self.current_index += 1
        
        # 履歴上限チェック
        if len(self.operations) > self.max_history:
            self.operations.pop(0)
            self.current_index -= 1
    
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
            self.operations[self.current_index].redo(widget)
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
