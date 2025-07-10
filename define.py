
import os
import json

def get_version():
    """
    escargot.code-workspaceファイルからバージョン情報を取得します。
    バージョン情報が見つからない場合は「不明」を返します。
    
    Returns:
        str: バージョン文字列
    """
    try:
        # ワークスペースファイルのパスを取得
        workspace_path = os.path.join(os.getcwd(), "platypus.code-workspace")
        
        # ファイルが存在するか確認
        if not os.path.exists(workspace_path):
            return "不明"
            
        # JSONファイルを読み込む
        with open(workspace_path, 'r', encoding='utf-8') as f:
            workspace_data = json.load(f)
            
        # バージョン情報を探す
        # 通常はsettingsやmetadataなどに格納されている可能性がある
        version = "不明"
        
        # 基本的な場所を確認
        if "version" in workspace_data:
            version = workspace_data["version"]
        elif "settings" in workspace_data and "version" in workspace_data["settings"]:
            version = workspace_data["settings"]["version"]
        elif "metadata" in workspace_data and "version" in workspace_data["metadata"]:
            version = workspace_data["metadata"]["version"]
        elif "launch" in workspace_data and "version" in workspace_data["launch"]:
            version = workspace_data["launch"]["version"]
            
        return version
    
    except Exception as e:
        print(f"バージョン情報の取得中にエラーが発生しました: {e}")
        return "不明"

APPNAME = "Platypus"
VERSION = get_version()

SUPPORTED_FORMATS_RGB = ('.png', '.jpg', '.jpeg', '.tif', '.tiff', '.bmp', '.gif')
SUPPORTED_FORMATS_RAW = ('.cr2', '.cr3', '.nef', '.arw', '.dng', '.orf', '.raf', '.rw2', '.sr2', '.pef', '.raw')
