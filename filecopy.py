import os
import shutil
import subprocess
import json
from datetime import datetime

# JSONファイルの読み込み
with open('file_formats.json', 'r') as f:
    file_formats = json.load(f)

# 拡張子のリストを作成
extensions = set()
for ext_list in file_formats['file_formats'].values():
    extensions.update(ext_list)

def get_original_date(file_path):
    try:
        # ExifToolを使用してメタデータを取得
        result = subprocess.run(['exiftool', '-OriginalDate', file_path],
                                capture_output=True, text=True)
        output = result.stdout.strip()
        if output:
            # 日付部分を抽出
            date_str = output.split(': ')[-1]
            return datetime.strptime(date_str, '%Y:%m:%d %H:%M:%S')
    except Exception as e:
        print(f"Error reading {file_path}: {e}")
    return None

def copy_files_to_date_folders(src_folder, dest_folder):
    os.makedirs(dest_folder, exist_ok=True)

    for root, dirs, files in os.walk(src_folder):
        for file in files:
            if any(file.lower().endswith(ext) for ext in extensions):
                src_file_path = os.path.join(root, file)
                original_date = get_original_date(src_file_path)

                if original_date:
                    date_folder = original_date.strftime('%Y-%m-%d')
                    dest_date_folder = os.path.join(dest_folder, date_folder)

                    # 日付フォルダを作成
                    os.makedirs(dest_date_folder, exist_ok=True)

                    # コピー先のファイルパス
                    dest_file_path = os.path.join(dest_date_folder, file)

                    # ファイルをコピー
                    shutil.copy2(src_file_path, dest_file_path)

                    print(f"Copied {src_file_path} to {dest_file_path}")

if __name__ == "__main__":
    # コピー元とコピー先のフォルダを指定
    source_folder = input("コピー元フォルダのパスを入力してください: ")
    destination_folder = input("コピー先フォルダのパスを入力してください: ")

    copy_files_to_date_folders(source_folder, destination_folder)
    print("ファイルのコピーと振り分けが完了しました。")
