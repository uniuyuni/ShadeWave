
import dashscope
from dashscope import MultiModalConversation
import base64
from pathlib import Path
import json
import os
import requests

# DashScope APIキーを設定
dashscope.api_key = os.environ.get("DASHSCOPE_API_KEY")
dashscope.base_http_api_url = 'https://dashscope-intl.aliyuncs.com/api/v1'

def encode_image_to_base64(image_path):
    """画像ファイルをBase64エンコード"""
    with open(image_path, 'rb') as f:
        return base64.b64encode(f.read()).decode('utf-8')

def save_base64_image(base64_str, output_path):
    """Base64文字列を画像ファイルとして保存"""
    image_data = base64.b64decode(base64_str)
    with open(output_path, 'wb') as f:
        f.write(image_data)
    print(f"画像を保存しました: {output_path}")

def download_image_from_url(url, output_path):
    """URLから画像をダウンロードして保存"""
    try:
        print(f"画像をダウンロード中: {url}")
        response = requests.get(url, timeout=30)
        response.raise_for_status()
        
        with open(output_path, 'wb') as f:
            f.write(response.content)
        print(f"画像を保存しました: {output_path}")
        return True
    except Exception as e:
        print(f"ダウンロードエラー: {str(e)}")
        return False

def edit_image_with_qwen(input_image_path, prompt, output_path="edited_image.jpg"):
    """
    Qwen Image Editを使って画像を編集
    
    Args:
        input_image_path: 入力画像のパス
        prompt: 編集指示のプロンプト
        output_path: 出力画像のパス（デフォルト: edited_image.jpg）
    """
    try:
        # 画像をBase64エンコード
        print(f"画像を読み込んでいます: {input_image_path}")
        image_base64 = encode_image_to_base64(input_image_path)
        
        # APIリクエストを送信
        print("画像編集を実行中...")
        messages = [
            {
                "role": "user",
                "content": [
                    {
                        "image": f"data:image/jpeg;base64,{image_base64}"
                    },
                    {
                        "text": prompt
                    }
                ]
            }
        ]
        
        response = MultiModalConversation.call(
            model='qwen-image-edit',  # または 'qwen-vl-plus' など
            messages=messages,
            watermark=False,
        )
        
        # レスポンスを確認
        if response.status_code == 200:
            output = response.output
            
            # 編集された画像を取得（レスポンス形式に応じて調整が必要な場合があります）
            if 'choices' in output and len(output['choices']) > 0:
                choice = output['choices'][0]
                if 'message' in choice and 'content' in choice['message']:
                    content = choice['message']['content']
                    
                    # コンテンツから画像データを抽出
                    for item in content:
                        if isinstance(item, dict) and 'image' in item:
                            image_url = item['image']
                            print(f"画像URL: {image_url}")
                            
                            # URLから画像をダウンロード
                            if download_image_from_url(image_url, output_path):
                                return True
                            else:
                                return False

            
            print("レスポンス:", json.dumps(output, indent=2, ensure_ascii=False))
            print("画像データが見つかりませんでした。")
            return False
        else:
            print(f"エラー: {response.code} - {response.message}")
            return False
            
    except Exception as e:
        print(f"エラーが発生しました: {str(e)}")
        return False

# 使用例
if __name__ == "__main__":
    # 設定
    INPUT_IMAGE = "X-T5 Room input 0.jpg"  # 編集したい画像のパス
    PROMPT="""
    Remove ALL red masked pixels completely.
    Fill red areas with perfectly matched background.
    Ensure a natural finish that blends seamlessly with the surrounding area when filling in.
    Keep every non-red pixel EXACTLY as original.
    Zero modifications beyond red mask boundaries.
    This image is a cropped section of a larger image. Other images exist above, below, to the left, and to the right, and will be composited later. Therefore, please refrain from performing any processing that could interfere with the compositing work, such as moving the object's position, resizing it, or altering the overall color tone.
    """
    NEGATIVE_PROMPT="""
    DO NOT TOUCH any non-red areas - not even slightly!
    No color changes, no brightness changes, no texture edits, no noise alteration outside red areas.
    As the sole exception, if the texture boundary is located approximately 192 pixels from the top, bottom, left, or right edge of the image, blend the boundary to make it disappear. Avoid editing areas outside the boundary whenever possible.
    """
    OUTPUT_IMAGE = "X-T5 Room output 0.jpg"  # 出力ファイル名
    
    # 画像編集を実行
    edit_image_with_qwen(INPUT_IMAGE, PROMPT, OUTPUT_IMAGE)
