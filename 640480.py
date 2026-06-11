import cv2
import numpy as np

# ================= 參數設定區 =================
# 原始圖片路徑 (使用 r 前綴)
INPUT_PATH = r"D:\project\NCU\114\Jumior\python\FinalProject\clean_bg.jpg"
# 縮放後的圖片儲存路徑
OUTPUT_PATH = r"D:\project\NCU\114\Jumior\python\FinalProject\synthesis_result_640x480.jpg"
# 目標尺寸
TARGET_SIZE = (640, 480)
# ==============================================

def resize_single_image():
    print(f"正在讀取圖片: {INPUT_PATH}")
    
    # 使用 NumPy 讀取圖片，確保路徑相容性最高
    img_array = np.fromfile(INPUT_PATH, dtype=np.uint8)
    if img_array.size == 0:
        print("❌ 錯誤：找不到圖片，請確認路徑是否正確！")
        return
        
    img = cv2.imdecode(img_array, cv2.IMREAD_COLOR)
    
    if img is None:
        print("❌ 錯誤：圖片無法解析！")
        return

    print(f"原始尺寸: {img.shape[1]}x{img.shape[0]}")
    print(f"準備縮放至: {TARGET_SIZE[0]}x{TARGET_SIZE[1]}")

    # 執行縮放 (使用 cv2.INTER_AREA 確保畫質)
    resized_img = cv2.resize(img, TARGET_SIZE, interpolation=cv2.INTER_AREA)

    # 儲存結果
    is_success, buffer = cv2.imencode(".jpg", resized_img)
    if is_success:
        buffer.tofile(OUTPUT_PATH)
        print(f"🎉 大功告成！縮放後的圖片已儲存至：\n'{OUTPUT_PATH}'")
    else:
        print("❌ 錯誤：儲存圖片失敗。")

if __name__ == '__main__':
    resize_single_image()