import cv2
import numpy as np
import glob
import os

def generate_clean_background(image_folder, output_path="clean_bg.jpg"):
    print(f"正在讀取資料夾 '{image_folder}' 中的影像...")
    
    # 取得資料夾內所有圖片 (支援 jpg, png)
    exts = ('*.jpg', '*.jpeg', '*.png')
    image_paths = []
    for ext in exts:
        image_paths.extend(glob.glob(os.path.join(image_folder, ext)))
        
    if not image_paths:
        print("❌ 錯誤：找不到任何影像檔案，請確認資料夾路徑。")
        return

    images = []
    first_shape = None

    for path in image_paths:
        # 🌟 修正中文路徑讀圖問題 🌟
        # 使用 numpy 先讀取原始位元組，再用 cv2.imdecode 解碼
        img_array = np.fromfile(path, dtype=np.uint8)
        img = cv2.imdecode(img_array, cv2.IMREAD_COLOR)
        
        if img is None:
            print(f"⚠️ 無法解析影像：{path}")
            continue
            
        # 必須確保所有圖片尺寸一致，才能進行矩陣運算
        if first_shape is None:
            first_shape = img.shape
        elif img.shape != first_shape:
            print(f"⚠️ 忽略尺寸不符的影像：{path}")
            continue
            
        images.append(img)

    img_count = len(images)
    print(f"✅ 成功讀取 {img_count} 張有效影像。")
    
    if img_count == 0:
        print("❌ 錯誤：沒有成功讀取任何影像，程式終止。")
        return

    # 建議至少要有幾張物體位置不一樣的照片，效果才會好
    if img_count < 3:
        print("⚠️ 警告：影像數量過少，如果物體都停在同一個地方，可能會無法完全消除。")

    print("正在透過時間軸中位數運算 (Temporal Median) 提取純淨背景...")
    
    # 將所有圖片堆疊成一個巨大的 4D NumPy 陣列 (N張圖片, H高度, W寬度, C色彩通道)
    image_stack = np.array(images)

    # 沿著 N 維度 (axis=0) 取像素的中位數
    median_bg = np.median(image_stack, axis=0).astype(np.uint8)

    # 儲存結果
    cv2.imwrite(output_path, median_bg)
    print(f"🎉 大功告成！純淨背景已儲存為：'{output_path}'")
    
    # 顯示合成出來的乾淨背景
    h, w = median_bg.shape[:2]
    display_img = cv2.resize(median_bg, (int(w*0.5), int(h*0.5)))
    cv2.imshow("Synthesized Clean Background", display_img)
    print("請點擊圖片視窗並按鍵盤任意鍵結束程式...")
    cv2.waitKey(0)
    cv2.destroyAllWindows()

if __name__ == '__main__':
    # ================= 參數設定區 =================
    # 使用 r 前綴來避免 Windows 反斜線跳脫問題
    FOLDER_PATH = r"C:\Users\jeffe\OneDrive\圖片\Camera Roll" 
    # ==============================================
    
    generate_clean_background(FOLDER_PATH)