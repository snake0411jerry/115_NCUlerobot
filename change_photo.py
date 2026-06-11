import cv2
import numpy as np
from ultralytics import YOLO

# ================= 參數設定區 =================
# 1. 替換成你剛剛訓練出來的最佳權重路徑
MODEL_PATH = r'D:\project\NCU\114\Jumior\python\FinalProject\runs\segment\yolo_robot\0611\weights\best.pt'

# 2. 準備測試影像與乾淨背景影像的路徑
TEST_IMG_PATH = r"C:\Users\jeffe\OneDrive\圖片\Camera Roll\WIN_20260605_10_20_11_Pro.jpg" # 有雜物的照片
BG_IMG_PATH = r'D:\project\NCU\114\Jumior\python\FinalProject\clean_bg.jpg'   # 純淨背景照片
# ==============================================

def main():
    print("載入 YOLO 模型中...")
    model = YOLO(MODEL_PATH)

    print("讀取影像中...")
    test_img = cv2.imread(TEST_IMG_PATH)
    bg_img = cv2.imread(BG_IMG_PATH)

    if test_img is None or bg_img is None:
        print("錯誤：找不到影像檔案，請確認路徑是否正確！")
        return

    # 確保乾淨背景圖的尺寸與測試圖完全一致
    h, w = test_img.shape[:2]
    bg_img = cv2.resize(bg_img, (w, h))

    print("進行預測...")
    # 進行推論
    results = model(test_img)

    # 判斷是否有偵測到物件 (Masks)
    if results[0].masks is not None:
        print("✅ 成功偵測到目標物！正在進行背景替換...")
        
        mask = results[0].masks.data[0].cpu().numpy()
        mask = cv2.resize(mask, (w, h))
        mask = cv2.GaussianBlur(mask, (15, 15), 0)
        mask_3d = np.stack((mask, mask, mask), axis=-1)
        
        output_img = (test_img * mask_3d + bg_img * (1 - mask_3d)).astype(np.uint8)
        
    else:
        print("⚠️ 沒有偵測到目標物，將輸出純淨背景。")
        output_img = bg_img.copy()

    # ================= 顯示方式修改 =================
    # 將視窗稍微縮小 (50%)，確保在筆電螢幕上能完整顯示且不會出現灰邊
    display_w = int(w * 0.5)
    display_h = int(h * 0.5)
    
    test_img_resized = cv2.resize(test_img, (display_w, display_h))
    output_img_resized = cv2.resize(output_img, (display_w, display_h))
    
    # 開啟兩個獨立的視窗
    cv2.imshow('Original Image (Before)', test_img_resized)
    cv2.imshow('Masked Image (After)', output_img_resized)
    # ================================================

    # 儲存最終合成結果 (儲存的仍是高畫質原圖)
    cv2.imwrite('synthesis_result.jpg', output_img)
    print("合成結果已儲存為 'synthesis_result.jpg'")

    print("請點擊圖片視窗並按鍵盤任意鍵結束程式...")
    cv2.waitKey(0)
    cv2.destroyAllWindows()

if __name__ == '__main__':
    main()