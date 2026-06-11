from ultralytics import YOLO

def main():
    print("初始化 YOLOv8-seg 模型...")
    # 載入 YOLOv8 nano 版本的實例分割預訓練權重 (這會自動下載 yolov8n-seg.pt)
    model = YOLO("yolov8n-seg.pt")

    print("開始訓練...")
    # 開始訓練，請注意 data 的路徑要指向你正確的 yaml 位置
    results = model.train(
        data=r"D:\project\NCU\114\Jumior\python\FinalProject\My First Project.v1i.yolov11\data.yaml", # 資料集設定檔的絕對路徑
        epochs=100,               # 訓練週期數 (如果提早收斂，YOLO 會自動 early stopping)
        imgsz=640,                # 影像輸入尺寸
        batch=16,                 # 批次大小 (如果你的 GPU VRAM 不夠，可以改成 8 或 4)
        device=0,                 # 0 代表使用第一張 GPU。如果你沒有 GPU，請改成 'cpu'
        project="yolo_robot",     # 儲存訓練結果的主資料夾名稱
        name="eraser_seg",        # 這次訓練的子資料夾名稱
        degrees=180.0,            # 隨機旋轉 -180~180 度
        fliplr=0.5,               # 50% 機率水平翻轉
        flipud=0.5,               # 50% 機率垂直翻轉
        hsv_v=0.3,                # 隨機改變明暗度 (防止死背反光)
        mosaic=1.0,               # 開啟馬賽克增強 (把多張圖拼在一起，對小物件極度有效)
        mixup=0.2,                # 隨機混合圖片
        
        # --- 凍結骨幹網路 ---
        freeze=10,                # <--- 加上了逗號
        workers=0,                # <--- 建議 Windows 先設為 0 避免 DataLoader 報錯
        patience=20               # 如果 20 個 epochs 精準度都沒有提升，就提早結束訓練
    )

    print("訓練結束！")
    print(f"你的最佳權重檔 (best.pt) 已經儲存在: yolo_robot/eraser_seg/weights/best.pt")

if __name__ == '__main__':
    # 在執行 PyTorch 多執行緒資料加載時，必須要包在 __main__ 裡面
    main()