import speech_recognition as sr
import whisper
import serial  # 假設使用 UART/USB 控制 SO-101
import time

# 1. 載入 Whisper 模型
# 對於簡單的指令，'base' 或 'small' 模型通常已經夠準確，且推論速度快很多
print("載入 Whisper 模型中...")
model = whisper.load_model("small") 
print("模型載入完成！")

# 2. 初始化機械手臂連線 (請替換為你的實際 COM port 與 Baudrate)
# arm_serial = serial.Serial('COM3', 115200, timeout=1)

# 3. 初始化收音器
recognizer = sr.Recognizer()
mic = sr.Microphone()

def control_arm(command_text):
    """根據辨識結果發送訊號給 SO-101"""
    if "開啟" in command_text or "啟動" in command_text:
        print(">> 執行動作：啟動機械手臂")
        # arm_serial.write(b'START\n')
    elif "停止" in command_text:
        print(">> 執行動作：停止機械手臂 (緊急煞車)")
        # arm_serial.write(b'STOP\n')
    else:
        print(f"忽略指令: {command_text}")

print("請開始說話 (說出「開啟」或「停止」)...")

with mic as source:
    # 稍微適應環境噪音
    recognizer.adjust_for_ambient_noise(source, duration=1)
    
    while True:
        try:
            # 聆聽麥克風，設定 timeout 避免永久卡死
            audio = recognizer.listen(source, timeout=5, phrase_time_limit=3)
            
            # 將音訊轉存為暫存檔供 Whisper 讀取 (也可以透過 byte buffer 處理)
            with open("temp.wav", "wb") as f:
                f.write(audio.get_wav_data())
            
            # 進行語音轉文字
            result = model.transcribe(
                "temp.wav", 
                language="zh", 
                initial_prompt="以下是給機械手臂的語音指令：開啟、停止、啟動、暫停。"
            )
            text = result["text"].strip()

            if text:
                # 移除標點符號 (Whisper 常會自動加句號)
                clean_text = text.replace("。", "").replace("，", "")
                print(f"辨識結果: {clean_text}")
                
                # 機械手臂的指令通常很短，如果長度超過 5 個字，極大概率是雜音或幻覺
                if len(clean_text) <= 5:
                    control_arm(clean_text)
                else:
                    print(f"指令過長，視為環境雜音忽略: {clean_text}")
                
        except sr.WaitTimeoutError:
            pass  # 沒聽到聲音，繼續迴圈
        except Exception as e:
            print(f"發生錯誤: {e}")