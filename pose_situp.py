#Anaconda Prompt安裝 pip install opencv-python
#Anaconda Prompt安裝 pip install opencv-python-headless
#Anaconda Prompt安裝 pip install numpy
#Anaconda Prompt安裝 pip install mediapipe
#提取影片中 20% 的圖片：根據 total_frames 計算要提取的圖片數量，並確定提取間隔 interval，確保只保存 20% 的圖片。
#間隔計算：為避免 interval 為 0，在計算時加入 max(1, total_frames // num_images)，確保最小間隔為 1。
#保存關鍵點數據和角度：在 CSV 文件中保存每個關鍵點的 (x, y, z) 坐標，以及計算的角度和動作標籤（"squat"或"other"）
#仰臥起坐判斷：使用髖部、肩膀和膝蓋的角度來判斷仰臥起坐動作，並對角度範圍擴展 10% (50 < hip_angle < 140) 以容忍動作的不標準情況。
#輸出檔案：只有當 action_label == "sit_up" 時，才將數據寫入 CSV 檔案
#檢測條件：可以調整 min_detection_confidence 提高或降低姿態偵測的嚴格程度。

import cv2
import mediapipe as mp
import numpy as np
import csv
import os
import time

# 初始化 MediaPipe 的工具
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles
mp_pose = mp.solutions.pose

# 設定影片和輸出圖片、CSV 路徑
video_path = 'D:/Pose Train/situp.mp4'
background_image_path = 'D:/Pose Train/back.jpg'
output_csv_path = f'D:/Pose Train/pose_data_{int(time.time())}.csv'  # 添加時間戳以避免文件覆蓋
output_image_dir = 'D:/Pose Train/ExtractedImages'

# 確保輸出目錄存在
if not os.path.exists(output_image_dir):
    os.makedirs(output_image_dir)

cap = cv2.VideoCapture(video_path)
bg = cv2.imread(background_image_path)

if bg is None:
    print("Background image is None, check the path.")
    exit()

if not cap.isOpened():
    print("Cannot open video file. Check the video path.")
    exit()

total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))  # 獲取影片總幀數
num_images = int(total_frames * 0.2)  # 提取影片中 20% 的圖片數量
interval = max(1, total_frames // num_images)  # 計算提取間隔，確保間隔至少為1

# 計算關鍵點的夾角（用於動作判斷）
def calculate_angle(a, b, c):
    """計算三個點 (a, b, c) 所形成的角度"""
    a = np.array(a)  # 第一個點
    b = np.array(b)  # 第二個點
    c = np.array(c)  # 第三個點

    radians = np.arctan2(c[1] - b[1], c[0] - b[0]) - np.arctan2(a[1] - b[1], a[0] - b[0])
    angle = np.abs(radians * 180.0 / np.pi)

    # 角度標準化到 [0, 180] 範圍
    if angle > 180.0:
        angle = 360 - angle

    return angle

# 初始化姿勢檢測模型
with mp_pose.Pose(
    static_image_mode=False,
    model_complexity=2,  # 提高模型的精度
    min_detection_confidence=0.7,  # 提高檢測的信心值門檻
    enable_segmentation=True,
    min_tracking_confidence=0.7  # 提高跟踪的信心值門檻
) as pose:

    # 檢查是否有寫入 CSV 的權限
    output_dir = os.path.dirname(output_csv_path)
    if not os.access(output_dir, os.W_OK):
        print(f"Cannot write to directory: {output_dir}")
        exit()

    # 初始化 CSV 文件，保存關鍵點數據
    with open(output_csv_path, mode='w', newline='') as file:
        writer = csv.writer(file)
        # CSV 標頭：每個關鍵點的 x, y, z 坐標，並加上腰部角度和動作標籤
        headers = [f"x_{i}" for i in range(33)] + [f"y_{i}" for i in range(33)] + [f"z_{i}" for i in range(33)] + ['hip_angle', 'label']
        writer.writerow(headers)

        frame_count = 0  # 用於追蹤目前的幀數
        image_count = 0  # 用於追蹤儲存的圖片數量

        while cap.isOpened() and image_count < num_images:
            ret, img = cap.read()
            if not ret:
                print("End of video or cannot receive frame.")
                break

            # 圖像預處理
            img = cv2.resize(img, (520, 300))
            img = cv2.GaussianBlur(img, (5, 5), 0)  # 使用高斯模糊去噪
            img = cv2.convertScaleAbs(img, alpha=1.5, beta=0)  # 增強圖像對比度
            img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

            # 處理影片中的每一幀
            results = pose.process(img_rgb)

            # 收集姿勢數據並檢查保存條件
            if results.pose_landmarks and frame_count % interval == 0:
                landmarks = results.pose_landmarks.landmark
                pose_data = []
                for landmark in landmarks:
                    pose_data.extend([landmark.x, landmark.y, landmark.z])

                # 計算腰部角度（用於仰臥起坐檢測）
                hip = [landmarks[mp_pose.PoseLandmark.LEFT_HIP.value].x,
                       landmarks[mp_pose.PoseLandmark.LEFT_HIP.value].y]
                knee = [landmarks[mp_pose.PoseLandmark.LEFT_KNEE.value].x,
                        landmarks[mp_pose.PoseLandmark.LEFT_KNEE.value].y]
                shoulder = [landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].x,
                            landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].y]

                # 計算腰部的角度
                hip_angle = calculate_angle(shoulder, hip, knee)
                print(f"Detected hip angle: {hip_angle}")

                # 判斷腰部角度以確定是否為仰臥起坐動作（擴展10%範圍）
                if 50 < hip_angle < 140:  # 擴大仰臥起坐的角度範圍
                    action_label = "sit_up"
                else:
                    action_label = "other"

                # 只有當動作標記為 "sit_up" 才將數據保存到 CSV
                if action_label == "sit_up":
                    writer.writerow(pose_data + [hip_angle, action_label])
                    print(f"Frame {frame_count}: Pose data saved to CSV with label '{action_label}'.")

                # 保存圖片並增加圖片計數
                image_count += 1
                output_image_path = os.path.join(output_image_dir, f'image_{image_count}.jpg')
                cv2.imwrite(output_image_path, img)
                print(f"Saved: {output_image_path}")

            # 當檢測到 segmentation_mask 時更換背景
            if results.segmentation_mask is not None:
                condition = np.stack((results.segmentation_mask,) * 3, axis=-1) > 0.5  # 提高去背的門檻值
                img = np.where(condition, img, cv2.resize(bg, (img.shape[1], img.shape[0])))

            # 自定義關節點和骨骼線條的繪製樣式
            landmark_drawing_spec = mp_drawing.DrawingSpec(color=(0, 255, 0), thickness=5, circle_radius=3)
            connection_drawing_spec = mp_drawing.DrawingSpec(color=(0, 0, 255), thickness=2)

            # 繪製人體姿勢標誌
            mp_drawing.draw_landmarks(
                img,
                results.pose_landmarks,
                mp_pose.POSE_CONNECTIONS,
                landmark_drawing_spec=landmark_drawing_spec,
                connection_drawing_spec=connection_drawing_spec
            )

            # 顯示圖像視窗
            img = cv2.resize(img, (640, 720))
            cv2.imshow('Pose Studio', img)

            # 按下 q 鍵停止
            if cv2.waitKey(5) == ord('q'):
                break

            frame_count += 1

cap.release()
cv2.destroyAllWindows()

print(f"Pose data has been saved to {output_csv_path}.")