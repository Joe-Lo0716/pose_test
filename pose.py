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
video_path = 'D:/Pose Train/pushup3.mp4'
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
num_images = 100  # 要提取的圖片數量
interval = max(1, total_frames // num_images)  # 計算提取間隔，確保間隔至少為1

# 初始化姿勢檢測模型
with mp_pose.Pose(
    static_image_mode=False,
    model_complexity=1,
    min_detection_confidence=0.5,
    enable_segmentation=True,
    min_tracking_confidence=0.5) as pose:

    # 檢查是否有寫入 CSV 的權限
    output_dir = os.path.dirname(output_csv_path)
    if not os.access(output_dir, os.W_OK):
        print(f"Cannot write to directory: {output_dir}")
        exit()

    # 初始化 CSV 文件，保存關鍵點數據
    with open(output_csv_path, mode='w', newline='') as file:
        writer = csv.writer(file)
        # CSV 標頭：每個關鍵點的 x, y, z 坐標，並加上動作標籤
        writer.writerow([f"x_{i}" for i in range(33)] + [f"y_{i}" for i in range(33)] + [f"z_{i}" for i in range(33)] + ['label'])

        frame_count = 0  # 用於追蹤目前的幀數
        image_count = 0  # 用於追蹤儲存的圖片數量

        while cap.isOpened() and image_count < num_images:
            ret, img = cap.read()
            if not ret:
                print("End of video or cannot receive frame.")
                break

            img = cv2.resize(img, (520, 300))
            img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

            # 處理影片中的每一幀
            results = pose.process(img_rgb)

            # 收集姿勢數據並檢查保存條件
            if results.pose_landmarks and frame_count % interval == 0:
                landmarks = results.pose_landmarks.landmark
                pose_data = []
                for landmark in landmarks:
                    pose_data.extend([landmark.x, landmark.y, landmark.z])

                # 保存數據到 CSV，假設該影片為跳躍動作，標記為 "jump"
                writer.writerow(pose_data + ["push-up"])
                print(f"Frame {frame_count}: Pose data saved to CSV.")

                # 保存圖片並增加圖片計數
                image_count += 1
                output_image_path = os.path.join(output_image_dir, f'image_{image_count}.jpg')
                cv2.imwrite(output_image_path, img)
                print(f"Saved: {output_image_path}")

            # 當檢測到 segmentation_mask 時更換背景
            if results.segmentation_mask is not None:
                condition = np.stack((results.segmentation_mask,) * 3, axis=-1) > 0.1
                img = np.where(condition, img, cv2.resize(bg, (img.shape[1], img.shape[0])))

            # 繪製人體姿勢標誌
            mp_drawing.draw_landmarks(
                img,
                results.pose_landmarks,
                mp_pose.POSE_CONNECTIONS,
                landmark_drawing_spec=mp_drawing_styles.get_default_pose_landmarks_style()
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


