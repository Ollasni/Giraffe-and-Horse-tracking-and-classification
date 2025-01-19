import cv2
import numpy as np

video_path = 'input.mp4'

cap = cv2.VideoCapture(video_path)
i = 0
frame_skip = 5
frames = []
while True:
    for _ in range(frame_skip):
        cap.grab()
    ret, frame = cap.read()
    i += 1
    print(i)
    if not ret:
        break
    frames.append(frame)

cap.release()

if len(frames) == 0:
    print("Не удалось считать кадры из видео.")
    exit()


stack = np.stack(frames, axis=3)


median_background = np.median(stack, axis=3).astype(np.uint8)


cv2.imwrite('background.jpg', median_background)
print("Фоновое изображение сохранено в файл background.jpg")
