import cv2
import os

video_path = r'Tenis.mp4'
output_folder = r'Tenis'
os.makedirs(output_folder, exist_ok=True)

frame_interval = 1  
cap = cv2.VideoCapture(video_path)

frame_count = 0
image_count = 0

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break
    
    if frame_count % frame_interval == 0:
        image_path = os.path.join(output_folder, f'image_{image_count:05d}.jpg')
        cv2.imwrite(image_path, frame)
        image_count += 1
    
    frame_count += 1

cap.release()
print(f'Extracción completa: {image_count} imágenes guardadas.')