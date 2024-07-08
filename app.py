import cv2
from ultralytics import YOLO

# Model training
base_model = YOLO('../yolov8n.pt')
base_model.train(data='../dataset/data.yaml', epochs=30, batch=8, plots=True)

# Trained model
best_model = YOLO(r'..\runs\detect\train\weights\best.pt')

# Video for test
video_path = 'drill.mp4'
cap = cv2.VideoCapture(video_path)

fps = cap.get(cv2.CAP_PROP_FPS)
wait_time = int(1000 / fps)

colors = {
    'pistol': (0, 0, 255),
    'rifle': (255, 0, 0),
    'default': (0, 255, 0)
}
while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    results = best_model(frame)

    for result in results:
        for box in result.boxes:
            x1, y1, x2, y2 = box.xyxy[0].cpu().numpy().astype(int)
            conf = box.conf.cpu().numpy()[0]
            class_id = box.cls.cpu().numpy()[0]
            class_name = best_model.names[int(class_id)]
            label = f"{class_name}: {conf:.2f}"

            color = colors.get(class_name, colors['default'])

            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
            cv2.putText(frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, color, 2)

    cv2.imshow('weapon detection', frame)
    if cv2.waitKey(wait_time) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()