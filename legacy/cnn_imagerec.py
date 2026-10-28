import cv2
import numpy as np
from ultralytics import YOLO
from tensorflow.keras.applications.mobilenet_v2 import (
    MobileNetV2,
    decode_predictions,
    preprocess_input,
)


classifier = MobileNetV2(weights="imagenet")
detector = YOLO("yolov8n.pt")
camera = cv2.VideoCapture(0)
mode = "mobilenet"

while camera.isOpened():
    ok, frame = camera.read()
    if not ok:
        break

    if mode == "yolo":
        frame = detector(frame, verbose=False)[0].plot()
    else:
        image = cv2.resize(frame, (224, 224))
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image = preprocess_input(np.expand_dims(image.astype(np.float32), axis=0))
        predictions = classifier.predict(image, verbose=0)
        _, name, score = decode_predictions(predictions, top=1)[0][0]
        cv2.putText(
            frame,
            f"{name}: {score:.0%}",
            (10, 40),
            cv2.FONT_HERSHEY_SIMPLEX,
            1,
            (0, 255, 0),
            2,
        )

    cv2.imshow("m=MobileNet, y=YOLO, q=quit", frame)
    key = cv2.waitKey(1) & 0xFF
    if key == ord("q"):
        break
    if key == ord("m"):
        mode = "mobilenet"
    elif key == ord("y"):
        mode = "yolo"

camera.release()
cv2.destroyAllWindows()
