import cv2
import easyocr
from ultralytics import YOLO


class NumberPlateDetector:
    def __init__(self, model_path):
        self.model = YOLO(model_path)
        self.reader = easyocr.Reader(['en'], gpu=False)

    def detect(self, frame):
        results = self.model(frame, conf=0.4)
        detections = []

        for r in results:
            for box in r.boxes:
                x1, y1, x2, y2 = map(int, box.xyxy[0])

                plate_img = frame[y1:y2, x1:x2]

                text = ""
                if plate_img.size > 0:
                    ocr_result = self.reader.readtext(plate_img)
                    if ocr_result:
                        text = ocr_result[0][1]

                detections.append({
                    "bbox": (x1, y1, x2, y2),
                    "text": text
                })

                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.putText(
                    frame,
                    text if text else "Plate",
                    (x1, y1 - 10),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.7,
                    (0, 255, 0),
                    2
                )

        return frame, detections
