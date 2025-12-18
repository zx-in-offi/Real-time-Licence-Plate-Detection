import cv2
import easyocr
from ultralytics import YOLO


class NumberPlateDetector:
    def __init__(self, model_path: str):
        # Load YOLOv8 model
        self.model = YOLO(model_path)

        # Initialize OCR
        self.reader = easyocr.Reader(['en'], gpu=False)

    def detect(self, frame):
        """
        Detect number plates and read text using OCR
        :param frame: BGR image from OpenCV
        :return: annotated frame, detection list
        """

        original_h, original_w = frame.shape[:2]

        # Resize for YOLO (important for detection stability)
        resized = cv2.resize(frame, (640, 640))

        # Run YOLO inference
        results = self.model(
            resized,
            conf=0.05,   # LOW threshold for small dataset
            iou=0.3,
            imgsz=640,
            verbose=True
        )

        detections = []

        for r in results:
            print("Boxes detected:", len(r.boxes))  # DEBUG

            for box in r.boxes:
                conf = float(box.conf[0])
                print("Detected box confidence:", conf)
                x1, y1, x2, y2 = map(int, box.xyxy[0])

                # Scale boxes back to original image size
                x1 = int(x1 * original_w / 640)
                x2 = int(x2 * original_w / 640)
                y1 = int(y1 * original_h / 640)
                y2 = int(y2 * original_h / 640)

                # Safety clamp
                x1, y1 = max(0, x1), max(0, y1)
                x2, y2 = min(original_w, x2), min(original_h, y2)

                plate_crop = frame[y1:y2, x1:x2]

                plate_text = ""
                if plate_crop.size > 0:
                    # OCR preprocessing
                    gray = cv2.cvtColor(plate_crop, cv2.COLOR_BGR2GRAY)
                    gray = cv2.bilateralFilter(gray, 11, 17, 17)
                    _, thresh = cv2.threshold(
                        gray, 0, 255,
                        cv2.THRESH_BINARY + cv2.THRESH_OTSU
                    )

                    ocr_result = self.reader.readtext(thresh)
                    if ocr_result:
                        plate_text = ocr_result[0][1]

                detections.append({
                    "bbox": (x1, y1, x2, y2),
                    "text": plate_text
                })

                # Draw bounding box
                cv2.rectangle(
                    frame, (x1, y1), (x2, y2),
                    (0, 255, 0), 2
                )

                # Draw text
                label = plate_text if plate_text else "Plate"
                cv2.putText(
                    frame,
                    label,
                    (x1, y1 - 10),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.7,
                    (0, 255, 0),
                    2
                )

        return frame, detections
