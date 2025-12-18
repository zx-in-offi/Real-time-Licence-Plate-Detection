import cv2
import matplotlib.pyplot as plt
import sys
import os
# Add project root to sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from src.detector import NumberPlateDetector

detector = NumberPlateDetector("models/license_plate.pt")

img = cv2.imread("data/images/test.jpg")
output, detections = detector.detect(img)

print("Detections:", detections)

# Convert BGR → RGB for matplotlib
output_rgb = cv2.cvtColor(output, cv2.COLOR_BGR2RGB)

plt.figure(figsize=(10, 6))
plt.imshow(output_rgb)
plt.axis("off")
plt.title("YOLOv8 License Plate Detection + OCR")
plt.show()
