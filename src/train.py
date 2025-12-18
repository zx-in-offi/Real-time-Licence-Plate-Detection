from ultralytics import YOLO

model = YOLO("yolov8n.pt")

model.train(
    data="data/licence_plate_dataset/data.yaml",
    epochs=20,
    imgsz=640,
    batch=8,
    device="cpu"
)
