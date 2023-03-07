from ultralytics import YOLO

# last letter determines model size (nano,small,medium,large,x-large)
# larger models are slower but more accurate
# "yolov8n.pt"  # nano, fastest
# "yolov8s.pt"  # small, acceptable, but unstable
# "yolov8m.pt"  # medium, more stable, better results, decent overall quality
model = YOLO("yolov8m.pt")  # nano

# main loop
if __name__ == '__main__':
    while True:
        result = model.predict(source="0", show=True)
