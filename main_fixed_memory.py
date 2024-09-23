from ultralytics import YOLO

if __name__ == '__main__':
    # last letter determines model size (nano,small,medium,large,x-large)
    # larger models are slower but more accurate
    # "yolov8n.pt"  # nano, fastest
    # "yolov8s.pt"  # small, acceptable, but unstable
    # "yolov8m.pt"  # medium, more stable, better results, decent overall quality
    model = YOLO("yolov8s.pt")

    # Run batched inference on a list of camera stream
    results = model(source="0", show=True, stream=True)

    # Access results generator
    for result in results:
        # You don't need to do anything with the result,
        # just accessing it is enough to show the live camera feed
        # and not terminate the program.
        continue
