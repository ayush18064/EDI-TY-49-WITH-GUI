from ultralytics import YOLO
import torch

def main():
    # Step 1: Load the YOLOv8 model
    MODEL_PATH = 'yolov8n.pt'  # You can use 'yolov8n.pt', 'yolov8s.pt', etc.
    model = YOLO(MODEL_PATH)

    # Step 2: Train the model
    # Replace the 'data' path with the path to your YAML file (that includes train, val, and test dataset paths)
    DATA_YAML_PATH = 'data.yaml'  # Replace with the correct path to your .yaml file
    model.train(data=DATA_YAML_PATH, epochs=150,device="mps",batch=16)

    # Step 3: Evaluate the model on the validation set
    model.val()

    # Step 4: Make predictions on a new test image
    TEST_IMAGE_PATH = 'images.jpg'  # Replace with the path to your test image
    results = model.predict(source=TEST_IMAGE_PATH, save=True)

    # Results will be saved in a folder, and you can view the predictions.

if __name__ == '__main__':
    main()
