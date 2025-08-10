import cv2
import os
import glob
from ultralytics import YOLO
import argparse

def get_available_models():
    """Get list of available trained models"""
    model_paths = []
    
    # Look for models in runs directory
    runs_dir = "runs/detect"
    if os.path.exists(runs_dir):
        for subdir in os.listdir(runs_dir):
            weights_dir = os.path.join(runs_dir, subdir, "weights")
            if os.path.exists(weights_dir):
                best_model = os.path.join(weights_dir, "best.pt")
                last_model = os.path.join(weights_dir, "last.pt")
                if os.path.exists(best_model):
                    model_paths.append(("best", best_model, subdir))
                if os.path.exists(last_model):
                    model_paths.append(("last", last_model, subdir))
    
    # Look for base models
    base_models = glob.glob("*.pt")
    for model in base_models:
        model_paths.append(("base", model, os.path.splitext(model)[0]))
    
    return model_paths

def select_model():
    """Interactive model selection"""
    models = get_available_models()
    
    if not models:
        print("No models found!")
        return None
    
    print("\nAvailable models:")
    print("-" * 50)
    for i, (model_type, path, name) in enumerate(models):
        print(f"{i+1}. [{model_type}] {name}")
        print(f"   Path: {path}")
    print("-" * 50)
    
    while True:
        try:
            choice = int(input(f"Select model (1-{len(models)}): ")) - 1
            if 0 <= choice < len(models):
                return models[choice][1]
            else:
                print(f"Please enter a number between 1 and {len(models)}")
        except ValueError:
            print("Please enter a valid number")

def test_webcam(model_path, camera_id=1):
    """Test model with webcam feed"""
    print(f"Loading model: {model_path}")
    model = YOLO(model_path)
    
    print(f"Opening camera {camera_id}...")
    cap = cv2.VideoCapture(camera_id)
    
    if not cap.isOpened():
        print(f"Error: Could not open camera {camera_id}")
        print("Available cameras: 0 (default), 1, 2...")
        return
    
    # Set camera properties for better performance
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    cap.set(cv2.CAP_PROP_FPS, 30)
    
    print("Press 'q' to quit, 's' to save current frame")
    
    frame_count = 0
    
    while True:
        ret, frame = cap.read()
        if not ret:
            print("Error reading frame from camera")
            break
        
        # Run inference
        results = model(frame)
        
        # Draw results on frame
        annotated_frame = results[0].plot()
        
        # Display FPS and detection info
        detections = results[0].boxes
        if detections is not None:
            num_detections = len(detections)
            cv2.putText(annotated_frame, f"Detections: {num_detections}", 
                       (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        
        cv2.putText(annotated_frame, f"Frame: {frame_count}", 
                   (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        
        # Show frame
        cv2.imshow("Poker Chip Detection", annotated_frame)
        
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
        elif key == ord('s'):
            save_path = f"detection_frame_{frame_count}.jpg"
            cv2.imwrite(save_path, annotated_frame)
            print(f"Frame saved as {save_path}")
        
        frame_count += 1
    
    cap.release()
    cv2.destroyAllWindows()
    print("Camera released and windows closed")

def main():
    parser = argparse.ArgumentParser(description="Test YOLO model with webcam")
    parser.add_argument("--model", type=str, help="Path to model file")
    parser.add_argument("--camera", type=int, default=1, help="Camera ID (default: 1)")
    
    args = parser.parse_args()
    
    if args.model and os.path.exists(args.model):
        model_path = args.model
    else:
        model_path = select_model()
        if not model_path:
            return
    
    print(f"\nStarting webcam test with model: {model_path}")
    print(f"Using camera: {args.camera}")
    
    test_webcam(model_path, args.camera)

if __name__ == "__main__":
    main()