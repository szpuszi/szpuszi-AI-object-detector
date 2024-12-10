import cv2
import numpy as np
from ultralytics import YOLO
import time
import torch

def main():
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}")
    
    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
    
    model = YOLO('yolov8x.pt')
    
    colors = {
        'person': (0, 255, 0),
        'face': (255, 200, 0),
        'eye': (255, 0, 0),
        'nose': (0, 0, 255),
        'mouth': (128, 0, 255),
        'ear': (255, 0, 128),
        'hand': (0, 255, 255),
        'arm': (255, 128, 0),
        'leg': (128, 128, 0),
        'foot': (0, 128, 128),
        
        'cell phone': (255, 0, 0),
        'laptop': (0, 165, 255),
        'keyboard': (255, 0, 255),
        'mouse': (128, 0, 128),
        'remote': (128, 128, 255),
        'tv': (255, 255, 0),
        'monitor': (0, 255, 128),
        
        'chair': (128, 64, 0),
        'couch': (0, 128, 255),
        'bed': (255, 128, 128),
        'dining table': (128, 0, 0),
        'toilet': (192, 192, 192),
        'sink': (128, 128, 128),
        
        'bottle': (0, 255, 255),
        'cup': (255, 128, 255),
        'book': (0, 0, 128),
        'clock': (128, 255, 0),
        'vase': (255, 0, 128),
        'scissors': (128, 0, 255),
        'toothbrush': (0, 255, 128),
        'hair drier': (128, 128, 0),
        
        'backpack': (64, 128, 0),
        'umbrella': (192, 0, 192),
        'handbag': (128, 64, 128),
        'tie': (0, 128, 64),
        
        'banana': (255, 255, 128),
        'apple': (128, 255, 128),
        'sandwich': (255, 128, 64),
        'orange': (0, 128, 255),
        'pizza': (128, 0, 64),
        
        'bicycle': (255, 192, 0),
        'car': (0, 192, 255),
        'motorcycle': (192, 0, 255),
        
        'dog': (128, 255, 255),
        'cat': (255, 128, 255),
        'bird': (255, 255, 0),
    }
    
    print("AI Detektor uruchomiony. Naciśnij 'q' aby wyjść.")
    print(f"Wykrywane obiekty ({len(colors)} kategorii):")
    for obj in colors.keys():
        print(f"- {obj}")
    
    while True:
        start_time = time.time()
        ret, frame = cap.read()
        if not ret:
            print("Błąd: Nie można odczytać klatki z kamery.")
            break
            
        results = model(frame, 
                       conf=0.45,
                       iou=0.45,
                       device=device)
        
        detected_counts = {}
        
        for result in results:
            boxes = result.boxes
            for box in boxes:
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                class_id = int(box.cls[0])
                conf = float(box.conf[0])
                class_name = result.names[class_id]
                
                if class_name in colors:
                    detected_counts[class_name] = detected_counts.get(class_name, 0) + 1
                    
                    color = colors.get(class_name, (0, 255, 0))
                    cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
                    
                    label = f'{class_name} {conf:.2f}'
                    cv2.putText(frame, 
                              label, 
                              (x1, y1-10), 
                              cv2.FONT_HERSHEY_SIMPLEX, 
                              0.6, 
                              color, 
                              2)
        
        fps = 1.0 / (time.time() - start_time)
        cv2.putText(frame, f'FPS: {fps:.1f}', 
                   (10, 30), 
                   cv2.FONT_HERSHEY_SIMPLEX, 
                   1, 
                   (255, 255, 255), 
                   2)
        
        y_position = 60
        for obj_name, count in detected_counts.items():
            text = f'{obj_name}: {count}'
            cv2.putText(frame, 
                       text, 
                       (10, y_position), 
                       cv2.FONT_HERSHEY_SIMPLEX, 
                       0.7, 
                       colors.get(obj_name, (0, 255, 0)), 
                       2)
            y_position += 30
            
            if y_position > frame.shape[0] - 30:
                y_position = 60
        
        cv2.imshow('AI Multi-Object Detector', frame)
        
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main() 