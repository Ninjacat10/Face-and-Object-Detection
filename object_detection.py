import cv2
import numpy as np
import urllib.request
import os

model_urls = {
    "yolov4-tiny.weights": "https://github.com/AlexeyAB/darknet/releases/download/darknet_yolo_v4_pre/yolov4-tiny.weights",
    "yolov4-tiny.cfg": "https://raw.githubusercontent.com/AlexeyAB/darknet/master/cfg/yolov4-tiny.cfg",
    "coco.names": "https://raw.githubusercontent.com/AlexeyAB/darknet/master/data/coco.names"
}

def download_file(filename, url):
    if not os.path.exists(filename):
        print(f"Downloading {filename}...")
        urllib.request.urlretrieve(url, filename)

def calculate_iou(box1, box2):
    """Calculate the Intersection over Union (IoU) of two bounding boxes"""
    x1, y1, w1, h1 = box1
    x2, y2, w2, h2 = box2
    
    xi1 = max(x1, x2)
    yi1 = max(y1, y2)
    xi2 = min(x1+w1, x2+w2)
    yi2 = min(y1+h1, y2+h2)
    inter_area = max(0, xi2 - xi1) * max(0, yi2 - yi1)
    
    box1_area = w1 * h1
    box2_area = w2 * h2
    union_area = box1_area + box2_area - inter_area
    if union_area == 0: return 0
    return inter_area / union_area

class SmoothTracker:
    """A tracker that applies an Exponential Moving Average to bounding boxes to stop flickering!"""
    def __init__(self):
        self.tracked_objects = [] 

    def update(self, new_objects):
        updated_tracked = []
        
        # Match new frame objects to our existing tracked ones
        for new_obj in new_objects:
            best_iou = 0
            best_match_idx = -1
            
            for i, tracked_obj in enumerate(self.tracked_objects):
                if new_obj['type'] == tracked_obj['type']:
                    iou = calculate_iou(new_obj['box'], tracked_obj['box'])
                    if iou > best_iou:
                        best_iou = iou
                        best_match_idx = i
                        
            if best_iou > 0.3:
                # MATCH FOUND! Blend them using Exponential Moving Average to remove jitter/flicker
                matched = self.tracked_objects.pop(best_match_idx)
                
                # 70% of the old box, 30% of the new box (creating a beautifully smooth transition)
                old_box = np.array(matched['box'], dtype=float)
                new_box = np.array(new_obj['box'], dtype=float)
                smoothed_box = (old_box * 0.7 + new_box * 0.3).astype(int).tolist()
                
                # Update probabilities and tracking count
                updated_tracked.append({
                    'box': smoothed_box,
                    'preds': new_obj['preds'], 
                    'missed': 0,
                    'type': new_obj['type']
                })
            else:
                # No match, completely new unique object
                new_obj['missed'] = 0
                updated_tracked.append(new_obj)
                
        # Persist objects that weren't seen in this exact split-second frame
        # This stops the 'flashing on and off' flicker entirely!
        for tracked_obj in self.tracked_objects:
            tracked_obj['missed'] += 1
            if tracked_obj['missed'] < 5: # Keep alive for 5 frames
                updated_tracked.append(tracked_obj)
                
        self.tracked_objects = updated_tracked
        return self.tracked_objects

def main():
    for filename, url in model_urls.items():
        download_file(filename, url)

    with open("coco.names", "r") as f:
        classes = [line.strip() for line in f.readlines()]
        
    # YOLO Model for generic objects
    net = cv2.dnn.readNet("yolov4-tiny.weights", "yolov4-tiny.cfg")
    net.setPreferableBackend(cv2.dnn.DNN_BACKEND_OPENCV)
    net.setPreferableTarget(cv2.dnn.DNN_TARGET_CPU)
    output_layers = net.getUnconnectedOutLayersNames()

    # Haar Cascade intentionally added back to get the extremely TIGHT facial bounding boxes
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

    cap = cv2.VideoCapture(0)
    tracker = SmoothTracker()

    # Highly Visible Core Colors requested (BGR format)
    COLORS = {
        'face': (0, 0, 255),       # Red
        'object': (255, 0, 0),     # Blue
        'text_bg': (0, 0, 0),      # Black 
        'alt_pred': (0, 150, 0)    # Dark Green
    }

    print("Model ready! Press 'q' to quit.")

    while True:
        ret, frame = cap.read()
        if not ret: break

        height, width, _ = frame.shape
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        current_frame_objects = []

        # 1. Detect TIGHT Faces using Haar Cascade
        faces, _, face_weights = face_cascade.detectMultiScale3(
            gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30), outputRejectLevels=True
        )
        if len(faces) > 0:
            for i in range(len(faces)):
                x, y, w, h = faces[i]
                weight = float(face_weights[i])
                prob = min(max((weight / 10.0), 0), 0.999)
                current_frame_objects.append({
                    'type': 'face',
                    'box': [x, y, w, h],
                    'preds': [(-1, prob)] # -1 signifies 'Face'
                })

        # 2. Detect Generic Objects using YOLO
        blob = cv2.dnn.blobFromImage(frame, 0.00392, (320, 320), (0, 0, 0), True, crop=False)
        net.setInput(blob)
        outs = net.forward(output_layers)

        class_ids = []
        confidences = []
        boxes = []

        for out in outs:
            for detection in out:
                scores = detection[5:]
                top_classes = np.argsort(scores)[-3:][::-1]
                best_class_id = top_classes[0]
                confidence = scores[best_class_id]
                
                # Rule added: If it predicts a 'person' (0), we skip it! YOLO draws a massive box
                # around the whole body, but we are handling human Faces cleanly using Haar above.
                if confidence > 0.4 and best_class_id != 0: 
                    center_x = int(detection[0] * width)
                    center_y = int(detection[1] * height)
                    w = int(detection[2] * width)
                    h = int(detection[3] * height)
                    x = int(center_x - w / 2)
                    y = int(center_y - h / 2)

                    boxes.append([x, y, w, h])
                    confidences.append(float(confidence))
                    
                    top_preds = [(cls_id, float(scores[cls_id])) for cls_id in top_classes if scores[cls_id] > 0.05]
                    class_ids.append(top_preds)

        indices = cv2.dnn.NMSBoxes(boxes, confidences, 0.4, 0.3)
        if len(indices) > 0:
            for i in indices.flatten():
                current_frame_objects.append({
                    'type': 'object',
                    'box': boxes[i],
                    'preds': class_ids[i]
                })

        # 3. Smoothen all boxes!
        tracked_objects = tracker.update(current_frame_objects)

        # 4. Draw boxes with custom colors
        for obj in tracked_objects:
            x, y, w, h = obj['box']
            obj_type = obj['type']
            preds = obj['preds']
            
            box_color = COLORS['face'] if obj_type == 'face' else COLORS['object']
            
            # Highlight Bounding Box 
            cv2.rectangle(frame, (x, y), (x + w, y + h), box_color, 2)
            
            # Predict and Paint Text!
            for idx, (cls_id, prob) in enumerate(preds):
                if obj_type == 'face':
                    label = "Face"
                else:
                    label = str(classes[cls_id]).capitalize()
                    
                text = f"#{idx+1} {label}: {prob * 100:.1f}%" if obj_type != 'face' else f"Face: {prob * 100:.1f}%"
                
                y_offset = y - 10 - ((len(preds) - 1 - idx) * 20)
                if y_offset < 20: 
                    y_offset = y + 20 + (idx * 20)
                    
                # Draw a stark BLACK background behind the text to guarantee you can see it
                (text_w, text_h), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
                cv2.rectangle(frame, (x, y_offset - text_h - 4), (x + text_w, y_offset + 4), COLORS['text_bg'], -1)
                
                # Apply your custom visible colors:
                if idx == 0 and obj_type == 'face':
                    text_color = COLORS['face'] # Primary Face guess = Red
                elif idx == 0 and obj_type == 'object':
                    text_color = COLORS['object'] # Primary Object guess = Blue
                else:
                    text_color = COLORS['alt_pred'] # 2nd/3rd Alternative Guess = Dark Green

                cv2.putText(frame, text, (x, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.6, text_color, 2)

        cv2.imshow("Steady Face & Object Detection", frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
