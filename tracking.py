import cv2
import numpy as np
import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
from torchvision import models
import os

MODEL_PATH = 'model_efficientnet_updated.pth'
INPUT_VIDEO = 'input.mp4'
OUTPUT_VIDEO = 'output.mp4'
BACKGROUND_IMAGE = 'background.jpg'


class_thresholds = {
    0: 0.80,  # Giraffe
    1: 0.80,  # Horse
    2: 0.80   # Noise
}

class SimpleTracker:
    def __init__(self, max_distance=50):
        self.tracks = {}
        self.next_id = 0
        self.max_distance = max_distance

    def update(self, detections):

        centroids = []
        for i, det in enumerate(detections):
            x, y, w, h = det
            cx = int(x + w / 2)
            cy = int(y + h / 2)
            centroids.append((i, cx, cy, det))

        unmatched = set(self.tracks.keys())
        match_info = {}

        for i, cx, cy, det in centroids:
            track_id = None
            min_dist = float('inf')
            for tid in list(unmatched):
                (tcx, tcy, _, _) = self.tracks[tid]
                dist = np.sqrt((cx - tcx)**2 + (cy - tcy)**2)
                if dist < min_dist:
                    min_dist = dist
                    track_id = tid

            if track_id is not None and min_dist < self.max_distance:
                self.tracks[track_id] = (cx, cy, det, 0)
                unmatched.remove(track_id)
                match_info[i] = track_id
            else:
                self.tracks[self.next_id] = (cx, cy, det, 0)
                match_info[i] = self.next_id
                self.next_id += 1

        to_delete = []
        for tid in unmatched:
            (cx, cy, det, age) = self.tracks[tid]
            age += 1
            self.tracks[tid] = (cx, cy, det, age)
            if age > 10:
                to_delete.append(tid)

        for tid in to_delete:
            del self.tracks[tid]

        return self.tracks, match_info


def adjust_brightness_contrast(frame, x, y, w, h, alpha=1.2, beta=30):
    roi = frame[y:y+h, x:x+w].astype(np.float32)
    roi = roi * alpha + beta
    roi = np.clip(roi, 0, 255).astype(np.uint8)
    frame[y:y+h, x:x+w] = roi
    return frame


def main():
    device = torch.device("mps" if torch.backends.mps.is_available() 
                          else "cuda" if torch.cuda.is_available() 
                          else "cpu")

    model = models.efficientnet_b0(weights=None)
    num_ftrs = model.classifier[1].in_features
    model.classifier[1] = nn.Linear(num_ftrs, 3) 
    model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
    model.eval()
    model.to(device)

    inference_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406],
                             [0.229, 0.224, 0.225])
    ])

    cap = cv2.VideoCapture(INPUT_VIDEO)
    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(OUTPUT_VIDEO, fourcc, fps, (width, height))

    background = cv2.imread(BACKGROUND_IMAGE)
    background = cv2.resize(background, (width, height))

    tracker = SimpleTracker()

    track_class = {}

    class_names = ['Giraffe', 'Horse', 'Noise']

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        diff = cv2.absdiff(frame, background)
        gray_diff = cv2.cvtColor(diff, cv2.COLOR_BGR2GRAY)
        _, fgmask = cv2.threshold(gray_diff, 50, 255, cv2.THRESH_BINARY)

        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        fgmask = cv2.morphologyEx(fgmask, cv2.MORPH_OPEN, kernel)
        fgmask = cv2.morphologyEx(fgmask, cv2.MORPH_CLOSE, kernel)

        contours, _ = cv2.findContours(fgmask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        valid_detections = []
        detection_classes = []

        for cnt in contours:
            if cv2.contourArea(cnt) < 500:
                continue
            x, y, w, h = cv2.boundingRect(cnt)

            roi = frame[y:y+h, x:x+w]
            roi_rgb = cv2.cvtColor(roi, cv2.COLOR_BGR2RGB)
            roi_pil = inference_transform(
                torchvision.transforms.functional.to_pil_image(roi_rgb)
            )
            roi_input = roi_pil.unsqueeze(0).to(device)

            with torch.no_grad():
                outputs = model(roi_input)
                probs = nn.functional.softmax(outputs, dim=1)[0]
                class_idx = torch.argmax(probs).item()
                class_label = class_names[class_idx]
                prob_val = probs[class_idx].item()

            threshold = class_thresholds.get(class_idx, 0.0)
            if (class_label != 'Noise') and (prob_val >= threshold):
                valid_detections.append((x, y, w, h))
                detection_classes.append((class_label, prob_val))

        tracks, match_info = tracker.update(valid_detections)

        for i, track_id in match_info.items():
            class_label, prob_val = detection_classes[i]
            track_class[track_id] = (class_label, prob_val)

        for tid, (cx, cy, (x, y, w, h), age) in tracks.items():
            frame = adjust_brightness_contrast(frame, x, y, w, h, alpha=1.3, beta=40)

            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

            if tid in track_class:
                class_label, prob_val = track_class[tid]
                text = f"{class_label}: {prob_val:.2f}"
                cv2.putText(frame, text, (x, max(y - 10, 0)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)

        out.write(frame)

    cap.release()
    out.release()
    cv2.destroyAllWindows()

if __name__ == '__main__':
    main()
