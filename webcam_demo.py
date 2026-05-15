"""
A presentation-ready real-time ASL classifier with:
  - Large, readable prediction label (visible from the back of a classroom)
  - Per-class confidence bars (so the audience sees the full distribution)
  - Temporal smoothing (averages logits over a sliding window) to stop flicker
  - Stability indicator that only "commits" a prediction when it's held steady
  - Header bar with project title and author
  - FPS counter in the corner


Press 'q' to quit.
Press 'f' to toggle full-screen.

Requires: opencv-python, torch, torchvision, numpy
"""

import argparse
import time
from collections import deque

import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models, transforms


# ---------- Display constants ------------------------------------------------
# Sized for a laptop screen during a classroom presentation.
WINDOW_W, WINDOW_H = 1280, 720

# Colors are BGR (because OpenCV).
NAVY = (97, 39, 30)        
ICE = (252, 220, 202)
WHITE = (255, 255, 255)
LIGHT_GRAY = (240, 240, 240)
DARK = (31, 41, 55)
MUTED = (150, 150, 150)
GREEN = (143, 157, 42)
CORAL = (81, 111, 231)

# Layout (in pixels, on a 1280x720 canvas)
HEADER_H = 60
FOOTER_H = 200             # bottom panel where the bars + big label live
CAM_AREA_TOP = HEADER_H
CAM_AREA_BOTTOM = WINDOW_H - FOOTER_H

# Smoothing window: average softmax probs over this many recent frames.
SMOOTHING_WINDOW = 7

# A prediction is "committed" only if it's been the top class for this many frames in a row.
STABILITY_FRAMES = 10


# ---------- Model ------------------------------------------------------------
def build_model(num_classes: int) -> nn.Module:
    model = models.mobilenet_v2(weights=None)
    in_features = model.classifier[1].in_features
    model.classifier = nn.Sequential(
        nn.Dropout(0.3),
        nn.Linear(in_features, num_classes),
    )
    return model


def pretty_label(class_name: str) -> str:
    """Convert internal class names to human-readable labels for display."""
    return {
        "i_love_you": "I LOVE YOU",
        "yes": "YES",
        "stop": "STOP",
        "hello": "HELLO",
    }.get(class_name, class_name.replace("_", " ").upper())


# ---------- Drawing helpers --------------------------------------------------
def draw_filled_rect(canvas, p1, p2, color):
    cv2.rectangle(canvas, p1, p2, color, thickness=-1)


def draw_text(canvas, text, org, font_scale, color, thickness=1, font=cv2.FONT_HERSHEY_SIMPLEX):
    cv2.putText(canvas, text, org, font, font_scale, color, thickness, cv2.LINE_AA)


def text_size(text, font_scale, thickness, font=cv2.FONT_HERSHEY_SIMPLEX):
    (w, h), _ = cv2.getTextSize(text, font, font_scale, thickness)
    return w, h


def render_frame(canvas, cam_frame, smoothed_probs, class_names,
                 committed_label, stability_count, fps):
    """Draw everything onto the 1280x720 canvas."""
    canvas[:] = LIGHT_GRAY  # background

    # ----- Header bar -----
    draw_filled_rect(canvas, (0, 0), (WINDOW_W, HEADER_H), NAVY)
    draw_text(canvas, "ASL TRANSLATOR", (24, 40), 0.85, WHITE, thickness=2)
    draw_text(canvas, "Yuval Paz  |  Computer Vision  |  Spring 2026",
              (320, 38), 0.55, ICE, thickness=1)
    fps_text = f"{fps:4.1f} fps"
    fw, _ = text_size(fps_text, 0.55, 1)
    draw_text(canvas, fps_text, (WINDOW_W - fw - 24, 38), 0.55, ICE, thickness=1)

    # ----- Camera area -----
    # The cam frame keeps its aspect; we letterbox it into the available area.
    cam_h = CAM_AREA_BOTTOM - CAM_AREA_TOP
    cam_w = WINDOW_W
    fh, fw = cam_frame.shape[:2]
    scale = min(cam_w / fw, cam_h / fh)
    new_w, new_h = int(fw * scale), int(fh * scale)
    resized = cv2.resize(cam_frame, (new_w, new_h))
    x_off = (cam_w - new_w) // 2
    y_off = CAM_AREA_TOP + (cam_h - new_h) // 2
    canvas[y_off:y_off + new_h, x_off:x_off + new_w] = resized

    # Thin border around the camera region
    cv2.rectangle(canvas, (x_off - 2, y_off - 2),
                  (x_off + new_w + 2, y_off + new_h + 2), NAVY, thickness=2)

    # ----- Footer panel -----
    footer_top = WINDOW_H - FOOTER_H
    draw_filled_rect(canvas, (0, footer_top), (WINDOW_W, WINDOW_H), WHITE)
    draw_filled_rect(canvas, (0, footer_top), (WINDOW_W, footer_top + 4), CORAL)

    # Left side: BIG committed label
    label_x = 32
    label_y = footer_top + 70
    if committed_label is not None:
        draw_text(canvas, "RECOGNIZED:", (label_x, label_y - 30), 0.55, MUTED, thickness=1)
        draw_text(canvas, committed_label, (label_x, label_y + 50), 1.8, NAVY, thickness=4)
        draw_text(canvas, "● STABLE", (label_x, footer_top + FOOTER_H - 30),
                  0.55, GREEN, thickness=2)
    else:
        draw_text(canvas, "WAITING FOR STABLE PREDICTION...", (label_x, label_y),
                  0.7, MUTED, thickness=1)
        # Stability progress bar
        progress = min(stability_count / STABILITY_FRAMES, 1.0)
        bar_x, bar_y, bar_w, bar_h = label_x, label_y + 30, 200, 6
        cv2.rectangle(canvas, (bar_x, bar_y), (bar_x + bar_w, bar_y + bar_h),
                      MUTED, thickness=1)
        cv2.rectangle(canvas, (bar_x, bar_y),
                      (bar_x + int(bar_w * progress), bar_y + bar_h),
                      CORAL, thickness=-1)

    # Right side: per-class confidence bars
    bars_x = 620
    bars_y = footer_top + 28
    bar_w_max = 460
    bar_h = 22
    bar_gap = 12
    draw_text(canvas, "CONFIDENCE", (bars_x, bars_y - 6), 0.5, MUTED, thickness=1)

    for i, cname in enumerate(class_names):
        prob = float(smoothed_probs[i])
        y = bars_y + 16 + i * (bar_h + bar_gap)

        label_str = pretty_label(cname)
        draw_text(canvas, label_str, (bars_x, y + bar_h - 6), 0.55, DARK, thickness=1)

        b_x = bars_x + 130
        cv2.rectangle(canvas, (b_x, y), (b_x + bar_w_max, y + bar_h), LIGHT_GRAY, thickness=-1)

        fill_w = int(bar_w_max * prob)
        is_top = i == int(np.argmax(smoothed_probs))
        bar_color = NAVY if is_top else ICE
        cv2.rectangle(canvas, (b_x, y), (b_x + fill_w, y + bar_h), bar_color, thickness=-1)

        pct = f"{prob * 100:5.1f}%"
        draw_text(canvas, pct, (b_x + bar_w_max + 10, y + bar_h - 6),
                  0.55, DARK if is_top else MUTED,
                  thickness=2 if is_top else 1)

    return canvas


# ---------- Main loop --------------------------------------------------------
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default="results/models/best_model.pth")
    parser.add_argument("--cam", type=int, default=0)
    parser.add_argument("--threshold", type=float, default=0.55,
                        help="Min smoothed prob to commit a prediction.")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available()
                          else "mps" if torch.backends.mps.is_available()
                          else "cpu")
    print(f"Using device: {device}")
    ckpt = torch.load(args.model, map_location=device, weights_only=False)
    class_names = ckpt["class_names"]
    print(f"Classes: {class_names}")

    model = build_model(num_classes=len(class_names)).to(device)
    model.load_state_dict(ckpt["state_dict"])
    model.eval()

    tf = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    cap = cv2.VideoCapture(args.cam)
    if not cap.isOpened():
        raise RuntimeError("Could not open webcam. On macOS, check System Settings → "
                           "Privacy & Security → Camera, and allow Terminal.")

    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

    canvas = np.zeros((WINDOW_H, WINDOW_W, 3), dtype=np.uint8)

    prob_history = deque(maxlen=SMOOTHING_WINDOW)
    last_top_idx = None
    stability_count = 0
    committed_idx = None

    fps = 0.0
    fps_alpha = 0.9
    prev_t = time.time()

    window_name = "ASL Translator"
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(window_name, WINDOW_W, WINDOW_H)
    is_fullscreen = False

    print("Demo running. Press 'q' to quit, 'f' to toggle full-screen.")
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # Mirror the frame for natural user experience.
        frame = cv2.flip(frame, 1)

        # Use a center-square crop as model input.
        # NOTE: the model was trained WITHOUT horizontal flip augmentation, because
        # flipping changes which side of the hand faces the camera in ASL.
        # The mirror above is for the user's comfort while signing — we un-flip
        # before passing to the model so it sees the world the way it was trained.
        h, w = frame.shape[:2]
        s = min(h, w)
        y0, x0 = (h - s) // 2, (w - s) // 2
        center_crop = frame[y0:y0 + s, x0:x0 + s]
        model_input = cv2.flip(center_crop, 1)
        rgb = cv2.cvtColor(model_input, cv2.COLOR_BGR2RGB)

        x = tf(rgb).unsqueeze(0).to(device)
        with torch.no_grad():
            logits = model(x)
            probs = F.softmax(logits, dim=1).squeeze(0).cpu().numpy()

        # Temporal smoothing
        prob_history.append(probs)
        smoothed = np.mean(np.stack(prob_history, axis=0), axis=0)

        # Stability tracking
        top_idx = int(np.argmax(smoothed))
        top_prob = float(smoothed[top_idx])
        if top_idx == last_top_idx:
            stability_count += 1
        else:
            stability_count = 1
            last_top_idx = top_idx

        if stability_count >= STABILITY_FRAMES and top_prob >= args.threshold:
            committed_idx = top_idx
        elif top_prob < args.threshold * 0.8:
            committed_idx = None

        committed_label = pretty_label(class_names[committed_idx]) if committed_idx is not None else None

        # Guide rectangle on the camera frame
        cv2.rectangle(frame, (x0, y0), (x0 + s, y0 + s), (0, 220, 0), 2)

        # FPS update
        now = time.time()
        dt = now - prev_t
        prev_t = now
        inst_fps = 1.0 / max(dt, 1e-6)
        fps = fps_alpha * fps + (1 - fps_alpha) * inst_fps if fps > 0 else inst_fps

        canvas = render_frame(canvas, frame, smoothed, class_names,
                              committed_label, stability_count, fps)
        cv2.imshow(window_name, canvas)

        key = cv2.waitKey(1) & 0xFF
        if key == ord("q"):
            break
        elif key == ord("f"):
            is_fullscreen = not is_fullscreen
            cv2.setWindowProperty(
                window_name, cv2.WND_PROP_FULLSCREEN,
                cv2.WINDOW_FULLSCREEN if is_fullscreen else cv2.WINDOW_NORMAL,
            )

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()