import argparse
import time
from collections import deque

import cv2
import mediapipe as mp
import keyboard


def parse_args():
    parser = argparse.ArgumentParser(description="Webcam body controls for steering and nitro")
    parser.add_argument("--camera", type=int, default=0, help="Camera index (default: 0)")
    parser.add_argument("--fps", type=float, default=30.0, help="Target FPS cap (default: 30)")
    parser.add_argument("--alpha", type=float, default=0.3, help="EMA smoothing factor (0-1, higher = smoother)")
    parser.add_argument("--min-visibility", type=float, default=0.5, help="Min landmark visibility to trust (default: 0.5)")
    parser.add_argument("--calibration-seconds", type=float, default=3.0, help="Calibration duration in seconds (default: 3)")
    parser.add_argument("--turn-threshold", type=float, default=None, help="Override steering threshold (normalized Y diff)")
    parser.add_argument("--nitro-offset", type=float, default=None, help="Override nitro offset below nose (normalized)")
    parser.add_argument("--mirror-controls", action="store_true", help="Swap left/right steering interpretation")
    parser.add_argument("--no-draw", action="store_true", help="Do not draw landmarks (privacy/perf)")
    parser.add_argument("--nitro-hold-frames", type=int, default=3, help="Frames required before nitro engages (debounce)")
    return parser.parse_args()


class KeyController:
    def __init__(self):
        self.pressed = set()

    def press(self, key: str):
        if key in self.pressed:
            return
        try:
            keyboard.press(key)
            self.pressed.add(key)
        except Exception:
            pass

    def release(self, key: str):
        if key not in self.pressed:
            return
        try:
            keyboard.release(key)
        except Exception:
            pass
        finally:
            self.pressed.discard(key)

    def set_active(self, keys_active):
        for k in list(self.pressed):
            if k not in keys_active:
                self.release(k)
        for k in keys_active:
            self.press(k)


def ema_update(prev_value, new_value, alpha):
    if prev_value is None:
        return new_value
    return alpha * new_value + (1 - alpha) * prev_value


def compute_thresholds(calib_samples):
    if not calib_samples:
        return 0.05, 0.02
    diffs = [s["diff"] for s in calib_samples]
    lefts = [s["left_y"] for s in calib_samples]
    rights = [s["right_y"] for s in calib_samples]

    def robust_std(values):
        values_sorted = sorted(values)
        median = values_sorted[len(values_sorted)//2]
        mad_list = sorted([abs(v - median) for v in values])
        mad = mad_list[len(mad_list)//2]
        return 1.4826 * mad if mad > 1e-6 else 0.0

    diff_std = robust_std(diffs)
    turn_threshold = max(0.03, 2.5 * diff_std)

    wrists = lefts + rights
    wrist_std = robust_std(wrists)
    nitro_offset = max(0.015, 0.5 * wrist_std)
    return turn_threshold, nitro_offset


def main():
    args = parse_args()

    mp_pose = mp.solutions.pose
    mp_draw = mp.solutions.drawing_utils
    pose = mp_pose.Pose()

    cap = cv2.VideoCapture(args.camera)
    if not cap.isOpened():
        print("Error: Unable to open camera.")
        return

    target_dt = 1.0 / max(1.0, args.fps)
    last_frame_time = 0.0

    left_y_smooth = None
    right_y_smooth = None
    nose_y_smooth = None

    calib_samples = []
    calib_end_time = time.time() + max(0.0, args.calibration_seconds)
    calibrated = False
    turn_threshold = args.turn_threshold
    nitro_offset = args.nitro_offset

    keys = KeyController()
    nitro_queue = deque(maxlen=max(1, args.nitro_hold_frames))

    window_name = "Steering Wheel Control"

    try:
        while True:
            now = time.time()
            if now - last_frame_time < target_dt:
                time.sleep(max(0.0, target_dt - (now - last_frame_time)))
            last_frame_time = time.time()

            ret, frame = cap.read()
            if not ret:
                print("Warning: Camera frame not received.")
                break

            frame = cv2.flip(frame, 1)
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = pose.process(rgb)

            action_text = "IDLE"
            active_keys = set()

            if results.pose_landmarks:
                if not args.no_draw:
                    mp_draw.draw_landmarks(frame, results.pose_landmarks, mp_pose.POSE_CONNECTIONS)

                landmarks = results.pose_landmarks.landmark
                left_wrist = landmarks[mp_pose.PoseLandmark.LEFT_WRIST]
                right_wrist = landmarks[mp_pose.PoseLandmark.RIGHT_WRIST]
                nose = landmarks[mp_pose.PoseLandmark.NOSE]

                if (left_wrist.visibility >= args.min_visibility and
                        right_wrist.visibility >= args.min_visibility and
                        nose.visibility >= args.min_visibility):

                    left_y_smooth = ema_update(left_y_smooth, left_wrist.y, args.alpha)
                    right_y_smooth = ema_update(right_y_smooth, right_wrist.y, args.alpha)
                    nose_y_smooth = ema_update(nose_y_smooth, nose.y, args.alpha)

                    cv2.putText(frame, f"Left Y: {left_y_smooth:.3f}", (10, 30),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)
                    cv2.putText(frame, f"Right Y: {right_y_smooth:.3f}", (10, 55),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)

                    diff = left_y_smooth - right_y_smooth
                    if args.mirror_controls:
                        diff = -diff

                    if not calibrated and time.time() <= calib_end_time:
                        calib_samples.append({
                            "left_y": float(left_y_smooth),
                            "right_y": float(right_y_smooth),
                            "nose_y": float(nose_y_smooth),
                            "diff": float(diff),
                        })
                        remaining = max(0.0, calib_end_time - time.time())
                        cv2.putText(frame, f"Calibrating... {remaining:.1f}s", (10, 85),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 215, 255), 2)
                    elif not calibrated:
                        auto_turn, auto_nitro = compute_thresholds(calib_samples)
                        if turn_threshold is None:
                            turn_threshold = auto_turn
                        if nitro_offset is None:
                            nitro_offset = auto_nitro
                        calibrated = True

                    if turn_threshold is None:
                        turn_threshold = 0.05
                    if nitro_offset is None:
                        nitro_offset = 0.02

                    cv2.putText(frame, f"Turn thr: {turn_threshold:.3f}", (10, 110),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (200, 200, 200), 2)
                    cv2.putText(frame, f"Nitro off: {nitro_offset:.3f}", (10, 135),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (200, 200, 200), 2)

                    if abs(diff) < turn_threshold:
                        action_text = "FORWARD"
                        active_keys.add("w")
                    elif diff > 0:
                        action_text = "RIGHT"
                        active_keys.add("d")
                    else:
                        action_text = "LEFT"
                        active_keys.add("a")

                    nitro_condition = (right_y_smooth < (nose_y_smooth - nitro_offset))
                    nitro_queue.append(1 if nitro_condition else 0)
                    if sum(nitro_queue) == nitro_queue.maxlen:
                        active_keys.add("space")
                        cv2.putText(frame, "NITRO", (50, 185),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 3)

            keys.set_active(active_keys)

            cv2.putText(frame, action_text, (50, 145),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 255), 3)

            cv2.imshow(window_name, frame)

            k = cv2.waitKey(1) & 0xFF
            if k == ord('q') or k == 27:
                break
    finally:
        for k in list(keys.pressed):
            keys.release(k)
        cap.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()


