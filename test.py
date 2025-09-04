import cv2
import mediapipe as mp
import keyboard

mp_pose = mp.solutions.pose
pose = mp_pose.Pose()
mp_draw = mp.solutions.drawing_utils

cap = cv2.VideoCapture(0)

# Save neutral (initial) nose depth for reference
neutral_depth = None

while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame = cv2.flip(frame, 1)
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = pose.process(rgb)

    if results.pose_landmarks:
        mp_draw.draw_landmarks(frame, results.pose_landmarks, mp_pose.POSE_CONNECTIONS)

        landmarks = results.pose_landmarks.landmark
        nose = landmarks[mp_pose.PoseLandmark.NOSE]
        left_shoulder = landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER]
        right_shoulder = landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER]
        right_wrist = landmarks[mp_pose.PoseLandmark.RIGHT_WRIST]

        # --- Calibration: take first frame as neutral depth
        if neutral_depth is None:
            neutral_depth = nose.z  

        # Forward detection (if nose closer than neutral)
        forward = nose.z < neutral_depth - 0.15  # threshold adjust

        # Lean detection (only matters when forward = True)
        mid_shoulder_x = (left_shoulder.x + right_shoulder.x) / 2
        lean_value = nose.x - mid_shoulder_x

        # Release all movement keys first
        keyboard.release("a")
        keyboard.release("d")
        keyboard.release("w")

        if forward:
            if lean_value > 0.07:   # Lean right
                print("FORWARD + RIGHT → W + D")
                keyboard.press("w")
                keyboard.press("d")
            elif lean_value < -0.07:  # Lean left
                print("FORWARD + LEFT → W + A")
                keyboard.press("w")
                keyboard.press("a")
            else:  # Just forward
                print("FORWARD → W")
                keyboard.press("w")
        else:
            print("NEUTRAL → STOP")

        # Nitro detection
        if right_wrist.y < nose.y:
            print("🚀 NITRO → SPACE")
            keyboard.press_and_release("space")

    cv2.imshow("Asphalt Controller", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
