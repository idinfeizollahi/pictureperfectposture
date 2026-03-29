import cv2
import mediapipe as mp
import numpy as np
import time

mp_drawing = mp.solutions.drawing_utils
mp_pose = mp.solutions.pose

bad_posture_start = None
bad_posture_alerted = False


def calculate_angle(a, b, c):
    a = np.array(a)  # First
    b = np.array(b)  # Mid
    c = np.array(c)  # End

    radians = np.arctan2(c[1] - b[1], c[0] - b[0]) - np.arctan2(a[1] - b[1], a[0] - b[0])
    angle = np.abs(radians * 180.0 / np.pi)

    if angle > 180.0:
        angle = 360 - angle

    return angle


# VIDEO FEED
cap = cv2.VideoCapture(0)

with mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5) as pose:
    while cap.isOpened():
        ret, frame = cap.read()

        # Recolor image to RGB
        image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        image.flags.writeable = False

        # Make detection
        results = pose.process(image)

        # Recolor back to BGR
        image.flags.writeable = True
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

        try:
            landmarks = results.pose_landmarks.landmark

            left_shoulder = [
                landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].x,
                landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].y
            ]
            right_shoulder = [
                landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value].x,
                landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value].y
            ]

            left_hip = [
                landmarks[mp_pose.PoseLandmark.LEFT_HIP.value].x,
                landmarks[mp_pose.PoseLandmark.LEFT_HIP.value].y
            ]
            right_hip = [
                landmarks[mp_pose.PoseLandmark.RIGHT_HIP.value].x,
                landmarks[mp_pose.PoseLandmark.RIGHT_HIP.value].y
            ]

            nose = [
                landmarks[mp_pose.PoseLandmark.NOSE.value].x,
                landmarks[mp_pose.PoseLandmark.NOSE.value].y
            ]

            shoulder_mid = [
                (landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].x +
                 landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value].x) / 2,
                (landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].y +
                 landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value].y) / 2
            ]

            hip_mid = [
                (landmarks[mp_pose.PoseLandmark.LEFT_HIP.value].x +
                 landmarks[mp_pose.PoseLandmark.RIGHT_HIP.value].x) / 2,
                (landmarks[mp_pose.PoseLandmark.LEFT_HIP.value].y +
                 landmarks[mp_pose.PoseLandmark.RIGHT_HIP.value].y) / 2
            ]

            vertical_ref = [shoulder_mid[0], shoulder_mid[1] - 0.1]
            angle = calculate_angle(vertical_ref, shoulder_mid, hip_mid)
            print(angle)

            # BAD POSTURE LOGIC
            if angle < 177:
                if bad_posture_start is None:
                    bad_posture_start = time.time()
                    bad_posture_alerted = False
                else:
                    elapsed = time.time() - bad_posture_start
                    if elapsed > 2:
                        bad_posture_alerted = True
            else:
                bad_posture_start = None
                bad_posture_alerted = False

        except Exception:
            pass

        # DRAW LANDMARKS
        if results.pose_landmarks:
            mp_drawing.draw_landmarks(
                image,
                results.pose_landmarks,
                mp_pose.POSE_CONNECTIONS,
                mp_drawing.DrawingSpec(color=(245, 117, 66), thickness=2, circle_radius=2),
                mp_drawing.DrawingSpec(color=(245, 66, 230), thickness=2, circle_radius=2)
            )

        # VISUAL WARNING
        if bad_posture_alerted:
            cv2.putText(
                image,
                "BAD POSTURE ALERT",
                (50, 100),
                cv2.FONT_HERSHEY_SIMPLEX,
                1,
                (0, 0, 255),
                3,
                cv2.LINE_AA
            )

        cv2.imshow("Mediapipe Feed", image)

        if cv2.waitKey(10) & 0xFF == ord('q'):
            break

cap.release()
cv2.destroyAllWindows()