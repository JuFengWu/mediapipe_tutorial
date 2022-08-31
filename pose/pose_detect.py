import cv2
import mediapipe as mp
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles
mp_pose = mp.solutions.pose


cap = cv2.VideoCapture('test.mp4')

with mp_pose.Pose( min_detection_confidence=0.5,min_tracking_confidence=0.5) as pose:

  while cv2.waitKey(1)!=27:
    ret, frame = cap.read()
    frame = cv2.resize(frame, (540, 960), interpolation=cv2.INTER_AREA)
    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = pose.process(frame)
    frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
    mp_drawing.draw_landmarks(
        frame,
        results.pose_landmarks,
        mp_pose.POSE_CONNECTIONS,
        landmark_drawing_spec=mp_drawing_styles.get_default_pose_landmarks_style())
    cv2.imshow('frame', frame)
    cv2.waitKey(5)