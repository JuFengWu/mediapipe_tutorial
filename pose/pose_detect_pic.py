import cv2
import mediapipe as mp
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles
mp_pose = mp.solutions.pose
mp_holistic = mp.solutions.holistic

file = 'test_pose.png'
frame = cv2.imread(file)
with mp_pose.Pose( min_detection_confidence=0.5,min_tracking_confidence=0.5) as pose:
    
    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = pose.process(frame)
    frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
    print(results.pose_landmarks)
    mp_drawing.draw_landmarks(
        frame,
        results.pose_landmarks,
        mp_pose.POSE_CONNECTIONS,
        landmark_drawing_spec=mp_drawing_styles.get_default_pose_landmarks_style())
   
    cv2.imshow('frame', frame)
    cv2.waitKey(0)

    mp_drawing.plot_landmarks(
        results.pose_world_landmarks, mp_holistic.POSE_CONNECTIONS)