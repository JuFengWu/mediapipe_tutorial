import cv2
import mediapipe as mp
from mpl_toolkits.mplot3d import Axes3D
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles
mp_hands = mp.solutions.hands

IMAGE_FILES = ['/home/leowu/program/handout/mediapipe/hand_detection/hand_1.png']

with mp_hands.Hands(
    static_image_mode=True,
    max_num_hands=2,
    min_detection_confidence=0.5) as hands:
  for idx, file in enumerate(IMAGE_FILES):
    # Read an image, flip it around y-axis for correct handedness output (see
    # above).
    image = cv2.flip(cv2.imread(file), 1)
    # Convert the BGR image to RGB before processing.
    results = hands.process(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))

    # Print handedness and draw hand landmarks on the image.
    print('Handedness:', results.multi_handedness)
    if not results.multi_hand_landmarks:
      continue
    image_height, image_width, _ = image.shape
    annotated_image = image.copy()
    i = 0
    for hand_landmarks in results.multi_hand_landmarks:
      print('hand_landmarks:', hand_landmarks)
      print(
          f'Index finger tip coordinates: (',
          f'{hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP].x * image_width}, '
          f'{hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP].y * image_height})'
      )
      mp_drawing.draw_landmarks(
          annotated_image,
          hand_landmarks,
          mp_hands.HAND_CONNECTIONS,
          mp_drawing_styles.get_default_hand_landmarks_style(),
          mp_drawing_styles.get_default_hand_connections_style())
      singlePointImage = image.copy()
      showPoint = 0
      xPoint = hand_landmarks.landmark[showPoint].x * image_width
      yPoint = hand_landmarks.landmark[showPoint].y * image_height
      print(len(hand_landmarks.landmark))
      
      print("hand is "+ results.multi_handedness[i].classification[0].label)
      singlePointImage = cv2.circle(singlePointImage, (int(xPoint),int(yPoint)), 5, (255,0,0), -1)
      cv2.imshow('image',singlePointImage)
      key = cv2.waitKey(0)
      i = i+1
      
    cv2.imwrite(
        'detect_hand_' + str(idx) + '.png', cv2.flip(annotated_image, 1))
    # Draw hand world landmarks.
    if not results.multi_hand_world_landmarks:
      continue
    #for hand_world_landmarks in results.multi_hand_world_landmarks:
    #  mp_drawing.plot_landmarks(hand_world_landmarks, mp_hands.HAND_CONNECTIONS, azimuth=5)
