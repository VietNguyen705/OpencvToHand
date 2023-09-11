import cv2
import mediapipe as mp
import numpy as np
import time
import autopy
import pyautogui

# Prepare Drawing Utilities
mp_drawing = mp.solutions.drawing_utils
mp_hands = mp.solutions.hands

# For webcam input:
hands = mp_hands.Hands(static_image_mode=False,
                       max_num_hands=2,
                       min_detection_confidence=0.5)

cap = cv2.VideoCapture(0,cv2.CAP_DSHOW)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 720)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 1280)
cap.set(cv2.CAP_PROP_FPS, 30)  # Attempt to set the frame rate to 60 FPS
initial_landmark = None
last_tab_press_time = None
alt_pressed = False  # Add a new line to initialize the alt pressed flag

# Define region of interest (ROI) in the frame
roi_top = 0.001  # Top boundary of ROI (25% from top)
roi_bottom = 0.7  # Bottom boundary of ROI (25% from bottom)
roi_left = 0.001  # Left boundary of ROI (25% from left)
roi_right = 0.5  # Right boundary of ROI (25% from right)
flick_threshold = 0.1  # Adjust this value as needed
smoothening = 2
prev_x = None
prev_y = None
scroll_amount=0
prev_mouse_x = None
prev_mouse_y = None
frame_counter = 0
while cap.isOpened():
    frame_counter += 1
    if frame_counter % 3 != 0:
        continue
    success, image = cap.read()
    if success:
        # Crop the frame to the central half
        height, width, _ = image.shape
        top = 0
        left = int(width * 0.25)
        bottom = int(height)
        right = int(width * 0.75)
        image = image[top:bottom, left:right]
    if not success:
        print("Ignoring empty camera frame.")
        continue
    # Get frame's width and height
    height, width, _ = image.shape

    # Define the center and radius of the circle
    center_x, center_y = width // 2, height // 2
    radius = height // 3

    # Flip the image horizontally for a later selfie-view display, and convert
    # the BGR image to RGB.
    image = cv2.cvtColor(cv2.flip(image, 1), cv2.COLOR_BGR2RGB)
    # To improve performance, optionally mark the image as not writeable to
    # pass by reference.
    image.flags.writeable = False
    results = hands.process(image)

    # Draw the hand annotations on the image.
    image.flags.writeable = True
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    if results.multi_hand_landmarks:
        for hand_landmarks, handedness in zip(results.multi_hand_landmarks, results.multi_handedness):
            if results.multi_handedness[0].classification[0].label == 'Left':
                index_tip = np.array([hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP].x, 
                                  hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP].y])
                # Get thumb tip position
                thumb_tip = np.array([hand_landmarks.landmark[mp_hands.HandLandmark.THUMB_TIP].x, 
                          hand_landmarks.landmark[mp_hands.HandLandmark.THUMB_TIP].y])
                # Get middle finger's middle joint position
                index_finger_mid = np.array([hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_MCP].x, 
                                  hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_MCP].y])

                # Scale the x and y coordinates of the middle finger's middle joint to the frame size
                index_finger_mid_scaled = np.array([index_finger_mid[0] * width, index_finger_mid[1] * height])

                # Scale the x and y coordinates of the thumb tip to the frame size
                thumb_tip_scaled = np.array([thumb_tip[0] * width, thumb_tip[1] * height])
                # Scale the x and y coordinates of the index tip to the frame size
                index_tip_scaled = np.array([index_tip[0] * width, index_tip[1] * height])
                # Smoothing factor
            
                mouse_x=None
                mouse_y=None
               # Check if the index tip is inside the ROI
                if roi_left * width < index_tip_scaled[0] < roi_right * width and roi_top * height < index_tip_scaled[1] < roi_bottom * height:
                   # Map the index tip position in the ROI to the screen resolution
                    screen_width, screen_height = autopy.screen.size()
                    mouse_x = ((index_tip[0] - roi_left) / (roi_right - roi_left)) * screen_width
                    mouse_y = ((index_tip[1] - roi_top) / (roi_bottom - roi_top)) * screen_height
                    # If this is the first frame, initialize the previous mouse coordinates
                    if prev_mouse_x is None and prev_mouse_y is None:
                        prev_mouse_x = mouse_x
                        prev_mouse_y = mouse_y
                    click_state = False  # Add a new line to initialize the click state flag
                    # Define the click action
                    click_distance_threshold = 15  # You can adjust this value as needed
                    if np.linalg.norm(thumb_tip_scaled - index_finger_mid_scaled) < click_distance_threshold:
                        if not click_state:  # Only click if the click state flag is False
                            autopy.mouse.click()
                            click_state = True
                    else:
                        click_state = False
            

                # Update the previous x and y coordinates for the right hand
                prev_x = hand_landmarks.landmark[mp_hands.HandLandmark.WRIST].x
                prev_y = hand_landmarks.landmark[mp_hands.HandLandmark.WRIST].y


                # Move the mouse cursor
                if mouse_x is not None and mouse_y is not None:
                    # Calculate the new mouse coordinates as a weighted average
                    mouse_x = prev_mouse_x + (mouse_x - prev_mouse_x) / smoothening
                    mouse_y = prev_mouse_y + (mouse_y - prev_mouse_y) / smoothening
                    # Update the previous mouse coordinates
                    prev_mouse_x = mouse_x
                    prev_mouse_y = mouse_y
                        # Move the mouse cursor to the smoothed coordinates
                    autopy.mouse.move(mouse_x, mouse_y)
            elif results.multi_handedness[0].classification[0].label == 'Right':
                # Define the center point of the frame
                frame_center = width // 2, height // 2
                # Scroll based on the distance from the center of the frame to the index finger tip
                index_finger_tip = hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP]
                index_finger_tip_y = index_finger_tip.y * height  # Convert to pixel coordinate
                index_finger_tip_x = index_finger_tip.x * width
                # Define a sensitivity for the scroll (this could be adjusted to your liking)
                scroll_sensitivity = -0.8

                # Calculate the distance from the center of the frame to the index finger tip
                distance = index_finger_tip_y - frame_center[1]
                if not alt_pressed and frame_center[0] - index_finger_tip_x > 2.0:
                    pyautogui.keyDown('alt')
                    alt_pressed = True
                    last_tab_press_time = None  # Reset the time of the last 'tab' key press
                elif alt_pressed and frame_center[0] - index_finger_tip_x < 2.0:
                    pyautogui.keyUp('alt')
                    alt_pressed = False

                if alt_pressed:
                    current_time = time.time()  # Get the current time
                    if last_tab_press_time is None or current_time - last_tab_press_time > 0.5:
                        pyautogui.press('tab')
                        last_tab_press_time = current_time  # Update the time of the last 'tab' key press
                    

                # Scroll amount is the distance multiplied by the sensitivity
                scroll_amount = distance * scroll_sensitivity
                
                # print('Scroll amount: ', scroll_amount)
                # Determine the scroll direction
                pyautogui.scroll(int(scroll_amount))

               
            
        
        # mp_drawing.draw_landmarks(image, hand_landmarks, mp_hands.HAND_CONNECTIONS)

    cv2.rectangle(image, (int(roi_left * width), int(roi_top * height)), (int(roi_right * width), int(roi_bottom * height)), (0, 255, 0), 2)
    cv2.imshow('MediaPipe Hands', image)
    if cv2.waitKey(10) & 0xFF == ord('q'):
        break

hands.close()
cap.release()
cv2.destroyAllWindows()