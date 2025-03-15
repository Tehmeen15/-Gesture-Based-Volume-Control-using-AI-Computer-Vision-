import cv2
import mediapipe as mp
import numpy as np
from math import hypot
from ctypes import cast, POINTER
from comtypes import CLSCTX_ALL
from pycaw.pycaw import AudioUtilities, IAudioEndpointVolume

# Initialize MediaPipe Hand Detection
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(static_image_mode=False, max_num_hands=1, min_detection_confidence=0.7)
mp_draw = mp.solutions.drawing_utils

try:
    # Initialize Audio Control
    devices = AudioUtilities.GetSpeakers()
    interface = devices.Activate(IAudioEndpointVolume._iid_, CLSCTX_ALL, None)
    volume = cast(interface, POINTER(IAudioEndpointVolume))

    # Get volume range
    vol_range = volume.GetVolumeRange()
    min_vol, max_vol = vol_range[0], vol_range[1]
    print(f"Volume Range: {min_vol} to {max_vol}")
except Exception as e:
    print(f"Error initializing audio: {e}")
    exit(1)

# Initialize webcam
cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

if not cap.isOpened():
    print("Error: Could not open webcam")
    exit(1)

# Variables for smooth volume control
smoothing_factor = 0.5
previous_vol = 0

while True:
    # Read video frame
    success, img = cap.read()
    if not success:
        print("Failed to capture video. Check if webcam is connected.")
        break

    # Convert to RGB
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    
    # Process hand detection
    results = hands.process(img_rgb)
    
    # Lists for storing landmark positions
    lm_list = []
    
    if results.multi_hand_landmarks:
        for hand_landmark in results.multi_hand_landmarks:
            # Draw landmarks
            mp_draw.draw_landmarks(img, hand_landmark, mp_hands.HAND_CONNECTIONS)
            
            # Get landmark positions
            for id, landmark in enumerate(hand_landmark.landmark):
                h, w, c = img.shape
                cx, cy = int(landmark.x * w), int(landmark.y * h)
                lm_list.append([id, cx, cy])
        
        if len(lm_list) >= 9:  # Check if thumb and index finger landmarks are detected
            # Get thumb and index finger positions
            thumb_x, thumb_y = lm_list[4][1], lm_list[4][2]
            index_x, index_y = lm_list[8][1], lm_list[8][2]
            
            # Draw circles on thumb and index finger
            cv2.circle(img, (thumb_x, thumb_y), 15, (255, 0, 0), cv2.FILLED)
            cv2.circle(img, (index_x, index_y), 15, (255, 0, 0), cv2.FILLED)
            cv2.line(img, (thumb_x, thumb_y), (index_x, index_y), (255, 0, 0), 3)
            
            # Calculate distance between fingers
            length = hypot(thumb_x - index_x, thumb_y - index_y)
            
            # Hand range: 50 - 300 (adjusted for better control)
            # Volume range: min_vol to max_vol
            vol = np.interp(length, [50, 300], [min_vol, max_vol])
            
            # Smooth the volume changes
            vol = previous_vol * smoothing_factor + vol * (1 - smoothing_factor)
            previous_vol = vol
            
            try:
                # Set system volume
                volume.SetMasterVolumeLevel(vol, None)
                
                # Calculate volume percentage for display
                vol_percentage = np.interp(vol, [min_vol, max_vol], [0, 100])
                
                # Draw volume bar
                bar_height = np.interp(vol_percentage, [0, 100], [400, 150])
                
                cv2.rectangle(img, (50, 150), (85, 400), (0, 255, 0), 3)
                cv2.rectangle(img, (50, int(bar_height)), (85, 400), (0, 255, 0), cv2.FILLED)
                cv2.putText(img, f'{int(vol_percentage)}%', (40, 450), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 3)
                
                # Display distance for debugging
                cv2.putText(img, f'Distance: {int(length)}', (40, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)
            except Exception as e:
                print(f"Error setting volume: {e}")

    # Display image
    cv2.imshow('Hand Gesture Volume Control', img)
    
    # Exit on 'q' press
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Clean up
cap.release()
cv2.destroyAllWindows()
hands.close() 