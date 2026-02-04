# ai-vision-cube-control
Gesture-Controlled 3D Cube: A Real-time Interaction via MediaPipe & OpenCV


Key Features

1. Pinch-to-Grab Control
   The 3D cube is anchored and manipulated via a pinch gesture, performed by bringing the thumb and index finger together.

3. Intuitive 3D Manipulation
   The 3D cube rotates along the X and Y axes based on swipe gestures.


Tech Stack & Principles

1. MediaPipe Hands
   Digitizes hand movements by capturing the 3D coordinates of 21 key joints in real-time.

3. Rotation Matrix
   Maps the change in hand position to angular displacement, applying rotation matrices to update the cube's orientation in 3D space.

5. OpenCV Rendering
   Renders the computed 3D cube points onto the 2D image plane in real-time.

                      
Installation
1. OS: macOS / Windows / Linux
   
2. Python: 3.8 +
   
3. in Terminal (pip3 install opencv-python mediapipe numpy)
   
4. (python3 main.py)

   :)

