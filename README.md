# ai-vision-cube-control
MediaPipe와 OpenCV를 이용해 사용자의 손동작으로 3D 큐브를 실시간으로 제어하는 프로젝트

Key Features
1. Pinch-to-Grab Control 엄지와 검지를 맞대는 핀치 제스처로 3D 큐브를 고정하고 조작합니다

2. Intuitive 3D Manipulation 스와이프시 큐브의 X,Y축 회전


Tech Stack & Principles
1. MediaPipe Hands: 손가락 마디(Landmarks) 21개를 실시간 트래킹하여 좌표화합니다

2. Rotation Matrix (Mathematics): 사용자의 손 움직임($\Delta x, \Delta y$)을 각도로 변환하여 3차원 회전 행렬 연산을 수행합니다

3. OpenCV Rendering: 수학적으로 계산된 3D 좌표를 실시간 영상 위에 와이어프레임(Wireframe)으로 그려냅니다.

                      
Installation
1. OS: macOS / Windows / Linux
2. Python: 3.8 +
3. in Terminal (pip3 install opencv-python mediapipe numpy)
4. python3 main.py

