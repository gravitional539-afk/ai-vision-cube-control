import cv2
import mediapipe as mp
import numpy as np

# ===== 초기 설정: 손 인식 + 얼굴 인식 =====
mp_hands = mp.solutions.hands
mp_face = mp.solutions.face_detection 
hands = mp_hands.Hands(max_num_hands=1, min_detection_confidence=0.7)
face_detection = mp_face.FaceDetection(model_selection=0, min_detection_confidence=0.5)

# 🎥 카메라 초기화 (오류 처리 추가)
cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("❌ 오류: 카메라를 연결할 수 없습니다!")
    print("✅ 해결 방법:")
    print("   1. USB 카메라가 연결되어 있는지 확인하세요")
    print("   2. 다른 앱에서 카메라를 사용 중이면 종료하세요")
    print("   3. 시스템 설정 > 보안 > 카메라 권한을 확인하세요")
    exit()

# 📐 3D 큐브 데이터
points = np.array([
    [-1, -1, 1], [1, -1, 1], [1, 1, 1], [-1, 1, 1],
    [-1, -1, -1], [1, -1, -1], [1, 1, -1], [-1, 1, -1]
], dtype=float)
edges = [(0,1), (1,2), (2,3), (3,0), (4,5), (5,6), (6,7), (7,4), (0,4), (1,5), (2,6), (3,7)]

# 회전 각도 및 손 제어 변수
angle_x, angle_y = 0, 0
prev_pos = None

# ⚙️ 사용자 조정 가능한 설정
PINCH_THRESHOLD = 0.05  # 핀치 감도 (작을수록 민감함, 0.03~0.1 추천)
BLUR_STRENGTH = 55       # 블러 강도 (홀수만 가능: 21, 35, 55 등)
SENSITIVITY = 0.01       # 회전 감도 (작을수록 느림)

while cap.isOpened():
    success, image = cap.read()
    if not success: 
        print("⚠️ 경고: 카메라 프레임을 읽을 수 없습니다")
        break

    image = cv2.flip(image, 1)
    h, w, _ = image.shape
    rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    # ===== 얼굴 블러 처리 =====
    face_results = face_detection.process(rgb_image)
    if face_results.detections:
        for detection in face_results.detections:
            bbox = detection.location_data.relative_bounding_box
            x, y, fw, fh = int(bbox.xmin * w), int(bbox.ymin * h), int(bbox.width * w), int(bbox.height * h)
            
            # 👁️ 얼굴 영역 추출 (범위 검증 추가)
            y_start = max(0, y)
            y_end = min(h, y + fh)
            x_start = max(0, x)
            x_end = min(w, x + fw)
            
            face_roi = image[y_start:y_end, x_start:x_end]
            if face_roi.size > 0:
                # 블러 강도 조절 (BLUR_STRENGTH 숫자를 키우면 더 뿌예집니다)
                blurred_face = cv2.GaussianBlur(face_roi, (BLUR_STRENGTH, BLUR_STRENGTH), 0)
                image[y_start:y_end, x_start:x_end] = blurred_face

    # ===== 손 인식 및 큐브 제어 =====
    hand_results = hands.process(rgb_image)
    if hand_results.multi_hand_landmarks:
        for hand_landmarks in hand_results.multi_hand_landmarks:
            t = hand_landmarks.landmark[4]  # 엄지
            i = hand_landmarks.landmark[8]  # 검지
            
            # 손가락 거리 계산 (값 검증 추가)
            dist = np.linalg.norm(np.array([t.x - i.x, t.y - i.y]))
            curr_pos = np.array([t.x * w, t.y * h])

            # 🎯 핀치 제스처 감지 (PINCH_THRESHOLD로 조절 가능)
            if dist < PINCH_THRESHOLD:
                if prev_pos is not None:
                    dx = curr_pos[0] - prev_pos[0]
                    dy = curr_pos[1] - prev_pos[1]
                    angle_y += dx * SENSITIVITY
                    angle_x -= dy * SENSITIVITY  # 상하 쓸어올리기로 조절
                prev_pos = curr_pos
                cv2.circle(image, (int(curr_pos[0]), int(curr_pos[1])), 10, (0, 255, 0), -1)
            else:
                prev_pos = None

    # ===== 3D 회전 행렬 계산 =====
    rx = np.array([[1, 0, 0], 
                   [0, np.cos(angle_x), -np.sin(angle_x)], 
                   [0, np.sin(angle_x), np.cos(angle_x)]])
    ry = np.array([[np.cos(angle_y), 0, np.sin(angle_y)], 
                   [0, 1, 0], 
                   [-np.sin(angle_y), 0, np.cos(angle_y)]])
    
    # ===== 3D 투영 및 큐브 그리기 =====
    projected_points = []
    for p in points:
        rotated = ry @ (rx @ p)
        
        # ✅ 제로 나눗셈 방지 (z값 범위 검증)
        z_denominator = 4 - rotated[2]
        if z_denominator <= 0.1:  # 너무 가까우면 스킵
            z = 1
        else:
            z = 1 / z_denominator
        
        px = int(rotated[0] * z * 600 + w/2)
        py = int(rotated[1] * z * 600 + h/2)
        
        # 📍 화면 범위 검증 (화면 밖 좌표 처리)
        px = np.clip(px, 0, w - 1)
        py = np.clip(py, 0, h - 1)
        projected_points.append((px, py))

    # 큐브 엣지 그리기
    for edge in edges:
        p1 = projected_points[edge[0]]
        p2 = projected_points[edge[1]]
        # 범위 검증된 좌표로만 그리기
        if 0 <= p1[0] < w and 0 <= p1[1] < h and 0 <= p2[0] < w and 0 <= p2[1] < h:
            cv2.line(image, p1, p2, (0, 255, 255), 2)

    # 📺 화면 표시
    cv2.putText(image, f"Pinch to control | Press 'q' to quit", (10, 30), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
    cv2.imshow('Face Blur + 3D Cube Control', image)
    
    if cv2.waitKey(1) & 0xFF == ord('q'): 
        break

print("✅ 프로그램 종료됨")
cap.release()
cv2.destroyAllWindows()