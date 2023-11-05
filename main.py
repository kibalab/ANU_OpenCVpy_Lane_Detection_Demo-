import cv2
import numpy as np
from sklearn.cluster import KMeans

# 동영상 파일을 읽어옵니다. (카메라 사용 시 0 대신 카메라 번호 사용)
cap = cv2.VideoCapture('sample.mp4')

prev_frames_lines = []

while(cap.isOpened()):
    ret, frame = cap.read()

    if not ret:
        break

    # 영상 크기를 조정합니다.
    resized_frame = cv2.resize(frame, (1920, 1080))

    roi = resized_frame[int(1080/2):1000, 0:1920]

    # HSV 색 공간으로 변환합니다.
    hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)

    # 흰색의 HSV 범위를 정의합니다.
    lower_white = np.array([0, 0, 150])
    upper_white = np.array([255, 255, 255])

    # 영역을 추출합니다.
    white_mask = cv2.inRange(hsv, lower_white, upper_white)

    # 추출된 흰색 영역과 원본 이미지를 비트와이즈 AND 연산합니다.
    white_lane = cv2.bitwise_or(roi, roi, mask=white_mask)

    # 가우시안 블러로 이미지를 부드럽게 만듭니다.
    blurred = cv2.GaussianBlur(white_lane, (3, 3), 3)

    # Canny 엣지 검출을 수행합니다.
    edges = cv2.Canny(blurred, 50, 150)

    # 허프 변환을 사용하여 직선을 검출합니다.
    lines = cv2.HoughLinesP(edges, 1, np.pi / 180, 100, minLineLength=300, maxLineGap=10)

    # 검출된 직선을 원본 영상에 그립니다.
    if lines is not None:
        merged_lines = []
        for line in lines:
            x1, y1, x2, y2 = line[0]
            cv2.line(roi, (x1, y1), (x2, y2), (100, 100, 100), 10)
            x1, y1, x2, y2 = line[0]
            # 각 선의 기울기를 계산합니다
            slope = (y2 - y1) / (x2 - x1)
            # 기울기가 45도 이상인 선만 그립니다
            if abs(np.degrees(np.arctan(slope))) >= 35:
                if not merged_lines:
                    merged_lines.append([x1, y1, x2, y2])
                else:
                    merged = False
                    # 다른 선과의 거리를 계산 및 각도 차이 확인
                    for merged_line in merged_lines:
                        x3, y3, x4, y4 = merged_line
                        distance = np.sqrt((x1 - x3) ** 2 + (y1 - y3) ** 2)
                        angle1 = np.degrees(np.arctan(slope))
                        angle2 = np.degrees(np.arctan((y4 - y3) / (x4 - x3)))
                        angle_diff = abs(angle1 - angle2)
                        if distance < 10:
                            # 거리가 20 미만이면서 각도 차이가 5도 미만이면 두 선을 병합
                            x1 = min(x1, x3)
                            y1 = min(y1, y3)
                            x2 = max(x2, x4)
                            y2 = max(y2, y4)
                            merged_lines.remove(merged_line)
                            merged = True
                    if not merged:
                        merged_lines.append([x1, y1, x2, y2])

        if(len(prev_frames_lines) > 20):
            prev_frames_lines.pop(0)
        prev_frames_lines.append(merged_lines)

        for merged_line in merged_lines:
            x1, y1, x2, y2 = merged_line
            cv2.line(roi, (x1, y1), (x2, y2), (255, 255, 0), 5)

    # 화면 크기를 얻어옵니다.
    frame_height, frame_width, _ = resized_frame.shape

    # K-Means 클러스터링을 위해 선분의 중점 좌표와 각도를 추출합니다.
    line_data = []
    for merged_lines in prev_frames_lines:
        for line in merged_lines:
            x1, y1, x2, y2 = line
            # 선분의 중점 좌표 계산
            center_x = (x1 + x2) / 2
            center_y = (y1 + y2) / 2
            # 선분의 각도 계산
            angle = np.degrees(np.arctan2(y2 - y1, x2 - x1))
            line_data.append([center_x, center_y, angle])

    # K-Means 클러스터링을 사용하여 선분을 2개 그룹으로 나눕니다.
    kmeans = KMeans(n_clusters=2, random_state=1, n_init=10).fit(line_data)

    # 각 그룹의 중점과 각도를 계산합니다.
    group_centers = kmeans.cluster_centers_

    group_labels = kmeans.labels_

    # 각 그룹의 선분을 추출하여 각도를 계산합니다.
    group_0_lines = [line_data[i] for i in range(len(line_data)) if group_labels[i] == 0]
    group_1_lines = [line_data[i] for i in range(len(line_data)) if group_labels[i] == 1]
    group_angles = [np.mean([line[2] for line in group_0_lines]), np.mean([line[2] for line in group_1_lines])]


    # 화면에 2개의 선분 그리기
    for i in range(2):
        center_x, center_y, _ = group_centers[i]
        angle = group_angles[i]

        # 각도를 라디안으로 변환
        angle_rad = np.radians(angle)

        # 선분의 길이 (여기에서는 화면 가로 길이의 70%)
        line_length = int(1 * frame_width)

        # 선분의 끝점 계산
        x1 = int(center_x - 0.5 * line_length * np.cos(angle_rad))
        y1 = int(center_y - 0.5 * line_length * np.sin(angle_rad))
        x2 = int(center_x + 0.5 * line_length * np.cos(angle_rad))
        y2 = int(center_y + 0.5 * line_length * np.sin(angle_rad))

        # 화면에 선분 그리기
        cv2.line(roi, (x1, y1), (x2, y2), (0, 255, 0), 3)
        cv2.putText(roi, f"Center", (int(1920/2), int(center_y)), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
        cv2.putText(roi, f"Distance: {1920/2 - center_x : .2f}", (int(center_x), int(center_y)), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)

    # 결과를 화면에 표시합니다.
    cv2.imshow("White Lane Detection", resized_frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()


