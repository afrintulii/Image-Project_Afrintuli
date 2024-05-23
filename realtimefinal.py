import numpy as np
import cv2

# Initialize variables
background = None
frames_elapsed = 0
FRAME_HEIGHT = 600
FRAME_WIDTH = 600
CALIBRATION_TIME = 30
BG_WEIGHT = 0.5
OBJ_THRESHOLD = 18
startDistance = None
handshake_threshold = 5
distance_buffer = []
BUFFER_SIZE = 10

# Region of interest coordinates
region_top = 0
region_bottom = int(2 * FRAME_HEIGHT / 3)
region_left = int(FRAME_WIDTH / 2)
region_right = FRAME_WIDTH
region_width = region_right - region_left

class HandData:
    def __init__(self):
        self.top_points = []

    def update(self, top_points):
        self.top_points = top_points

def write_on_image(frame, hand):
    if hand and hand.top_points:
        for i, (x, y) in enumerate(hand.top_points):
            x += region_left
            y += region_top
            cv2.circle(frame, (x, y), 5, (0, 255, 0), -1)
            cv2.putText(frame, str(i), (x, y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)

        for i in range(len(hand.top_points) - 1):
            x1, y1 = hand.top_points[i]
            x2, y2 = hand.top_points[i + 1]
            x1 += region_left
            y1 += region_top
            x2 += region_left
            y2 += region_top
            cv2.line(frame, (x1, y1), (x2, y2), (0, 0, 255), 2)

    cv2.rectangle(frame, (region_left, region_top), (region_right, region_bottom), (255, 255, 255), 2)

def get_region(frame):
    region = frame[region_top:region_bottom, region_left:region_right]
    region = cv2.cvtColor(region, cv2.COLOR_BGR2GRAY)
    region = cv2.GaussianBlur(region, (5, 5), 0)
    return region

def get_average(region):
    global background
    if background is None:
        background = region.copy().astype("float")
        return
    cv2.accumulateWeighted(region, background, BG_WEIGHT)

def segment(region):
    global hand
    diff = cv2.absdiff(background.astype(np.uint8), region)
    thresholded_region = cv2.threshold(diff, OBJ_THRESHOLD, 255, cv2.THRESH_BINARY)[1]
    
    # Morphological operations to remove noise and fill gaps
    kernel = np.ones((3, 3), np.uint8)
    thresholded_region = cv2.morphologyEx(thresholded_region, cv2.MORPH_CLOSE, kernel, iterations=2)
    thresholded_region = cv2.morphologyEx(thresholded_region, cv2.MORPH_OPEN, kernel, iterations=2)
    
    contours, _ = cv2.findContours(thresholded_region.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if len(contours) == 0:
        return None
    else:
        max_contour = max(contours, key=cv2.contourArea)
        return max_contour

def get_hand_data(segmented_image, hand):
    convexHull = cv2.convexHull(segmented_image)
    hand.top_points = []

    finger_tips = {}
    for contour_point in convexHull:
        x, y = contour_point[0]
        finger_index = x // (region_width // 5)
        if finger_index not in finger_tips:
            finger_tips[finger_index] = []
        finger_tips[finger_index].append((x, y))

    for points in finger_tips.values():
        topmost_point = min(points, key=lambda point: point[1])
        hand.top_points.append(topmost_point)

def main():
    global frames_elapsed, startDistance, distance_buffer
    capture = cv2.VideoCapture(0)
    hand = HandData()

    # Load and resize the initial image to be displayed
    img = cv2.imread("test.jpg")
    img = cv2.resize(img, (150, 150))
    img_orig = img.copy()  # Store the original image

    while True:
        ret, frame = capture.read()
        if not ret:
            break

        frame = cv2.resize(frame, (FRAME_WIDTH, FRAME_HEIGHT))
        frame = cv2.flip(frame, 1)
        
        # Display the initial image on the top-left corner of the frame
        img_h, img_w = img.shape[:2]
        frame[0:img_h, 0:img_w] = img

        region = get_region(frame)
        if frames_elapsed < CALIBRATION_TIME:
            get_average(region)
        else:
            hand_contour = segment(region)
            if hand_contour is not None:
                cv2.drawContours(region, [hand_contour], -1, (255, 255, 255))
                cv2.imshow("Segmented Image", region)
                get_hand_data(hand_contour, hand)

        write_on_image(frame, hand)

        # Find the tips of thumb and index finger
        if len(hand.top_points) >= 2:
            thumb_tip = min(hand.top_points, key=lambda p: p[0])  # Leftmost point
            index_tip = min([p for p in hand.top_points if p != thumb_tip], key=lambda p: p[0])  # Second leftmost point

            d_thumb_index = int(np.sqrt((index_tip[0] - thumb_tip[0]) ** 2 + (index_tip[1] - thumb_tip[1]) ** 2))

            if startDistance is None:
                startDistance = d_thumb_index
                distance_buffer = [d_thumb_index] * BUFFER_SIZE
            else:
                distance_buffer.append(d_thumb_index)
                if len(distance_buffer) > BUFFER_SIZE:
                    distance_buffer.pop(0)

                median_distance = int(np.median(distance_buffer))
                if abs(median_distance - startDistance) > handshake_threshold:
                    scale = ((median_distance - startDistance) // 2)
                    h1, w1 = img_orig.shape[:2]
                    newH, newW = ((h1 + scale) // 2) * 2, ((w1 + scale) // 2) * 2

                    newH = min(max(1, newH), FRAME_HEIGHT)
                    newW = min(max(1, newW), FRAME_WIDTH)

                    img = cv2.resize(img_orig, (newW, newH))  # Resize from the original image

        else:
            startDistance = None

        cv2.imshow("Camera Input", frame)
        frames_elapsed += 1

        key = cv2.waitKey(1) & 0xFF
        if key == ord('x'):
            break
        elif key == ord('u'):
            OBJ_THRESHOLD += 1
            print(f"Increasing OBJ_THRESHOLD to {OBJ_THRESHOLD}")
        elif key == ord('d'):
            OBJ_THRESHOLD -= 1
            print(f"Decreasing OBJ_THRESHOLD to {OBJ_THRESHOLD}")

    capture.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
