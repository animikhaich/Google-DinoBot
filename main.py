import numpy as np
import cv2, math
from mss.linux import MSS as mss
from PIL import Image
import time, pyautogui

monitor = {"top": 250, "left": 10, "width": 1280, "height": 320}

sct = mss()


def grab_frame(monitor: dict):
    # Get the screen capture
    frame = sct.grab(monitor)

    # Convert to Numpy Array
    frame = np.array(frame, np.uint8)

    # Convert to Grayscale
    frame = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)

    return frame


def binary_threshold(frame: np.ndarray):
    mean_pix_val = np.mean(frame)
    # Take care of day and night transitions
    if mean_pix_val < 150:
        _, thresh = cv2.threshold(frame, 127, 255, cv2.THRESH_BINARY)
    else:
        _, thresh = cv2.threshold(frame, 127, 255, cv2.THRESH_BINARY_INV)

    return thresh


def remove_clutter(frame, kernel_size=5, repeat=1):
    kernel = np.ones((kernel_size, kernel_size), np.uint8)

    frame = cv2.morphologyEx(frame, cv2.MORPH_OPEN, kernel)
    return frame


def get_contour_boxes(frame):
    # Get Contours
    contours, _ = cv2.findContours(frame, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)

    # frame = np.zeros((h, w, 3), np.uint8)
    frame = cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR)

    # Generate Box Coordinates from the Contours
    objects = list()
    for contour in contours:
        xs = contour[:, :, 0]
        ys = contour[:, :, 1]

        xmin = np.min(xs)
        ymin = np.min(ys)
        xmax = np.max(xs)
        ymax = np.max(ys)

        objects.append([xmin, ymin, xmax, ymax])
        cv2.rectangle(frame, (xmin, ymin), (xmax, ymax), (0, 255, 0), 2)

    return frame, objects


def calc_fps(start_time, frame=None, print_on_console=False):
    fps = 1 / (time.time() - start_time)
    fps_str = f"FPS: {fps:.2f}"
    if print_on_console:
        print(fps_str)
    if (
        frame is not None
        and isinstance(frame, np.ndarray)
        and min(frame.shape[:-2]) > 50
    ):
        cv2.putText(
            frame, fps_str, (20, 30), cv2.FONT_HERSHEY_COMPLEX, 1, (0, 255, 0), 1
        )
        return frame


def get_dino(frame, area_sorted_boxes):
    height, width = frame.shape[:2]
    edge_thresh = width * 0.1

    left_boxes = filter(
        lambda box: box[2] < edge_thresh and box[0] < edge_thresh, area_sorted_boxes
    )

    # Sort Bounding Boxes by Area
    sorted_boxes = sorted(
        left_boxes, key=lambda box: (box[2] - box[0]) * (box[3] - box[1]), reverse=True
    )

    if len(sorted_boxes) < 1:
        return frame, None

    dino_box = sorted_boxes[0]
    x1, y1, x2, y2 = dino_box
    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 255), 2)

    return frame, dino_box


def detect_edges(frame: np.ndarray):
    edges = cv2.Canny(frame, 100, 200)
    return edges


def polar2cartesian(polar_lines):
    if polar_lines is None:
        return None

    cartesian_lines = list()

    for line in polar_lines:
        line = np.squeeze(line)
        rho, theta = line
        a = math.cos(theta)
        y0 = b * rho
        pt1 = (int(x0 + 1000 * (-b)), int(y0 + 1000 * (a)))
        pt2 = (int(x0 - 1000 * (-b)), int(y0 - 1000 * (a)))
        cartesian_lines.append([pt1, pt2])
    return cartesian_lines


def get_land(frame: np.ndarray):

    # Some kinda-constants
    frame_width = frame.shape[1]
    minLineLength = frame_width // 2
    maxLineGap = frame_width // 20

    # Get the HoughLines
    lines = cv2.HoughLinesP(
        image=frame,
        rho=1,
        theta=np.pi / 180,
        threshold=50,
        lines=np.array([]),
        minLineLength=minLineLength,
        maxLineGap=maxLineGap,
    )

    # Ignore if lines are not found
    if lines is None:
        return []

    # iterate over the lines, clean them and return
    clean_lines = [np.squeeze(line) for line in lines]

    sorted_lines = sorted(
        clean_lines,
        key=lambda line: np.linalg.norm(
            np.array(line[0], line[1]) - np.array(line[2], line[3])
        ),
        reverse=True,
    )

    return sorted_lines[0]


def draw_lines(frame: np.ndarray, lines: list):
    for line in lines:
        if len(line) == 0:
            continue

        x1, y1, x2, y2 = line
        frame = cv2.line(frame, (x1, y1), (x2, y2), (0, 0, 255), 3, cv2.LINE_AA)

    return frame


def calc_perp_dist(line, point):
    p1 = np.array([line[0], line[1]])
    p2 = np.array([line[2], line[3]])
    p3 = np.array(point)

    dist = np.linalg.norm(np.cross(p2 - p1, p1 - p3)) / np.linalg.norm(p2 - p1)

    return dist


def get_bottom_center_point(bbox):
    if bbox is None:
        return None

    x1, y1, x2, y2 = bbox
    cx = x1 + (x2 - x1) // 2

    return (cx, y2)


while True:
    start_time = time.time()
    gray_frame = grab_frame(monitor)
    edge_frame = detect_edges(gray_frame)
    line = get_land(edge_frame)

    thresh_frame = binary_threshold(gray_frame)
    cleaned_frame = remove_clutter(thresh_frame, kernel_size=5)
    contour_frame, area_sorted_boxes = get_contour_boxes(cleaned_frame)
    dino_frame, dino_box = get_dino(contour_frame, area_sorted_boxes)

    dino_point = get_bottom_center_point(dino_box)

    dist = calc_perp_dist(line, dino_point)
    # print(dist)

    display_frame = draw_lines(dino_frame, [line])
    fps_frame = calc_fps(start_time, display_frame)

    cv2.imshow("TheGame", fps_frame)
    if cv2.waitKey(1) == ord("q"):
        break

cv2.destroyAllWindows()
