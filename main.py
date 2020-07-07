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
        b = math.sin(theta)
        x0 = a * rho
        y0 = b * rho
        pt1 = (int(x0 + 1000 * (-b)), int(y0 + 1000 * (a)))
        pt2 = (int(x0 - 1000 * (-b)), int(y0 - 1000 * (a)))
        cartesian_lines.append([pt1, pt2])
    return cartesian_lines


def get_land(frame: np.ndarray):
    lines = cv2.HoughLines(frame, 1, np.pi / 180, 150, None, 0, 0)
    lines = polar2cartesian(lines)

    frame = cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR)
    for line in lines:
        frame = cv2.line(frame, line[0], line[1], (0, 0, 255), 3, cv2.LINE_AA)

    cv2.imshow("Land", frame)
    return frame, lines


counter = 0
while True:
    start_time = time.time()
    gray_frame = grab_frame(monitor)
    edge_frame = detect_edges(gray_frame)
    get_land(edge_frame)
    thresh_frame = binary_threshold(gray_frame)
    cleaned_frame = remove_clutter(thresh_frame, kernel_size=5)
    contour_frame, area_sorted_boxes = get_contour_boxes(cleaned_frame)
    dino_frame, left_boxes = get_dino(contour_frame, area_sorted_boxes)

    # counter += 1
    # if counter % 500 == 0:
    #     pyautogui.press("up")

    fps_frame = calc_fps(start_time, dino_frame)

    cv2.imshow("TheGame", fps_frame)
    if cv2.waitKey(1) & 0xFF == ord("q"):
        cv2.destroyAllWindows()
        break
