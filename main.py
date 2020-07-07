import numpy as np
import cv2
from mss.linux import MSS as mss
from PIL import Image
import time, pyautogui

monitor = {"top": 250, "left": 10, "width": 1280, "height": 320}

sct = mss()


def grab_and_thresh(monitor: dict):
    # Get the screen capture
    image = sct.grab(monitor)

    # Convert to Numpy Array
    img = np.array(image, np.uint8)

    # Convert to Grayscale
    img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)

    mean_pix_val = np.mean(img)
    # Take care of day and night transitions
    if mean_pix_val < 150:
        _, thresh = cv2.threshold(img, 127, 255, cv2.THRESH_BINARY)
    else:
        _, thresh = cv2.threshold(img, 127, 255, cv2.THRESH_BINARY_INV)

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


counter = 0
while True:
    start_time = time.time()
    thresh_frame = grab_and_thresh(monitor)
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
