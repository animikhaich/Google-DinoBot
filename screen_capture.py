import numpy as np
import cv2
from mss.linux import MSS as mss
from PIL import Image
import time

monitor = {"top": 250, "left": 10, "width": 1280, "height": 320}

sct = mss()


def grab_and_thresh(monitor: dict):
    # Get the screen capture
    image = sct.grab(monitor)

    # Convert to Numpy Array
    img = np.array(image, np.uint8)

    # Convert to Grayscale
    img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)

    # Threshold image
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

    # Sort Bounding Boxes by Area
    sorted_boxes = sorted(
        objects, key=lambda box: (box[2] - box[0]) * (box[3] - box[1]), reverse=True
    )

    return frame, sorted_boxes


while True:
    thresh_frame = grab_and_thresh(monitor)
    cleaned_frame = remove_clutter(thresh_frame, kernel_size=5)
    contour_frame, boxes = get_contour_boxes(cleaned_frame)

    cv2.imshow("TheGame", contour_frame)
    if cv2.waitKey(25) & 0xFF == ord("q"):
        cv2.destroyAllWindows()
        break
