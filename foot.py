import cv2
import numpy as np
import imutils

# Constants
REFERENCE_OBJECT_WIDTH = 4.0  # Width of the reference object in cm
REFERENCE_OBJECT_HEIGHT = 4.0  # Height of the reference object in cm

def find_reference_object(contours):
    # Find the contour with the largest area (reference object)
    reference_contour = max(contours, key=lambda c: cv2.contourArea(c))
    return reference_contour

def calculate_foot_size(foot_contour, reference_object_contour):
    # Find the bounding rectangle for both foot and reference object contours
    foot_x, foot_y, foot_w, foot_h = cv2.boundingRect(foot_contour)
    ref_x, ref_y, ref_w, ref_h = cv2.boundingRect(reference_object_contour)

    # Calculate the scale factors for width and height
    width_scale = REFERENCE_OBJECT_WIDTH / float(ref_w)
    height_scale = REFERENCE_OBJECT_HEIGHT / float(ref_h)

    # Calculate the real-world dimensions of the foot
    foot_real_width = foot_w * width_scale
    foot_real_height = foot_h * height_scale

    return foot_real_width, foot_real_height

def main():
    # Load and preprocess the image
    image = cv2.imread('foot-bottom.jpg')
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (7, 7), 0)
    edges = cv2.Canny(blurred, 50, 150)

    # Find contours
    contours = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    contours = imutils.grab_contours(contours)

    # Identify the reference object contour
    reference_object_contour = find_reference_object(contours)

    # Filter the foot contour by excluding the reference object contour
    foot_contour = find_reference_object([c for c in contours if not np.array_equal(c, reference_object_contour)])

    # Calculate foot size
    foot_width, foot_height = calculate_foot_size(foot_contour, reference_object_contour)
    print(f"Foot dimensions: {foot_width:.2f} cm (width) x {foot_height:.2f} cm (height)")

if __name__ == "__main__":
    main()