import cv2
import numpy as np
import streamlit as st
from matplotlib import pyplot as plt

# Function to process the uploaded image
def process_image(image):
    # Convert image to the HSV color space to detect red color better
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

    # Define the range for detecting red color in HSV
    lower_red = np.array([0, 50, 50])
    upper_red = np.array([10, 255, 255])

    # Mask to keep only red regions
    mask1 = cv2.inRange(hsv, lower_red, upper_red)

    # Add the second red mask range
    lower_red2 = np.array([170, 50, 50])
    upper_red2 = np.array([180, 255, 255])
    mask2 = cv2.inRange(hsv, lower_red2, upper_red2)

    # Combine the two masks
    mask = mask1 + mask2

    # Find contours (blobs/colonies) based on the red mask
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    colony_info = []  # To store details about each colony

    # Loop through each detected colony
    for i, contour in enumerate(contours):
        # Get the area of the colony (size)
        area = cv2.contourArea(contour)

        # Ignore small blobs that might be noise
        if area < 10:
            continue

        # Get a bounding box around the colony
        x, y, w, h = cv2.boundingRect(contour)

        # Extract the colony's region of interest (ROI)
        colony_roi = image[y:y+h, x:x+w]

        # Calculate the mean color within the colony's region (BGR)
        mean_color = cv2.mean(colony_roi)[:3]
        mean_color_int = tuple(map(int, mean_color))

        # Get the color hex representation
        color_hex = "#{:02x}{:02x}{:02x}".format(mean_color_int[2], mean_color_int[1], mean_color_int[0])

        # Get the perimeter (for shape analysis)
        perimeter = cv2.arcLength(contour, True)

        # Shape: Calculate circularity = 4π(Area) / (Perimeter^2)
        circularity = 4 * np.pi * (area / (perimeter ** 2))

        # Classify the shape based on circularity
        shape = "Circle" if circularity > 0.8 else "Irregular"

        # Store colony details
        colony_info.append({
            "index": i + 1,
            "area": area,
            "mean_color": mean_color_int,
            "color_hex": color_hex,
            "shape": shape,
            "circularity": circularity,
        })

        # Draw the contour around the colony
        cv2.drawContours(image, [contour], -1, (0, 255, 0), 2)
        # Annotate the colony number and color on the image
        #cv2.putText(image, f'{i + 1} {color_hex}', (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

    return image, colony_info

# Streamlit UI
# Streamlit UI configuration
st.set_page_config(page_title="Skavch Colony Count Engine", layout="wide")

# Add an image to the header
st.image("bg1.jpg", use_column_width=True)
st.title("Skavch Colony Count Engine")
st.write("Upload an image containing colonies to detect their size, shape, and color.")

# Image upload
uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # Read the image file
    file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
    image = cv2.imdecode(file_bytes, 1)

    # Process the image
    output_image, colonies = process_image(image)

    # Show the processed output image
    st.image(cv2.cvtColor(output_image, cv2.COLOR_BGR2RGB), caption="Detected Colonies", use_column_width=True)

    # Display the last colony number directly as total colonies detected
    total_colonies = len(colonies)
    st.write(f"Total number of colonies detected: {total_colonies}")

    # Display colony details
    st.write("Colony Details:")
    for colony in colonies:
        st.write(f"Colony {colony['index']}:")
        st.write(f" - Area (size): {colony['area']} pixels")
        st.write(f" - Mean Color (BGR): {colony['mean_color']}")
        st.write(f" - Color Hex: {colony['color_hex']}")
        st.write(f" - Shape: {colony['shape']}")
        st.write(f" - Circularity: {colony['circularity']:.2f}")
        st.write("---")
