import streamlit as st
import cv2
import numpy as np
from PIL import Image
import pytesseract

pytesseract.pytesseract.tesseract_cmd = '/opt/homebrew/bin/tesseract'
# Function to apply various preprocessing steps to an image
def preprocess_image(image, options):
    preprocessed_images = {}
    gray_image = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    
    if options['grayscale']:
        preprocessed_images['Grayscale Image'] = gray_image
    if options['gaussian_blur']:
        blurred_image = cv2.GaussianBlur(gray_image, (5, 5), 0)
        preprocessed_images['Gaussian Blurred Image'] = blurred_image
    if options['edge_detection']:
        edges_image = cv2.Canny(gray_image, 100, 200)
        preprocessed_images['Edge Detected Image'] = edges_image
    if options['thresholding']:
        ret, threshold_image = cv2.threshold(gray_image, 127, 255, cv2.THRESH_BINARY)
        preprocessed_images['Thresholded Image'] = threshold_image
    if options['dilation']:
        kernel = np.ones((5,5),np.uint8)
        dilated_image = cv2.dilate(gray_image, kernel, iterations = 1)
        preprocessed_images['Dilated Image'] = dilated_image
    if options['erosion']:
        kernel = np.ones((5,5),np.uint8)
        eroded_image = cv2.erode(gray_image, kernel, iterations = 1)
        preprocessed_images['Eroded Image'] = eroded_image
    
    return preprocessed_images

# Function to perform OCR and extract text from an image
def perform_ocr(image):
    try:
        text = pytesseract.image_to_string(image)
        return text
    except pytesseract.TesseractNotFoundError:
        return "Tesseract OCR is not installed or not in PATH."

# Streamlit app layout
st.title('Image Processing and OCR App')

# Upload image
uploaded_file = st.file_uploader("Choose an image...", type=['png', 'jpg', 'jpeg', 'webp'])

# Checkboxes for preprocessing options
options = {
    'grayscale': st.checkbox('Grayscale Image', value=True),
    'gaussian_blur': st.checkbox('Gaussian Blurred Image'),
    'edge_detection': st.checkbox('Edge Detected Image'),
    'thresholding': st.checkbox('Thresholded Image'),
    'dilation': st.checkbox('Dilated Image'),
    'erosion': st.checkbox('Eroded Image')
}

# OCR button
ocr_button = st.checkbox("Extract Text from Image")

if uploaded_file is not None:
    image = np.array(Image.open(uploaded_file))
    st.image(image, caption='Original Image', use_column_width=True)
    
    # Preprocess and display images
    if st.button('Process Image'):
        preprocessed_images = preprocess_image(image, options)
        for title, img in preprocessed_images.items():
            if len(img.shape) == 2:
                img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
            st.image(img, caption=title, use_column_width=True)

    # Perform OCR and display text
    if ocr_button:
        text = perform_ocr(image)
        if text.strip() != '':
            st.subheader('Extracted Text:')
            st.write(text)
        else:
            st.write('No text recognized in the image.')