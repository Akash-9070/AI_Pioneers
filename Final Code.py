# Import necessary libraries
import pandas as pd  # For handling data in tabular form
import json  # For saving data in JSON format
from PIL import Image, ImageEnhance  # For handling and enhancing images
import pytesseract  # For extracting text from images
from io import BytesIO  # For reading bytes in memory
import requests  # For making HTTP requests to fetch images
import re  # For pattern matching in text
import numpy as np  # For numerical operations
from skimage.restoration import denoise_tv_chambolle  # For denoising images
import cv2  # For image processing
from skimage.transform import rotate  # For rotating images

# Resize image while maintaining the aspect ratio
def resize_image(image, base_width):
    w_percent = (base_width / float(image.size[0]))  # Calculate width percentage
    h_size = int((float(image.size[1]) * float(w_percent)))  # Calculate the new height
    return image.resize((base_width, h_size), Image.LANCZOS)  # Resize using high-quality filter

# Apply denoising using Chambolle method
def denoise_image(image):
    image_np = np.array(image)  # Convert image to a NumPy array
    denoised_np = denoise_tv_chambolle(image_np, weight=0.1)  # Apply total variation denoising
    return Image.fromarray((denoised_np * 255).astype(np.uint8))  # Convert back to PIL image

# Apply adaptive thresholding to enhance contrast and binarize image
def adaptive_threshold(image):
    image_np = np.array(image)  # Convert image to NumPy array
    # If image is already grayscale
    if len(image_np.shape) == 2:
        gray = image_np
    else:
        gray = cv2.cvtColor(image_np, cv2.COLOR_BGR2GRAY)  # Convert color image to grayscale
    thresh = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
                                   cv2.THRESH_BINARY, 11, 2)  # Apply adaptive thresholding
    return Image.fromarray(thresh)  # Convert back to PIL image

# Correct skew by detecting the angle and rotating the image
def deskew_image(image):
    image_np = np.array(image)  # Convert image to NumPy array
    coords = np.column_stack(np.where(image_np > 0))  # Get coordinates of non-black pixels
    angle = cv2.minAreaRect(coords)[-1]  # Find the minimum area rectangle angle
    # Adjust angle to avoid large rotations
    if angle < -45:
        angle = -(90 + angle)
    else:
        angle = -angle
    rotated = rotate(image_np, angle, resize=True)  # Rotate image based on skew angle
    return Image.fromarray((rotated * 255).astype(np.uint8))  # Convert back to PIL image

# Enhance image sharpness
def enhance_sharpness(image):
    enhancer = ImageEnhance.Sharpness(image)  # Create sharpness enhancer
    return enhancer.enhance(2)  # Increase sharpness by factor of 2

# Remove noise or lines from the image using a median filter
def remove_lines(image):
    return image.filter(ImageFilter.MedianFilter(size=3))  # Apply median filter to remove noise

# Adjust contrast adaptively based on image brightness
def adaptive_contrast(image):
    gray_image = image.convert('L')  # Convert image to grayscale
    mean_brightness = np.mean(np.array(gray_image))  # Calculate mean brightness
    enhancer = ImageEnhance.Contrast(image)  # Create contrast enhancer
    # Adjust contrast based on brightness
    factor = 3.0 if mean_brightness < 128 else 1.2  
    return enhancer.enhance(factor)  # Apply contrast enhancement

# Preprocess image for better OCR results
def preprocess_image(image):
    image = image.convert('L')  # Convert image to grayscale
    skew_angle = detect_skew(image)  # Detect skew angle
    # If the skew angle is significant, deskew the image
    if abs(skew_angle) > 5:
        image = deskew_image(image)
    image = adaptive_contrast(image)  # Enhance contrast
    # Apply thresholding to make the image binary
    image = image.point(lambda p: p > 128 and 255)
    return image

# Detect skew angle in the image using Hough Line Transform
def detect_skew(image):
    image_np = np.array(image)  # Convert image to NumPy array
    edges = cv2.Canny(image_np, 50, 150, apertureSize=3)  # Detect edges in the image
    # Detect lines in the image
    lines = cv2.HoughLinesP(edges, 1, np.pi / 180, threshold=100, minLineLength=100, maxLineGap=10)
    angles = []
    # Calculate angle for each detected line
    if lines is not None:
        for line in lines:
            x1, y1, x2, y2 = line[0]
            angle = np.degrees(np.arctan2(y2 - y1, x2 - x1))
            angles.append(angle)
    return np.median(angles) if angles else 0  # Return median angle

# Download image from URL and extract text using Tesseract OCR
def process_image(url):
    try:
        response = requests.get(url, timeout=10)  # Fetch image from URL
        response.raise_for_status()  # Check for request errors
        img = Image.open(BytesIO(response.content))  # Open image from response
        preprocessed_img = preprocess_image(img)  # Preprocess image
        extracted_text_tesseract = pytesseract.image_to_string(preprocessed_img)  # Extract text using Tesseract
        return extracted_text_tesseract
    except requests.exceptions.RequestException as e:
        print(f"Error fetching the image from URL: {url}. Error: {e}")  # Handle request errors
    except OSError as e:
        print(f"Error opening or processing the image from URL: {url}. Error: {e}")  # Handle image processing errors
    return ""

# Extract value with unit from text based on the entity name
def extract_values_for_entity(text, entity_name):
    # Define a mapping from unit abbreviations to full unit names
    unit_mapping = {
        # Unit mappings for various measurement units
        'g': 'gram', 'kg': 'kilogram', 'mg': 'milligram', 'lb': 'pound', 'oz': 'ounce', 'ton': 'ton',
        'G': 'gram', 'KG': 'kilogram', 'MG': 'milligram', 'LB': 'pound', 'OZ': 'ounce', 'TON': 'ton',
        'cm': 'centimetre', 'm': 'metre', 'mm': 'millimetre', 'ft': 'foot', 'in': 'inch', 'yd': 'yard',
        'CM': 'centimetre', 'M': 'metre', 'MM': 'millimetre', 'FT': 'foot', 'IN': 'inch', 'YD': 'yard',
        'V': 'volt', 'kv': 'kilovolt', 'mv': 'millivolt', 'v': 'volt', 'KV': 'kilovolt', 'MV': 'millivolt', 
        'kV': 'kilovolt', 'mV': 'millivolt', 'W': 'watt', 'kW': 'kilowatt', 'w': 'watt', 'KW': 'kilowatt',
        'L': 'litre', 'ml': 'millilitre', 'gal': 'gallon', 'pt': 'pint', 'cup': 'cup',
        'l': 'litre', 'ML': 'millilitre', 'GAL': 'gallon', 'PT': 'pint', 'Cup': 'cup'
    }
    # Regular expression pattern to extract numerical values and units
    pattern = r'(\d+\.?\d*)\s?([a-zA-Z]+)'
    matches = re.findall(pattern, text)  # Find all matches
    for match in matches:
        value, unit_abbr = match
        full_unit = unit_mapping.get(unit_abbr, None)  # Map abbreviation to full unit name
        if full_unit:
            return f"{value} {full_unit}"  # Return value with full unit
    return None

# Process a dataset (CSV file) to extract values from images
def process_dataset(csv_file):
    df = pd.read_csv(csv_file)  # Load dataset into a pandas DataFrame

    results = []  # Store results

    # Iterate through each row in the dataset
    for i, row in df.iterrows():
        image_url = row['image_link']  # Get image URL from dataset
        image_id = row['index']  # Get image ID
        entity_name = row['entity_name']  # Get entity name to extract value for
        print(f"Processing image {i + 1} with ID: {image_id}")
        extracted_text_tesseract = process_image(image_url)  # Process image and extract text
        extracted_value_with_unit_tesseract = extract_values_for_entity(extracted_text_tesseract, entity_name)  # Extract value and unit
        print(f"Extracted value for '{entity_name}' using Tesseract: {extracted_value_with_unit_tesseract}")

        # Append results to list
        results.append({
            'index': image_id,
            'entity_name': entity_name,
            'extracted_value_tesseract': extracted_value_with_unit_tesseract
        })

    # Save results to a JSON file
    with open('extracted_results.json', 'w') as f:
        json.dump(results, f, indent=4)

# Define the path to the CSV file
csv_file_path = '/content/drive/MyDrive/Amazon ML Challenge/student_resource 3/part_1.csv'  # Replace with actual path
process_dataset(csv_file_path)  # Call function to process dataset

# New code to process JSON files and convert them to a combined CSV file
import json
import pandas as pd
import io
from google.colab import files

# Convert JSON data to pandas DataFrame
def json_to_dataframe(json_data, filename):
    if isinstance(json_data, list):
        df = pd.DataFrame(json_data)  # Convert list of dictionaries to DataFrame
    elif isinstance(json_data, dict):
        df = pd.DataFrame([json_data])  # Convert dictionary to DataFrame
    else:
        raise ValueError(f"Unsupported JSON structure in file: {filename}")
    
    # Add a column to identify the source file
    df['source_file'] = filename
    return df

# Upload multiple JSON files
print("Please select multiple JSON files to upload:")
uploaded = files.upload()

all_dataframes = []  # List to store DataFrames

# Process each uploaded file
for filename, content in uploaded.items():
    try:
        json_data = json.loads(content)  # Load JSON content
        df = json_to_dataframe(json_data, filename)  # Convert JSON to DataFrame
        all_dataframes.append(df)  # Append DataFrame to list
        print(f"Successfully processed: {filename}")
    except Exception as e:
        print(f"Error processing {filename}: {str(e)}")

# Combine all DataFrames
if all_dataframes:
    combined_df = pd.concat(all_dataframes, ignore_index=True)  # Combine all DataFrames into one
    
    # Convert to CSV format
    csv_buffer = io.StringIO()
    combined_df.to_csv(csv_buffer, index=False)  # Save to CSV in memory
    csv_content = csv_buffer.getvalue()

    # Save and download the combined CSV file
    with open('combined_output.csv', 'w') as f:
        f.write(csv_content)
    
    files.download('combined_output.csv')  # Download CSV file
    
    print("\nConversion complete. The combined CSV file has been downloaded.")
    
    # Display info about the combined dataset
    print("\nCombined dataset info:")
    print(f"Total number of rows: {len(combined_df)}")
    print(f"Columns: {', '.join(combined_df.columns)}")
    print("\nFirst few rows of the combined CSV:")
    print(combined_df.head().to_string())
    
    # Display source file distribution
    print("\nRows per source file:")
    print(combined_df['source_file'].value_counts())
else:
    print("No valid data to process. Please check your JSON files and try again.")
