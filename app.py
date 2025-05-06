from flask import Flask, request, jsonify
from flask_cors import CORS
import numpy as np
import cv2
import base64
from io import BytesIO

app = Flask(__name__)
CORS(app, resources={r"/*": {"origins": "https://imagecrypt.vercel.app"}})

def manipulate_pixels_optimized(rgb_pixels, binary_text):
    """
    Optimized function to manipulate only the blue (B) value of each pixel
    based on a binary string.

    - If the binary value is 1, modify the blue value (+1 unless 255, then -1).
    - If the binary value is 0, keep the blue value unchanged.
    - Marks the end of manipulation with [254, 254, 254] and [1, 1, 1].

    Parameters:
        rgb_pixels (list): A 3D list representing the RGB values of an image.
        binary_text (str): A binary string to use for manipulation.

    Returns:
        np.ndarray: The manipulated RGB pixel array as a NumPy array.
    """
    pixels_needed = len(binary_text)
    img_array = np.array(rgb_pixels, dtype=np.uint8).reshape(-1, 3)
    num_pixels = img_array.shape[0]

    modification_indices = np.arange(min(pixels_needed, num_pixels))
    blue_pixels = img_array[modification_indices, 2]
    binary_values = np.array([int(bit) for bit in binary_text[:len(modification_indices)]])

    # Apply changes based on binary values
    increase_mask = (binary_values == 1) & (blue_pixels < 255)
    decrease_mask = (binary_values == 1) & (blue_pixels == 255)

    img_array[modification_indices[increase_mask], 2] += 1
    img_array[modification_indices[decrease_mask], 2] -= 1

    # Mark the end of manipulation
    if pixels_needed < num_pixels - 1:  # Ensure space for both markers
        img_array[pixels_needed] = [254, 254, 254]
        img_array[pixels_needed + 1] = [1, 1, 1]
    elif pixels_needed == num_pixels - 1:
        img_array[pixels_needed] = [254, 254, 254]

    return img_array.reshape(np.array(rgb_pixels).shape)

def save_image_optimized(modified_pixels_np):
    """
    Converts a NumPy array of modified pixels back to a base64 encoded PNG image.

    Parameters:
        modified_pixels_np (np.ndarray): 3D NumPy array of modified RGB pixel values.

    Returns:
        str: Base64 encoded image data.
    """
    img = cv2.cvtColor(modified_pixels_np, cv2.COLOR_RGB2BGR)
    _, img_encoded = cv2.imencode('.png', img)
    return base64.b64encode(img_encoded).decode('utf-8')

def extract_binary_from_pixels_optimized(original_pixels, modified_pixels):
    """
    Optimized function to extract binary data from the differences between
    original and modified images.
    """
    original_array = np.array(original_pixels).reshape(-1, 3)
    modified_array = np.array(modified_pixels).reshape(-1, 3)

    comparison = (modified_array[:, 2] != original_array[:, 2])
    end_marker_index = np.where((modified_array[:-1] == [254, 254, 254]) & (modified_array[1:] == [1, 1, 1]))[0]

    if end_marker_index.size > 0:
        comparison = comparison[:end_marker_index[0]]

    binary_text = np.where(comparison, '1', '0').tobytes().decode('ascii')
    return binary_text

def binary_to_text_optimized(binary_string):
    """
    Optimized function to convert a binary string to its original text.
    """
    n = len(binary_string)
    remainder = n % 8
    if remainder != 0:
        binary_string += '0' * (8 - remainder)

    byte_array = bytearray(int(binary_string[i:i+8], 2) for i in range(0, len(binary_string), 8))
    try:
        decoded_text = byte_array.decode('utf-8', errors='ignore')
        return decoded_text
    except UnicodeDecodeError:
        return ""

@app.route("/generate", methods=["POST"])
def process_generate():
    if "image" not in request.files or "text" not in request.form:
        return jsonify({"error": "Missing image or text"}), 400

    text = request.form["text"]
    image_file = request.files["image"]

    print("Image and Text Received for Generate")

    binary_text = ''.join(format(ord(char), '08b') for char in text)

    image_np = np.frombuffer(image_file.read(), np.uint8)
    img = cv2.imdecode(image_np, cv2.IMREAD_COLOR)
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    modified_pixels_np = manipulate_pixels_optimized(img_rgb.tolist(), binary_text)
    modified_image_base64 = save_image_optimized(modified_pixels_np)

    return jsonify({"image_base64": modified_image_base64})

@app.route("/get", methods=["POST"])
def process_get():
    if "image1" not in request.files or "image2" not in request.files:
        return jsonify({"error": "Missing images"}), 400

    image_file_1 = request.files["image1"]
    image_file_2 = request.files["image2"]

    print("2 images Received for Get")

    image_np_1 = np.frombuffer(image_file_1.read(), np.uint8)
    image_np_2 = np.frombuffer(image_file_2.read(), np.uint8)

    img_1 = cv2.imdecode(image_np_1, cv2.IMREAD_COLOR)
    img_2 = cv2.imdecode(image_np_2, cv2.IMREAD_COLOR)

    img_rgb_1 = cv2.cvtColor(img_1, cv2.COLOR_BGR2RGB)
    img_rgb_2 = cv2.cvtColor(img_2, cv2.COLOR_BGR2RGB)

    try:
        extracted_binary = extract_binary_from_pixels_optimized(img_rgb_1.tolist(), img_rgb_2.tolist())
        final_text = binary_to_text_optimized(extracted_binary)
        return jsonify({"Final_text": final_text})
    except Exception as e:
        print(f"Error: {e}")
        return jsonify({"error": "Error while processing images"}), 500

if __name__ == "__main__":
    app.run(debug=True, port=5000)