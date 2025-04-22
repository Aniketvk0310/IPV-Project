import os
import uuid
import cv2
import numpy as np
from flask import Flask, render_template, request, send_file

app = Flask(__name__)
UPLOAD_FOLDER = 'static/uploads'
RESULT_FOLDER = 'static/results'

os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(RESULT_FOLDER, exist_ok=True)

def enhance_medical_image(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    enhanced = clahe.apply(gray)
    lap = cv2.Laplacian(enhanced, cv2.CV_64F)
    sharp = cv2.convertScaleAbs(enhanced - 0.5 * lap)
    return cv2.cvtColor(sharp, cv2.COLOR_GRAY2BGR)

def enhance_satellite_image(img):
    smooth = cv2.bilateralFilter(img, 9, 75, 75)
    lap = cv2.Laplacian(smooth, cv2.CV_64F)
    sharp = cv2.convertScaleAbs(smooth - 0.7 * lap)
    return sharp

def enhance_cctv_footage(img):
    denoised = cv2.fastNlMeansDenoisingColored(img, None, 10, 10, 7, 21)
    gamma = 1.5
    look_up = np.array([((i / 255.0) ** gamma) * 255 for i in np.arange(256)]).astype("uint8")
    gamma_corrected = cv2.LUT(denoised, look_up)
    gray = cv2.cvtColor(gamma_corrected, cv2.COLOR_BGR2GRAY)
    sobelx = cv2.Sobel(gray, cv2.CV_64F, 1, 0)
    sobely = cv2.Sobel(gray, cv2.CV_64F, 0, 1)
    edges = cv2.convertScaleAbs(cv2.add(sobelx, sobely))
    final = cv2.addWeighted(gray, 0.8, edges, 0.2, 0)
    return cv2.cvtColor(final, cv2.COLOR_GRAY2BGR)

def enhance_document(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    binary = cv2.adaptiveThreshold(gray, 255, 
                                   cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                   cv2.THRESH_BINARY, 11, 2)
    return cv2.cvtColor(binary, cv2.COLOR_GRAY2BGR)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/process', methods=['POST'])
def process_image():
    file = request.files['image']
    if not file:
        return "No image uploaded", 400

    filename = str(uuid.uuid4()) + ".png"
    filepath = os.path.join(UPLOAD_FOLDER, filename)
    file.save(filepath)

    image = cv2.imread(filepath)

    # Get domain
    domain = request.form.get("domain")
    
    # Apply domain-specific enhancement
    if domain == "medical":
        image = enhance_medical_image(image)
    elif domain == "satellite":
        image = enhance_satellite_image(image)
    elif domain == "cctv":
        image = enhance_cctv_footage(image)
    elif domain == "document":
        image = enhance_document(image)

    # Apply noise reduction
    noise_type = request.form.get("noise")
    if noise_type == "gaussian":
        image = cv2.GaussianBlur(image, (5, 5), 0)
    elif noise_type == "median":
        image = cv2.medianBlur(image, 5)
    elif noise_type == "bilateral":
        image = cv2.bilateralFilter(image, 9, 75, 75)

    # Apply sharpening
    sharpness = float(request.form.get("sharpness", 1.5))
    kernel = np.array([[-1, -1, -1],
                       [-1, 9 * sharpness, -1],
                       [-1, -1, -1]])
    image = cv2.filter2D(image, -1, kernel)

    # Edge Detection (optional)
    edge_type = request.form.get("edge")
    if edge_type == "sobel":
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        edge_img = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=5)
        image = cv2.convertScaleAbs(edge_img)
        image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
    elif edge_type == "canny":
        edge_img = cv2.Canny(image, 100, 200)
        image = cv2.cvtColor(edge_img, cv2.COLOR_GRAY2BGR)
    elif edge_type == "laplacian":
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        edge_img = cv2.Laplacian(gray, cv2.CV_64F)
        image = cv2.convertScaleAbs(edge_img)
        image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)

    result_path = os.path.join(RESULT_FOLDER, filename)
    cv2.imwrite(result_path, image)

    return send_file(result_path, mimetype='image/png')

if __name__ == '__main__':
    app.run(debug=True)
