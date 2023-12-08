import os
from flask import Flask, flash, request, redirect, url_for, render_template
from werkzeug.utils import secure_filename
import cv2 as cv
import numpy as np
import cvlib as cvl
from cvlib.object_detection import draw_bbox
import matplotlib.pyplot as plt
import base64
from io import BytesIO

UPLOAD_FOLDER = 'uploads'
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg'}

app = Flask(__name__)

app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

def image_process(image_path):
    def rescale_frame(frame, scale=1.0):
        width = int(frame.shape[1] * scale)
        height = int(frame.shape[0] * scale)
        dimension = (width, height)
        return cv.resize(frame, dimension, interpolation=cv.INTER_AREA)

    def mark_objects_with_numbers(image, boxes, labels, counts):
        for i in range(len(counts)):
            box = boxes[i]
            label = labels[i]

            x, y, w, h = map(int, box)
            cv.rectangle(image, (x, y), (x + w, y + h), (0, 255, 0), 2)

            # Mark the number on the object
            cv.putText(image, str(i + 1), (x, y - 10), cv.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

        return image

    img = cv.imread(image_path)
    resize_img = rescale_frame(img)

    # Perform object detection
    boxes, labels, counts = cvl.detect_common_objects(img)

    # Mark objects with numbers
    output = mark_objects_with_numbers(img.copy(), boxes, labels, counts)

    # Convert the processed image to base64
    _, buffer = cv.imencode('.png', cv.cvtColor(output, cv.COLOR_BGR2RGB))
    image_base64 = base64.b64encode(buffer).decode('utf-8')

    return image_base64

def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def remove_object_by_number(image, boxes, counts, number_to_remove):
    # Create a mask to exclude the bounding box of the specified number
    mask = np.ones(image.shape[:2], dtype="uint8") * 255

    for i in range(len(counts)):
        if i + 1 == number_to_remove:
            box = boxes[i]
            x, y, w, h = map(int, box)
            mask[y:y+h, x:x+w] = 0  # Set the region of the object to remove to 0

    # Apply the mask to remove the object
    result = cv.bitwise_and(image, image, mask=mask)

    return result


@app.route('/')
def index():
    return render_template('file.html')

@app.route('/edit',methods=["GET","POST"])
def edit():
    if request.method == 'POST':
        if 'file' not in request.files:
            flash('No file part')
            return 'error'
        file = request.files['file']
        if file.filename == '':
            return "Error no selected file."
        if file and allowed_file(file.filename):
            filename = secure_filename(file.filename)
            file.save(os.path.join(app.config['UPLOAD_FOLDER'], filename))
            image_path = f'{UPLOAD_FOLDER}/{filename}'
            processed_image = image_process(image_path)
            return render_template('file2.html', processed_image=processed_image)

    return render_template('file2.html')




@app.route('/remove_object', methods=["POST"])
def remove_object():
    if request.method == 'POST':
        object_number = int(request.form['object_number'])
        processed_image_base64 = request.form['processed_image']

        # Decode base64 and remove the specified object
        processed_image = cv.imdecode(np.frombuffer(base64.b64decode(processed_image_base64), dtype=np.uint8), -1)

        # Perform object detection again
        boxes, labels, counts = cvl.detect_common_objects(processed_image)

        if object_number > 0 and object_number <= len(counts):
            # Remove the specified object
            processed_image_with_removed_object = remove_object_by_number(processed_image, boxes, counts, object_number)

            # Convert the processed image with removed object to base64
            _, buffer = cv.imencode('.png', cv.cvtColor(processed_image_with_removed_object, cv.COLOR_BGR2RGB))
            processed_image_base64 = base64.b64encode(buffer).decode('utf-8')

            return render_template('file3.html', processed_image=processed_image_base64)

    return 'Error in removing object'


if __name__ == '__main__':
    app.run(debug=True)
