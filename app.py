from flask import Flask, render_template, request, send_file
from PIL import Image
from io import BytesIO

app = Flask(__name__)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/upload', methods=['POST'])
def upload():
    # Check if a file was submitted
    if 'file' not in request.files:
        return "No file part"

    file = request.files['file']

    # Check if the file has a name
    if file.filename == '':
        return "No selected file"

    # Process the image (in this case, just display it)
    processed_image = process_image(file)

    # Save the processed image to a BytesIO object
    img_io = BytesIO()
    processed_image.save(img_io, 'PNG')
    img_io.seek(0)

    # Return the processed image as a response
    return send_file(img_io, mimetype='image/png')

def process_image(file):
    # Use PIL to open and process the image (you can replace this with your own image processing logic)
    img = Image.open(file)
    # In this example, we're just returning the original image
    return img

if __name__ == '__main__':
    app.run(debug=True)
