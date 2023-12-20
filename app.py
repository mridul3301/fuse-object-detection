from flask import Flask, render_template, request, send_file
from PIL import Image
from io import BytesIO

import config
import torch

from torchvision import transforms

from model import YOLOv3
from utils import *

# Load the model
model = YOLOv3(num_classes=config.NUM_CLASSES)

model_path = "Yolov3_epoch80.pth"
state_dict = torch.load(model_path, map_location=torch.device('cpu'))
model.load_state_dict(state_dict)

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

    input_image = Image.open(file)

    transform = transforms.Compose([
    transforms.Resize((config.IMAGE_SIZE, config.IMAGE_SIZE)),  # Replace input_size with your model's input size
    transforms.ToTensor(),
        ])

    input_tensor = transform(input_image).unsqueeze(0).to(config.DEVICE)

        # Evaluation mode
    model.eval()

    # Perform inference on the image
    with torch.no_grad():
        out = model(input_tensor)
        batch_size, A, S, _, _ = out[0].shape
        anchor = torch.tensor([*config.ANCHORS[0]]).to(config.DEVICE) * S
        boxes_scale_i = cells_to_bboxes(
            out[0], anchor, S=S, is_preds=True
        )
        bboxes = []
        for idx, (box) in enumerate(boxes_scale_i):
            bboxes += box

        nms_boxes = non_max_suppression(
            bboxes, iou_threshold=0.5, threshold=0.6, box_format="midpoint",
        )
        x = plot_image(input_image, nms_boxes)
    # Use PIL to open and process the image (you can replace this with your own image processing logic)
    #img = Image.open(file)
    # In this example, we're just returning the original image
    return x

if __name__ == '__main__':
    app.run(debug=True)
