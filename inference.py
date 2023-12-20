import config
import torch

from torchvision import transforms
from PIL import Image

from model import YOLOv3
from utils import *

# Load the model
model = YOLOv3(num_classes=config.NUM_CLASSES)

model_path = "Yolov3_epoch80.pth"
state_dict = torch.load(model_path, map_location=torch.device('cpu'))
model.load_state_dict(state_dict)
# print(model)

# Load and preprocess image
image_path = "000193.jpg"
input_image = Image.open(image_path)

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
    plot_image(input_image, nms_boxes)