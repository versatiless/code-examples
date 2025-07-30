from groundingdino.util.inference import load_model, load_image, predict, annotate
import supervision as sv
import matplotlib
matplotlib.use('TkAgg')

CONFIG_PATH = 'groundingdino/config/GroundingDINO_SwinT_OGC.py'

# Checkpoint is the model weight. We need to download model weight
CHECK_POINT_PATH = 'weights/groundingdino_swint_ogc.pth'

my_GD_model = load_model(CONFIG_PATH, CHECK_POINT_PATH)

IMAGE_PATH = 'image/test.jpg'
TEXT_PROMPT = 'road crack'
BOX_THRESHOLD = 0.35
TEXT_THRESHOLD = 0.25

image_source, my_image = load_image(IMAGE_PATH)

detected_boxes, accuracy, obj_name = predict(
    model=my_GD_model,
    image=my_image,
    caption=TEXT_PROMPT,
    box_threshold=BOX_THRESHOLD,
    text_threshold=TEXT_THRESHOLD
)

# annotate image
annotated_image = annotate(image_source=image_source, boxes=detected_boxes, logits=accuracy, phrases=obj_name)
print(annotated_image.shape)

# display image using supervision
sv.plot_image(annotated_image)