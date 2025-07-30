The my_script.py uses the Grounding DINO model to perform text-guided object detection. Given an input image and a text prompt (e.g., "road crack"), the model detects and highlights matching regions in the image.

- Loads a pre-trained Grounding DINO model.
- Reads an image and a user-defined text prompt.
- Runs zero-shot object detection guided by the text.
- Annotates detected objects with bounding boxes and labels.
- Displays the result using the `supervision` library.
