# basic imports
import numpy as np
import os
import tensorflow as tf
import pathlib
import cv2

# object detection imports
from object_detection.utils import ops as utils_ops
from object_detection.utils import label_map_util
from object_detection.utils import visualization_utils as vis_util

# TF model prepocessing
# patch tf1 into `utils.ops`
utils_ops.tf = tf.compat.v1

# Patch the location of gfile
tf.gfile = tf.io.gfile

# Disable the warning about unsupported CPU instructions
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'


def load_model(load_model_name):
    base_url = 'http://download.tensorflow.org/models/object_detection/'
    model_file = load_model_name + '.tar.gz'
    model_dir = tf.keras.utils.get_file(
        fname=load_model_name,
        origin=base_url + model_file,
        untar=True)

    model_dir = str(pathlib.Path(model_dir)) + "\\saved_model"
    model = tf.saved_model.load(str(model_dir))
    model = model.signatures['serving_default']
    return model


def run_inference_for_single_frame(model, frame_img):
    # convert image to correct input
    frame_np = np.asarray(frame_img)
    input_tensor = tf.convert_to_tensor(frame_np)
    input_tensor = input_tensor[tf.newaxis, ...]

    # create output dictionary
    output_dict = model(input_tensor)

    # find objects
    num_detections = int(output_dict.pop('num_detections'))
    output_dict = {key: value[0, :num_detections].numpy()
                   for key, value in output_dict.items()}
    output_dict['num_detections'] = num_detections

    # detection_classes should be ints.
    output_dict['detection_classes'] = output_dict['detection_classes'].astype(np.int64)

    # Handle models with masks:
    if 'detection_masks' in output_dict:
        # Reframe the the bbox mask to the image size.
        detection_masks_reframed = utils_ops.reframe_box_masks_to_image_masks(
            output_dict['detection_masks'], output_dict['detection_boxes'],
            frame_np.shape[0], frame_np.shape[1])
        detection_masks_reframed = tf.cast(detection_masks_reframed > 0.5,
                                           tf.uint8)
        output_dict['detection_masks_reframed'] = detection_masks_reframed.numpy()
    return output_dict


def frame_obj_detect(model, frame_img):
    # the array based representation of the image will be used later in order to prepare the
    # result image with boxes and labels on it.
    frame_np = np.array(frame_img)
    # Actual detection.
    output_dict = run_inference_for_single_frame(model, frame_np)
    # Visualization of the results of a detection.
    vis_util.visualize_boxes_and_labels_on_image_array(
        frame_np,
        output_dict['detection_boxes'],
        output_dict['detection_classes'],
        output_dict['detection_scores'],
        category_index,
        instance_masks=output_dict.get('detection_masks_reframed', None),
        use_normalized_coordinates=True,
        line_thickness=8)
    # convert numpy array back to image and return it
    return frame_np


# create video object
vid_stream = cv2.VideoCapture(0, cv2.CAP_DSHOW)

# Labels list file path
PATH_TO_LABELS = 'object_detection\\data\\mscoco_label_map.pbtxt'
category_index = label_map_util.create_category_index_from_labelmap(PATH_TO_LABELS, use_display_name=True)

# Load object detecting model
model_name = 'ssd_mobilenet_v1_coco_2017_11_17'
detection_model = load_model(model_name)


while(True):
    # read the frame
    check, frame = vid_stream.read()

    if check:
        cv2.imshow("camera capture", cv2.resize(frame_obj_detect(detection_model, frame), (800, 600)))
    else:
        break   # exit loop if frame is not captured

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Clear the resources
vid_stream.release()
cv2.destroyAllWindows()
