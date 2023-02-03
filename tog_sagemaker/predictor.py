# -*- coding: utf-8 -*-
"""
@original_source: naresh.gangiredd
"""

import json
import matplotlib.colors as mcolors

import flask
import boto3
from dataset_utils.preprocessing import letterbox_image_padded
from misc_utils.visualization import visualize_detections
from keras import backend as K
from models.yolov3 import YOLOv3_Darknet53, YOLOv3_Darknet53_ABO, YOLOv3
from PIL import ImageDraw, ImageFont, Image
from tog.attacks import *
import os
import base64
import io

# from boto3.s3.connection import S3Connection
# from botocore.exceptions import ClientError
# import pickle

import logging

EPS = 8 / 255.  # Hyperparameter: epsilon in L-inf norm
EPS_ITER = 2 / 255.  # Hyperparameter: attack learning rate
N_ITER = 10  # Hyperparameter: number of attack iterations

coco_yolov3_weight_fname = 'YOLOv3_Darknet53.h5'
abo_yolov3_weight_fname = 'YOLOv3_Darknet53_ABO.h5'

# Define the path
prefix = '/opt/ml/'
model_path = os.path.join(prefix, 'model')
logging.info("Model Path" + str(model_path))

coco_yolov3_weight_fpath = os.path.join(model_path, coco_yolov3_weight_fname)
abo_yolov3_weight_fpath = os.path.join(model_path, abo_yolov3_weight_fname)

# Load the model components
coco_detector = YOLOv3_Darknet53(weights=coco_yolov3_weight_fpath)
abo_detector = YOLOv3_Darknet53_ABO(weights=abo_yolov3_weight_fpath)
logging.info("Loaded Models: COCO, ABO")

# The flask app for serving predictions
app = flask.Flask(__name__)


def read_image(img_str: str) -> Image:
    decoded = base64.b64decode(img_str)
    iob_decoded = io.BytesIO(decoded)
    return Image.open(iob_decoded)


def find_font_size(text, font, image, target_width_ratio):
    tested_font_size = 100
    tested_font = ImageFont.truetype(font, tested_font_size)
    observed_width, observed_height = get_text_size(text, image, tested_font)
    estimated_font_size = tested_font_size / (observed_width / image.width) * target_width_ratio
    return round(estimated_font_size)


def get_text_size(text, image, font):
    im = Image.new('RGB', (image.width, image.height))
    draw = ImageDraw.Draw(im)
    return draw.textsize(text, font)


class AdversarialExample:
    def __init__(self,
                 title,
                 image):
        self.title = title
        self.image = image

    def image_to_byte(self):
        img_arr = io.BytesIO()
        self.image.save(img_arr, format=self.image.format)
        img_arr = img_arr.getvalue()
        return img_arr

    def image_to_byte_string(self):
        return base64.encodebytes(self.image_to_byte()).decode('utf-8')

    def to_dict_json_ready(self):
        return {
            'img': self.image_to_byte_string(),
            'title': self.title
        }


def draw_images_with_detections(detections_dict):
    colors = list(mcolors.CSS4_COLORS.values())
    images_dict = {}
    width_ratio = 0.2  # Portion of the image the text width should be (between 0 and 1)
    font_family = "arial.ttf"

    for pid, title in enumerate(detections_dict.keys()):
        input_img, detections, model_img_size, classes = detections_dict[title]
        #         print((input_img.reshape(x_query.shape[1:])*255).astype(np.uint8))
        input_img = Image.fromarray((input_img.reshape(input_img.shape[1:]) * 255).astype(np.uint8))
        img_draw_context = ImageDraw.Draw(input_img)

        for box in detections:
            xmin = max(int(box[-4] * input_img.size[0] / model_img_size[1]), 0)
            ymin = max(int(box[-3] * input_img.size[1] / model_img_size[1]), 0)
            xmax = min(int(box[-2] * input_img.size[0] / model_img_size[1]), input_img.size[0])
            ymax = min(int(box[-1] * input_img.size[1] / model_img_size[1]), input_img.size[1])
            color = colors[int(box[0])]
            label = '{}: {:.2f}'.format(classes[int(box[0])], box[1])
            font_size = find_font_size(label, font_family, input_img, width_ratio)
            font = ImageFont.truetype(font_family, font_size)
            img_draw_context.rectangle(xy=(xmin, ymin, xmax, ymax), outline=color, width=4)
            img_draw_context.text(xy=(xmin, ymin), text=label, font=font)
        images_dict[title] = AdversarialExample(title=title, image=input_img)
    return images_dict


@app.route('/ping', methods=['GET'])
def ping():
    # Check if the classifier was loaded correctly
    try:
        # regressor
        status = 200
        logging.info("Status : 200")
    except:
        status = 400
    return flask.Response(response=json.dumps(' '), status=status, mimetype='application/json')


@app.route('/invocations', methods=['POST'])
def transformation():
    # Get input JSON data and convert it to a DF
    input_json = flask.request.get_json()
    img = read_image(input_json['input']['img'])
    prediction_model = input_json['input']['prediction_model']
    # TODO: make factory for this, because in future you are going to add more
    detector = coco_detector if prediction_model == "COCO" else abo_detector

    x_query, x_meta = letterbox_image_padded(img, size=detector.model_img_size)

    detections_query = detector.detect(x_query, conf_threshold=detector.confidence_thresh_default)

    x_adv_fabrication = tog_fabrication(victim=detector, x_query=x_query, n_iter=N_ITER, eps=EPS, eps_iter=EPS_ITER)
    x_adv_mislabeling_ml = tog_mislabeling(victim=detector, x_query=x_query, target='ml', n_iter=N_ITER, eps=EPS,
                                           eps_iter=EPS_ITER)
    x_adv_mislabeling_ll = tog_mislabeling(victim=detector, x_query=x_query, target='ll', n_iter=N_ITER, eps=EPS,
                                           eps_iter=EPS_ITER)
    x_adv_vanishing = tog_vanishing(victim=detector, x_query=x_query, n_iter=N_ITER, eps=EPS, eps_iter=EPS_ITER)
    x_adv_untargeted = tog_untargeted(victim=detector, x_query=x_query, n_iter=N_ITER, eps=EPS, eps_iter=EPS_ITER)

    detections_adv_fabrication = detector.detect(x_adv_fabrication, conf_threshold=detector.confidence_thresh_default)
    detections_adv_mislabeling_ml = detector.detect(x_adv_mislabeling_ml,
                                                    conf_threshold=detector.confidence_thresh_default)
    detections_adv_mislabeling_ll = detector.detect(x_adv_mislabeling_ll,
                                                    conf_threshold=detector.confidence_thresh_default)
    detections_adv_untargeted = detector.detect(x_adv_untargeted, conf_threshold=detector.confidence_thresh_default)
    detections_adv_vanishing = detector.detect(x_adv_vanishing, conf_threshold=detector.confidence_thresh_default)

    img_names = [
        'Benign (No Attack)',
        'TOG-fabrication',
        'TOG-mislabeling (ML)',
        'TOG-mislabeling (LL)',
        'TOG-vanishing',
        'TOG-untargeted']

    adversarial_detections_dicts = {
        img_names[0]: (x_query, detections_query, detector.model_img_size, detector.classes),
        img_names[1]: (x_adv_fabrication, detections_adv_fabrication, detector.model_img_size, detector.classes),
        img_names[2]: (x_adv_mislabeling_ml, detections_adv_mislabeling_ml, detector.model_img_size, detector.classes),
        img_names[3]: (x_adv_mislabeling_ll, detections_adv_mislabeling_ll, detector.model_img_size, detector.classes),
        img_names[4]: (x_adv_vanishing, detections_adv_vanishing, detector.model_img_size, detector.classes),
        img_names[5]: (x_adv_untargeted, detections_adv_untargeted, detector.model_img_size, detector.classes)}

    drawn_images_dict = draw_images_with_detections(adversarial_detections_dicts)

    drawn_images_converted_dict = {}
    for k in drawn_images_dict.keys():
        drawn_images_converted_dict[k] = drawn_images_dict[k].to_dict_json_ready()

    result = {
        'output': drawn_images_converted_dict
    }

    result_json = json.dumps(result)
    return flask.Response(response=result_json, status=200, mimetype='application/json')
