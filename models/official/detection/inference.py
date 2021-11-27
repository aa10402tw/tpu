# Copyright 2020 The TensorFlow Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
# pylint: disable=line-too-long
r"""A stand-alone binary to run model inference and visualize results.

It currently only supports model of type `retinanet` and `mask_rcnn`. It only
supports running on CPU/GPU with batch size 1.
"""
# pylint: enable=line-too-long



from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import base64
import csv
import io
import os
import cv2

from absl import flags
from absl import logging

import numpy as np
from PIL import Image
import tensorflow.compat.v1 as tf

from configs import factory as config_factory
from dataloader import mode_keys
from modeling import factory as model_factory
from utils import box_utils
from utils import input_utils
from utils import mask_utils
from utils.object_detection import visualization_utils
from hyperparameters import params_dict

from tqdm import tqdm
import os
os.environ["CUDA_VISIBLE_DEVICES"]="0"


FLAGS = flags.FLAGS

flags.DEFINE_string(
    'model', 'retinanet', 'Support `retinanet`, `mask_rcnn` and `shapemask`.')
flags.DEFINE_integer('image_size', 640, 'The image size.')
flags.DEFINE_string(
    'checkpoint_path', '', 'The path to the checkpoint file.')
flags.DEFINE_string(
    'config_file', '', 'The config file template.')
flags.DEFINE_string(
    'params_override', '', 'The YAML file/string that specifies the parameters '
    'override in addition to the `config_file`.')
flags.DEFINE_string(
    'label_map_file', '',
    'The label map file. See --label_map_format for the definition.')
flags.DEFINE_string(
    'label_map_format', 'csv',
    'The format of the label map file. Currently only support `csv` where the '
    'format of each row is: `id:name`.')
flags.DEFINE_string(
    'image_file_pattern', '',
    'The glob that specifies the image file pattern.')
flags.DEFINE_string(
    'output_html', '/tmp/test.html',
    'The output HTML file that includes images with rendered detections.')
flags.DEFINE_string(
    'output_dir', './_output',
    'The output folder.')
flags.DEFINE_integer(
    'max_boxes_to_draw', 10, 'The maximum number of boxes to draw.')
flags.DEFINE_float(
    'min_score_threshold', 0.05,
    'The minimum score thresholds in order to draw boxes.')


def main(unused_argv):
  del unused_argv
  # Load the label map.
  print(' - Loading the label map...')
  label_map_dict = {}
  if FLAGS.label_map_format == 'csv':
    with tf.gfile.Open(FLAGS.label_map_file, 'r') as csv_file:
      reader = csv.reader(csv_file, delimiter=':')
      for row in reader:
        if len(row) != 2:
          raise ValueError('Each row of the csv label map file must be in '
                           '`id:name` format.')
        id_index = int(row[0])
        name = row[1]
        label_map_dict[id_index] = {
            'id': id_index,
            'name': name,
        }
  else:
    raise ValueError(
        'Unsupported label map format: {}.'.format(FLAGS.label_mape_format))

  params = config_factory.config_generator(FLAGS.model)
  if FLAGS.config_file:
    params = params_dict.override_params_dict(
        params, FLAGS.config_file, is_strict=True)
  params = params_dict.override_params_dict(
      params, FLAGS.params_override, is_strict=True)
  params.override({
      'architecture': {
          'use_bfloat16': False,  # The inference runs on CPU/GPU.
      },
  }, is_strict=True)
  params.validate()
  params.lock()

  model = model_factory.model_generator(params)

  with tf.Graph().as_default():
    image_input = tf.placeholder(shape=(), dtype=tf.string)
    image = tf.io.decode_image(image_input, channels=3)
    image.set_shape([None, None, 3])

    image = input_utils.normalize_image(image)
    image_size = [FLAGS.image_size, FLAGS.image_size]
    image = tf.image.resize_images(image, image_size, align_corners=True)
    _, image_info = input_utils.resize_and_crop_image(
        image,
        image_size,
        image_size,
        aug_scale_min=1.0,
        aug_scale_max=1.0)
    image.set_shape([image_size[0], image_size[1], 3])

    # batching.
    images = tf.reshape(image, [1, image_size[0], image_size[1], 3])
    images_info = tf.expand_dims(image_info, axis=0)

    # model inference
    outputs = model.build_outputs(images, {'image_info': images_info}, mode=mode_keys.PREDICT)

    # outputs['detection_boxes'] = (
    #     outputs['detection_boxes'] / tf.tile(images_info[:, 2:3, :], [1, 1, 2]))

    predictions = outputs

    # Create a saver in order to load the pre-trained checkpoint.
    saver = tf.train.Saver()

    image_with_detections_list = []
    os.makedirs(FLAGS.output_dir, exist_ok=True)
    gpu_options = tf.GPUOptions(visible_device_list="0")
    with tf.Session(config=tf.ConfigProto(gpu_options=gpu_options)) as sess:
      print(' - Loading the checkpoint...')
      saver.restore(sess, FLAGS.checkpoint_path)
      devices = sess.list_devices()

      image_files = tf.gfile.Glob(FLAGS.image_file_pattern)
      pbar = tqdm(total = len(image_files))
      for i, image_file in enumerate(image_files):
        #print(' - Processing image %d...' % i, os.path.basename(image_file))

        with tf.gfile.GFile(image_file, 'rb') as f:
          image_bytes = f.read()

        image = Image.open(image_file)
        image = image.convert('RGB')  # needed for images with 4 channels.
        image = image.resize(image_size)
        width, height = image.size
        np_image = (np.array(image.getdata()).reshape(height, width, 3).astype(np.uint8))

        predictions_np = sess.run(predictions, feed_dict={image_input: image_bytes})
        pred_logit = predictions_np['logits']
        pred_label = np.argmax(pred_logit, axis=3)
        color_label = label2color(pred_label[0])
        color_label = cv2.cvtColor((color_label*255.0).astype(np.uint8), cv2.COLOR_BGR2RGB)
        #color_label = cv2.resize(color_label, image_size)

        image_name = os.path.basename(image_file).replace('.jpg', '')
        colorlabel_path = f"{FLAGS.output_dir}/{image_name}_colorlabel.png"
        predlogit_path = f"{FLAGS.output_dir}/{image_name}_predlogit.bin"

        cv2.imwrite(colorlabel_path, color_label)
        #cv2.imwrite(f"{FLAGS.output_dir}/{image_name}.png", np_image)
        #pred_logit.tofile(predlogit_path)
        pbar.set_postfix({"Processed": image_name})
        pbar.update()
        if i > 16:
            break

def label2color(label_mask):
    label_colours = np.asarray(
        [
            [0, 0, 0],
            [128, 0, 0],
            [0, 128, 0],
            [128, 128, 0],
            [0, 0, 128],
            [128, 0, 128],
            [0, 128, 128],
            [128, 128, 128],
            [64, 0, 0],
            [192, 0, 0],
            [64, 128, 0],
            [192, 128, 0],
            [64, 0, 128],
            [192, 0, 128],
            [64, 128, 128],
            [192, 128, 128],
            [0, 64, 0],
            [128, 64, 0],
            [0, 192, 0],
            [128, 192, 0],
            [0, 64, 128],
        ]
    )
    r = label_mask.copy()
    g = label_mask.copy()
    b = label_mask.copy()
    for ll in range(0, 21):
        r[label_mask == ll] = label_colours[ll, 0]
        g[label_mask == ll] = label_colours[ll, 1]
        b[label_mask == ll] = label_colours[ll, 2]
    rgb = np.zeros((label_mask.shape[0], label_mask.shape[1], 3))
    rgb[:, :, 0] = r / 255.0
    rgb[:, :, 1] = g / 255.0
    rgb[:, :, 2] = b / 255.0
    return rgb

if __name__ == '__main__':
  flags.mark_flag_as_required('model')
  flags.mark_flag_as_required('checkpoint_path')
  flags.mark_flag_as_required('label_map_file')
  flags.mark_flag_as_required('image_file_pattern')
  #flags.mark_flag_as_required('output_html')
  logging.set_verbosity(logging.INFO)
  tf.app.run(main)
