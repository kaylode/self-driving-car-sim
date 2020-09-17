import argparse
import base64
from datetime import datetime
import os
import shutil
import numpy as np
import socketio
import eventlet
import eventlet.wsgi
from PIL import Image
from flask import Flask
from io import BytesIO

import torch
import torchvision.transforms as tf
from models.regressor import Regressor
from trainer.checkpoint import load_checkpoint
from losses.mseloss import MSELoss

val_transforms = tf.Compose([
    tf.ToTensor(),
    tf.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

#initialize our server
sio = socketio.Server()
#our flask (web) app
app = Flask(__name__)
#init our model and image array as empty
model = None
prev_image_array = None

# Tốc độ tối thiểu và tối đa của xe
MAX_SPEED = 25
MIN_SPEED = 10

# Tốc độ thời điểm ban đầu
speed_limit = MAX_SPEED

#registering event handler for the server
@sio.on('telemetry')
def telemetry(sid, data):
    if data:
        # Lấy giá trị throttle hiện tại
        throttle = float(data["throttle"])
        # Góc lái hiện tại của ô tô
        steering_angle = float(data["steering_angle"])
    	  # Tốc độ hiện tại của ô tô
        speed = float(data["speed"])
        # Ảnh từ camera giữa
        image = Image.open(BytesIO(base64.b64decode(data["image"])))
        try:
			# Tiền xử lý ảnh, cắt, reshape
            image = val_transforms(image)

            print('*****************************************************')
            steering_angle = float(model.inference_img(image))
            
			# Tốc độ ta để trong khoảng từ 10 đến 25
            global speed_limit
            if speed > speed_limit:
                speed_limit = MIN_SPEED  # giảm tốc độ
            else:
                speed_limit = MAX_SPEED
            throttle = 1.0 #- steering_angle**2 - (speed/speed_limit)**2

            print('{} {} {}'.format(steering_angle, throttle, speed))
			
			# Gửi lại dữ liệu về góc lái, tốc độ cho phần mềm để ô tô tự lái
            send_control(steering_angle, throttle)
        except Exception as e:
            print(e)

        # save frame
        if args.image_folder != '':
            timestamp = datetime.utcnow().strftime('%Y_%m_%d_%H_%M_%S_%f')[:-3]
            image_filename = os.path.join(args.image_folder, timestamp)
            image.save('{}.jpg'.format(image_filename))
    else:
        
        sio.emit('manual', data={}, skip_sid=True)


@sio.on('connect')
def connect(sid, environ):
    print("connect ", sid)
    send_control(0, 0)


def send_control(steering_angle, throttle):
    sio.emit(
        "steer",
        data={
            'steering_angle': steering_angle.__str__(),
            'throttle': throttle.__str__()
        },
        skip_sid=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Remote Driving')
    parser.add_argument(
        '--model',
        default=None,
        type=str,
        help='Path to model h5 file. Model should be on the same path.'
    )
    parser.add_argument(
        '--image_folder',
        type=str,
        nargs='?',
        default='',
        help='Path to image folder. This is where the images from the run will be saved.'
    )
    args = parser.parse_args()

    # Load model mà ta đã train được từ bước trước
    device = torch.device("cuda")
    print("Using", device)
    criterion = MSELoss()
    optimizer = torch.optim.Adam
    model = Regressor(
                    n_classes = 1,
                    optim_params = {'lr': 1e-3},
                    criterion= criterion, 
                    optimizer= optimizer,
                    device = device)
    load_checkpoint(model, 'weights/udacity/ResNet34_30.pth')

    if args.image_folder != '':
        print("Creating image folder at {}".format(args.image_folder))
        if not os.path.exists(args.image_folder):
            os.makedirs(args.image_folder)
        else:
            shutil.rmtree(args.image_folder)
            os.makedirs(args.image_folder)
        print("RECORDING THIS RUN ...")
    else:
        print("NOT RECORDING THIS RUN ...")

    # wrap Flask application with engineio's middleware
    app = socketio.Middleware(sio, app)

    # deploy as an eventlet WSGI server
    eventlet.wsgi.server(eventlet.listen(('', 4567)), app)