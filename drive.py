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
import cv2

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

            print('{:10.4f} {:10.4f} {:10.4f}'.format(steering_angle, throttle, speed))
			
			# Gửi lại dữ liệu về góc lái, tốc độ cho phần mềm để ô tô tự lái
            send_control(steering_angle, throttle)
        except Exception as e:
            print(e)

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

    args = parser.parse_args()

    # Dùng GPU/CPU
    device = torch.device("cuda" if torch.cuda.is_available() else 'cpu') 
    print("Using", device)

    # Load model mà ta đã train được từ bước trước
    
    model = Regressor(
                    n_classes = 1,
                    optim_params = {'lr': 1e-3},
                    criterion= MSELoss(), 
                    optimizer= torch.optim.Adam,
                    device = device)
    
    if args.model is not None:
        load_checkpoint(model, args.model)

    # wrap Flask application with engineio's middleware
    app = socketio.Middleware(sio, app)

    # deploy as an eventlet WSGI server
    eventlet.wsgi.server(eventlet.listen(('', 4567)), app)