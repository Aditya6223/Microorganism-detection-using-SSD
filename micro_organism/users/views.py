from django.shortcuts import render
from django.http import HttpResponse
from django.contrib import messages
from .forms import UserRegistrationForm
#from .forms import makeappointmentform
from .models import UserRegistrationModel, UserAppointmentModel
import requests

# import os
# from django.conf import settings
# import pandas as pd

'''
Api call for expert system
Base URl is http://127.0.0.1:8084/
'''
BASEURL = 'http://127.0.0.1:8084/'


# Create your views here.
def home(request):
    return render(request, 'base.html')


def user_register_action(request):
    if request.method == 'POST':
        form = UserRegistrationForm(request.POST)
        if form.is_valid():
            print('Data is Valid')
            form.save()
            messages.success(request, 'You have been successfully registered')
            form = UserRegistrationForm()
            return render(request, 'register.html', {'form': form})
        else:
            messages.success(request, 'Email or Mobile Already Existed')
            print("Invalid form")

    else:
        form = UserRegistrationForm()
    return render(request, 'register.html', {'form': form})


def user_login_check(request):
    if request.method == "POST":
        loginid = request.POST.get('loginid')
        pswd = request.POST.get('password')
        print("Login ID = ", loginid, ' Password = ', pswd)
        try:
            check = UserRegistrationModel.objects.get(
                loginid=loginid, password=pswd)
            status = check.status
            print('Status is = ', status)
            if status == "activated":
                request.session['id'] = check.id
                request.session['loggeduser'] = check.name
                request.session['loginid'] = loginid
                request.session['email'] = check.email
                print("User id At", check.id, status)
                return render(request, 'users/user_home.html', {})
            else:
                messages.success(
                    request, 'Your Account has not been activated by the Admin🛑🤚')
                return render(request, 'user_login.html')
        except Exception as e:
            print('Exception is ', str(e))
            pass
        messages.success(request, 'Invalid Login id and password')
    return render(request, 'user_login.html', {})

def user_home(request):
    return render(request, 'users/user_home.html')

def upload_image(request):
    return render(request, "users/upload.html") 


# from django.shortcuts import render
# from django.core.files.storage import FileSystemStorage
# import cv2
# import os
# from ultralytics import YOLO

# # Load YOLOv8 model once
# model = YOLO("best.pt")  # Update with your trained YOLOv8 model path
#  # Render upload page

# def detect_objects(request):
#     if request.method == "POST" and request.FILES["image"]:
#         uploaded_image = request.FILES["image"]
#         fs = FileSystemStorage()
#         img_path = fs.save(uploaded_image.name, uploaded_image)
#         img_path = fs.url(img_path).lstrip("/")  # Get relative file path

#         # Load image
#         img = cv2.imread(img_path)
#         results = model(img)

#         # Process detection results
#         for result in results:
#             boxes = result.boxes.xyxy  # Bounding boxes
#             confidences = result.boxes.conf  # Confidence scores
#             class_ids = result.boxes.cls  # Class IDs

#             for box, conf, cls_id in zip(boxes, confidences, class_ids):
#                 x1, y1, x2, y2 = map(int, box)
#                 label = f"{model.names[int(cls_id)]}: {conf:.2f}"

#                 # Draw bounding box and label
#                 cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
#                 cv2.putText(img, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

#         # Save detected image
#         detected_img_path = f"media/detected_{uploaded_image.name}"
#         cv2.imwrite(detected_img_path, img)

#         return render(request, "users/result.html", {"image_url": "/" + detected_img_path,'label':label})

#     return render(request, "users/upload.html")

import cv2
import numpy as np
import tensorflow as tf
import os
from django.shortcuts import render
from django.core.files.storage import default_storage
import matplotlib.pyplot as plt
from io import BytesIO
import base64

# Load trained model
MODEL_PATH = "Micro_Organism_model.h5"  # Update with actual path
model = tf.keras.models.load_model(MODEL_PATH)

# Define class labels
class_labels = ["Amoeba", "Rod Bacteria", "Hydra", "Euglean"]

def preprocess_image(image_path, target_size=(224, 224)):
    image = cv2.imread(image_path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image_resized = cv2.resize(image, target_size)
    image_array = np.expand_dims(image_resized, axis=0) / 255.0
    return image, image_array

def draw_boxes(image, class_id):
    h, w, _ = image.shape
    x1, y1, x2, y2 = int(w*0.3), int(h*0.3), int(w*0.7), int(h*0.7)
    
    color = (0, 255, 0)
    label = class_labels[int(class_id)]
    cv2.rectangle(image, (x1, y1), (x2, y2), color, 2)
    cv2.putText(image, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
    
    return image

def detect_objects(request):
    if request.method == 'POST' and request.FILES['image']:
        image_file = request.FILES['image']
        file_path = default_storage.save('temp/' + image_file.name, image_file)
        full_path = os.path.join(default_storage.location, file_path)
        
        original_image, input_image = preprocess_image(full_path)
        predictions = model.predict(input_image)
        class_id = np.argmax(predictions)
        output_image = draw_boxes(original_image, class_id)
        
        # Convert image to base64 for displaying in HTML
        _, buffer = cv2.imencode('.jpg', cv2.cvtColor(output_image, cv2.COLOR_RGB2BGR))
        image_base64 = base64.b64encode(buffer).decode('utf-8')
        
        return render(request, 'users/result.html', {'image_base64': image_base64, 'prediction': class_labels[class_id]})
    
    return render(request, 'users/upload.html')
