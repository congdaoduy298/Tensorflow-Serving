import requests
import json
import numpy as np
import cv2 
import time 

def make_prediction(instances):
   headers = {"content-type": "application/json"}
   data = json.dumps({"signature_name": "serving_default", "instances": instances.tolist()})
   start = time.time()
   json_response = requests.post('http://localhost:8501/v1/models/detection:predict', data=data, headers=headers)
   print(f"Total response time: {time.time() - start}s")
   return json.loads(json_response.text)['predictions']

def read_image_from_cloud(url):
   response = requests.get(url)
   file_bytes = np.asarray(bytearray(response.content), dtype=np.uint8)
   image = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
   image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
   return image

def read_image_from_local(path):
   # Read A Local Image
   image = cv2.imread(path)
   image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
   return image 

image1 = read_image_from_cloud('https://i.ytimg.com/vi/MPV2METPeJU/maxresdefault.jpg')
image1 = np.expand_dims(image1, axis=0)

image2 = read_image_from_local('./images/dog.jpg')
image2 = np.expand_dims(image2, axis=0)

x_test = np.concatenate((image1, image2), axis=0)
predictions = make_prediction(x_test)

print(predictions)
