import requests
import json
import time 

def make_prediction(instances):
   headers = {"content-type": "application/json"}
   data = json.dumps({"signature_name": "serving_default", "instances": instances})
   start = time.time()
   json_response = requests.post('http://localhost:8501/v1/models/half_plus_two:predict', data=data, headers=headers)
   print(json_response.text)
   print(f"Total response time: {time.time() - start}s")


x_test = [0, 1, 2]
predictions = make_prediction(x_test)
