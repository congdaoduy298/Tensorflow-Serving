# Tensorflow Serving

## I. INSTALLATION

- Install Tensorflow Serving using Docker (Recommended)
    
   - Docker installation instructions are [on the Docker site](https://docs.docker.com/get-docker/).

   - After install Docker run the following code to get TFS 

    ```shell
    docker pull tensorflow/serving
    ```

- Install Tensorflow Serving [without Docker](https://github.com/tensorflow/serving/blob/master/tensorflow_serving/g3doc/setup.md) (Not Recommended)


## II. SERVE A MODEL

### 1. Serving an example

#### With Docker :

We will use a toy model called `Half Plus Two`, which generates `0.5 * x +
2` for the values of `x` we provide for prediction.

1. Update TensorFlow Serving.

	First pull the serving image:

```shell
docker pull tensorflow/serving
```
2. Create a folder to clone the TFS repo.

```shell
mkdir -p /tmp/tfserving
cd /tmp/tfserving
git clone https://github.com/tensorflow/serving
```

3. Serve a TensorFlow Model

	Next, run the TensorFlow Serving container pointing it to this model and opening
the REST API port (8501):

	```shell
	docker run -p 8501:8501 \
	  --mount type=bind,\
	source=/tmp/tfserving/serving/tensorflow_serving/servables/tensorflow/testdata/saved_model_half_plus_two_cpu,\
	target=/models/half_plus_two \
	  -e MODEL_NAME=half_plus_two -t tensorflow/serving &
	```

	This will run the docker container and launch the TensorFlow Serving Model
Server, bind the REST API port 8501, and map our desired model from our host to
where models are expected in the container. We also pass the name of the model
as an environment variable, which will be important when we query the model.

4. Get predictions from a model

	To query the model using the predict API, you can run

	```shell
	curl -d '{"instances": [0.0, 1.0, 2.0]}' \
	  -X POST http://localhost:8501/v1/models/half_plus_two:predict
	```

	To get a prediction by [python file](send_request.py), you run this:

	```python
	python send_request.py
	```

	This should return a set of values:

	```json
	{ "predictions": [2.0, 2.5, 3.0] }
	```
#### With ModelServer :

If you are currently running the server, first you need to turn it off. 

```bash
fuser -k 8501/tcp
```

And check whether the port is really turned off:

```bash
lsof -i -P -n | grep LISTEN
```

If the port is turned off, we'll serve the model by running only one command like this:

```shell
tensorflow_model_server --port=8500 --rest_api_port=8501 --model_name=haf_plus_two      
                        --model_base_path=/tmp/tfserving/serving/tensorflow_serving/servables/tensorflow/testdata/saved_model_half_plus_two_cpu
```

After you execute the tensorflow_model_server command above, ModelServer runs on your host, listening for inference requests. You can [get 
predictions](#4get_predictions_from_a_model). 

##### Arguments: 
```
    --port=8500                         TCP port to listen on for gRPC/HTTP API. Disabled if port set to zero.

    --rest_api_port=8501   	            Port to listen on for HTTP/REST API. If set to zero HTTP/REST API will not be exported. 
                                        This port must be different than the one specified in --port.

    --rest_api_num_threads=16        	Number of threads for HTTP/REST API processing. If not set, will be auto set based on number of CPUs.

    --rest_api_timeout_in_ms=30000      Timeout for HTTP/REST API calls.

    --grpc_max_threads=16               Max grpc server threads to handle grpc messages.

    --model_config_file=""              Use this for serving multiple models and choosing a specific model version. 
                                        (If used, --model_name, --model_base_path are ignored.)

    --model_config_file_poll_wait_seconds=0 
                                        Read model_config_file periodically after model_config_file_poll_wait_seconds seconds. 
                                        If unset or set to zero, poll will be done exactly once and not periodically.
    
    --model_name="default"              Name of model (ignored if --model_config_file flag is set)

    --model_base_path=""             	Path to export (ignored if --model_config_file flag is set, otherwise required)

    --enable_batching=false             Enable batching

    --batching_parameters_file=""       Configure batching parameters

    --max_num_load_retries=5            Maximum number of times it retries loading a model after the first failure, before giving up.

    --num_load_threads=0                The number of threads in the thread-pool used to load servables.

    --num_unload_threads=0              The number of threads in the thread-pool used to unload servables. 

    --monitoring_config_file=""         Configure monitoring.
```

#### Note: 
-   Docker use this command inside it so changing the above params we need to pass `-e PARAM=VALUE`.

As we can see before, `-e MODEL_NAME=half_plus_two` was used in Docker command .

### 2. Serve a custom model

After training model, you have to save model to SavedModel format.

- A SavedModel is a directory containing serialized signatures and the state needed to run them, including variable values and vocabularies.

Use `tf.saved_model` API to save and load a model:

- Save: ```tf.saved_model.save(model, path_to_dir)```

- Load: ```model = tf.saved_model.load(path_to_dir)```

`path_to_dir` includes a number folder that represents for the version of model.

Ex: ```path_to_dir = /tmp/tmpfcgkddlh/mobilenet/1/```

You will see these things in `1` folder:

```bash
assets  saved_model.pb  variables
```

- The assets directory contains files used by the TensorFlow graph, for example text files used to initialize vocabulary tables.

- The saved_model.pb file stores the actual TensorFlow program, or model, and a set of named signatures, each identifying a function that accepts tensor inputs and produces tensor outputs.

- The variables directory contains a standard training checkpoint.

So you can serve your custom model:

```shell
tensorflow_model_server --port=8500 --rest_api_port=8501 --model_name=mobilenet      
                        --model_base_path=/tmp/tmpfcgkddlh/mobilenet
```

For more information about [SavedModel](https://www.tensorflow.org/guide/saved_model).

### 3. Multiple model serving

To serve multiple model, you'll configure a config file to pass through the model_config_file parameter.

Here an example config file:
```
model_config_list: {
	config:{
		name: "half_plus_two",
		base_path: "/tmp/tfserving/serving/tensorflow_serving/servables/tensorflow/testdata/saved_model_half_plus_two_cpu",
		model_platform: "tensorflow"
        model_version_policy {
					  specific {
					    versions: 123
					  }
					}
	},
	config:{
		name: "half_plus_three",
		base_path: "/tmp/tfserving/serving/tensorflow_serving/servables/tensorflow/testdata/saved_model_half_plus_three",
		model_platform: "tensorflow"
	}
}
```

It will load version 123 of the half_plus_two model. If model_version_policy not used, the lastest model will be chosen.

### 4. Serving an object detection model

I download SSD ResNet101 V1 FPN 640x640 (RetinaNet101) from [TensorFlow 2 Detection Model Zoo](https://github.com/tensorflow/models/blob/master/research/object_detection/g3doc/tf2_detection_zoo.md).

First, check SignatureDefs of the model:

```bash
!saved_model_cli show --dir /home/user/Desktop/TF-ServingOD/ssd_resnet50_v1_fpn_shared_box_predictor_640x640_coco14_sync_2018_07_03/saved_model/1 --all
```

`--dir` Path of saved_model.pb and variables folder.

You will see the following result:

```bash
signature_def['serving_default']:
  The given SavedModel SignatureDef contains the following input(s):
    inputs['inputs'] tensor_info:
        dtype: DT_UINT8
        shape: (-1, -1, -1, 3)
        name: image_tensor:0
  The given SavedModel SignatureDef contains the following output(s):
    outputs['detection_boxes'] tensor_info:
        dtype: DT_FLOAT
        shape: (-1, 100, 4)
        name: detection_boxes:0
    outputs['detection_classes'] tensor_info:
        dtype: DT_FLOAT
        shape: (-1, 100)
        name: detection_classes:0
    outputs['detection_scores'] tensor_info:
        dtype: DT_FLOAT
        shape: (-1, 100)
        name: detection_scores:0
    outputs['num_detections'] tensor_info:
        dtype: DT_FLOAT
        shape: (-1)
        name: num_detections:0
  Method name is: tensorflow/serving/predict
```

Result tell us what input/output shape is. And this model will be served on Predict API. 

Input: A batch of RGB images (TensorFlow* models were trained with images in RGB order) 

Ouput:

- Detection Boxes: A batch of 100 bounding boxes (Format: xmin, xmax, ymin, ymax).

- Detection Classes: A batch of 100 numbers represents classes.

- Detection Scores: A batch of 100 scores of bounding boxes.

- Number Detections: A batch of number detections in images.

Add to config file:

```
config:{
		name: "detection",
		base_path: "/home/user/Desktop/TF-ServingOD/ssd_resnet50_v1_fpn_shared_box_predictor_640x640_coco14_sync_2018_07_03/saved_model",
		model_platform: "tensorflow"
	}
```

To serve SSD ResNet50:

```bash
tensorflow_model_server --rest_api_port=8501 --model_config_file=models.conf
```

Get a prediction by sending requests from [a Python file](send_detection_request.py).

```bash
python send_detection_request.py
```

## III. LOGGING AND MONITORING

To write logs into a text file, you need :

- Set environment variable TF_CPP_VMODULE.

```bash
export TF_CPP_VMODULE=http_server=1
```

- Run tensorflow_model_server with write file command.

```bash
tensorflow_model_server --rest_api_port=8501 --model_config_file=models.conf &> log &
```



