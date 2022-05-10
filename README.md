[Old README file](./README2.md)

Changes in this repository
 - `testing_end_to_end_modified.ipynb`
 - `testing_end_to_end_pruning_modified.ipynb` - pruning + tflite inference
 - Added a [Custom evaluation for pruned and quantized](./keras_segmentation/custom_evaluate4models.py) file, which can be evaluated from this [file](./sample_train.py)
 - Added a `custom_evaluate.py`
 - Added a `convert2onnx.py`
 - Weights are backed up to [dropbox](https://www.dropbox.com/sh/xtbltwge93am6u3/AABI4zv8_q426izlL1WXJUQSa?dl=0)
 - `testing_end_to_end_quantization.ipynb` - quantization tests
 - `vgg_unet_im_seg_base_ep20.onnx` - onnx file
 - `sample_train.py` - for quick training
 - `evaluate_custom.py` - for evaluating pruned and quantized models
 - `custom_inference_image_video.py` - Inference on a single image or a video file
 - `inference_onnx.py` - Inference on the onnx file


For converting to onnx, install tf2onnx, refer requirements.txt for the version.
For inference use mltf115_5(conda env, self reference)

[TensorRT Inference](https://github.com/sachinkmohan/Jetson_test_projects/blob/main/Image_classification_nd/optimized/divamgupta-isk-inference.py)

