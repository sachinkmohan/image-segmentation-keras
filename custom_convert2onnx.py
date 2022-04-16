from keras_segmentation.models.unet import vgg_unet

import tf2onnx
#import onnxruntime as rt
import tensorflow as tf




def main():
    model = vgg_unet(n_classes=50 ,  input_height=320, input_width=640  )
    model.load_weights('./custom_model_files/divam_ss_base_weights_ep20.h5')

    spec = (tf.TensorSpec((None, 320, 640, 3), tf.float32, name="input"),)
    # output_path = semantic_model.name + ".onnx"
    output_path = "vgg_unet_im_seg_base_ep20" + ".onnx"

    model_proto, _ = tf2onnx.convert.from_keras(model, input_signature=spec, opset=13, output_path=output_path)
    output_names = [n.name for n in model_proto.graph.output]
    print(output_names)
    print('done')


if __name__ == "__main__":
    main()