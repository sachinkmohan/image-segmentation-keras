#from keras_segmentation.models.unet import vgg_unet
from tensorflow.keras.models import load_model
from keras_segmentation.custom_evaluate4models import evaluate
import tensorflow_model_optimization as tfmot

#model = vgg_unet(n_classes=50 ,  input_height=320, input_width=640  )
#model = load_model('./custom_model_files/striped_divam_ss_pruned_ep_model_20_new.h5')
'''
model.train(
    train_images =  "dataset1/images_prepped_train/",
    train_annotations = "dataset1/annotations_prepped_train/",
    checkpoints_path = "/tmp/vgg_unet_1" , epochs=1
)
'''

#model.load_weights('./custom_model_files/divam_ss_base_weights_ep20.h5')

from tensorflow.keras.models import load_model

#model = vgg_unet(n_classes=50 ,  input_height=320, input_width=640  )
# Pruned model loading
#model_p = load_model('./custom_model_files/striped_divam_ss_pruned_ep_model_20_new.h5')

#Quantized model loading

quantize_scope = tfmot.quantization.keras.quantize_scope
with quantize_scope():
    model_qq=load_model('./divam_ss_q_model_.19.h5')

#print(evaluate(model_p, inp_images_dir="dataset1/images_prepped_test/"  , annotations_dir="dataset1/annotations_prepped_test/" ))
print(evaluate(model_qq, inp_images_dir="dataset1/images_prepped_test/"  , annotations_dir="dataset1/annotations_prepped_test/" ))



