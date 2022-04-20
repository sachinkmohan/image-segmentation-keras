import os
import six
import numpy as np
import cv2
import tensorflow as tf
import time

IMAGE_ORDERING = "channels_last"
font = cv2.FONT_HERSHEY_SIMPLEX
pruned_tflite_model2='./pruning_r_segnet.tflite'
with open(pruned_tflite_model2, 'rb') as fid:
    tflite_model = fid.read()
 # Load TFLite model and allocate tensors.
interpreter = tf.lite.Interpreter(model_content=tflite_model)
interpreter.allocate_tensors()
#get input and output tensors
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()
input_index = interpreter.get_input_details()[0]["index"]
output_index = interpreter.get_output_details()[0]["index"]


def get_image_array(image_input,
                    width, height,
                    imgNorm="sub_mean", ordering='channels_first', read_image_type=1):
    """ Load image array from input """

    if type(image_input) is np.ndarray:
        # It is already an array, use it as it is
        img = image_input
    elif isinstance(image_input, six.string_types):
        if not os.path.isfile(image_input):
            raise DataLoaderError("get_image_array: path {0} doesn't exist"
                                  .format(image_input))
        img = cv2.imread(image_input, read_image_type)
    else:
        raise DataLoaderError("get_image_array: Can't process input type {0}"
                              .format(str(type(image_input))))

    if imgNorm == "sub_and_divide":
        img = np.float32(cv2.resize(img, (width, height))) / 127.5 - 1
    elif imgNorm == "sub_mean":
        img = cv2.resize(img, (width, height))
        img = img.astype(np.float32)
        img = np.atleast_3d(img)

        means = [103.939, 116.779, 123.68]

        for i in range(min(img.shape[2], len(means))):
            img[:, :, i] -= means[i]

        img = img[:, :, ::-1]
    elif imgNorm == "divide":
        img = cv2.resize(img, (width, height))
        img = img.astype(np.float32)
        img = img/255.0

    if ordering == 'channels_first':
        img = np.rollaxis(img, 2, 0)
    return img

def inference_image():
    pass


def backbone_tflite(x):
    input_tensor= np.array(np.expand_dims(x,0), dtype=np.float32)
    interpreter.set_tensor(input_index, input_tensor)

    t1 = time.time()
    #Run the inference
    interpreter.invoke()
    t2 = time.time()
    print('Inference time is ', str(np.round((t2 - t1), 2)))
    output_details = interpreter.get_output_details()
    prediction = interpreter.get_tensor(output_details[0]['index'])[0]
    pr_p1 = prediction.reshape((160,  320, 50)).argmax(axis=2)
    return pr_p1



def inference_from_video():
    cap = cv2.VideoCapture('/home/mohan/git/backups/drive.mp4')
    prev_frame_time = 0 #calculating prev frame time, ref: https://www.geeksforgeeks.org/python-displaying-real-time-fps-at-which-webcam-video-file-is-processed-using-opencv/
    new_frame_time = 0 # calculating new frame time
    while cap.isOpened():
        ret, frame = cap.read()
        x = get_image_array(frame, 640, 320,
                    ordering=IMAGE_ORDERING)
        pr = backbone_tflite(x)
        pred_image = 255 * pr.squeeze()
        u8 = pred_image.astype(np.uint8)
        im_color = cv2.applyColorMap(u8, cv2.COLORMAP_TURBO)

        new_frame_time = time.time() #time taken to finish processing this frame

        #calculating fps
        fps = 1/(new_frame_time - prev_frame_time)
        prev_frame_time = new_frame_time
        fps = int(fps)
        fps = str(fps) #converted to string to display it on the frame
        cv2.putText(im_color, fps, (7,70), font, 3,  (100, 250, 0), 3, cv2.LINE_AA)
        cv2.imshow('im', im_color)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            cv2.destroyAllWindows()
            break


def inference_from_image(y):
    start = time.time()
    pre_pro_img = get_image_array(y, 640, 320,
                    ordering=IMAGE_ORDERING)
    pr = inference_tflite(pre_pro_img)
    end = time.time()
    pred_image = 255 * pr.squeeze()
    u8 = pred_image.astype(np.uint8)
    im_color = cv2.applyColorMap(u8, cv2.COLORMAP_TURBO)
    #print('Inference time is ', str(np.round((end - start), 2)))
    while(True):
        cv2.imshow('im', im_color)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            cv2.destroyAllWindows()
            break



if __name__ == '__main__':
    try:
        inp = cv2.imread("dataset1/images_prepped_test/0016E5_07965.png",1)
        inference_from_image(inp)
        #inference_from_video()
    except BaseException as err:
        # logger.error(err)
        cv2.destroyAllWindows()
        raise err