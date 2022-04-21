#!/usr/bin/env python
# coding: utf-8

# In[ ]:





# In[1]:


import onnxruntime as rt
import time
import cv2
IMAGE_ORDERING = "channels_last"
font = cv2.FONT_HERSHEY_SIMPLEX

# In[2]:


sess = rt.InferenceSession("./vgg_unet_im_seg_base_ep20.onnx", providers=['CUDAExecutionProvider'])


# In[3]:


input_name = sess.get_inputs()[0].name


# In[ ]:


print(input_name)


# In[4]:


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


# In[5]:


n_classes = 50
input_height = 320
input_width = 640
output_height = 160
output_width = 320


# In[7]:


import cv2
inp = cv2.imread("dataset1/images_prepped_test/0016E5_07965.png",1)


# In[8]:


import numpy as np

IMAGE_ORDERING = "channels_last"
x = get_image_array(inp, input_width, input_height,
                    ordering=IMAGE_ORDERING)


# In[ ]:




# In[9]:





# In[10]:


label_name = sess.get_outputs()[0].name


# In[11]:




#detections = sess.run(output_names, {input_name: x})
'''
t1 = time.time()

t2 = time.time()
print('Inference time ', str(np.round((t2 - t1), 2)))
'''


def inference_from_video():
    cap = cv2.VideoCapture('/home/mohan/git/backups/drive.mp4')
    prev_frame_time = 0 #calculating prev frame time, ref: https://www.geeksforgeeks.org/python-displaying-real-time-fps-at-which-webcam-video-file-is-processed-using-opencv/
    new_frame_time = 0 # calculating new frame time
    while cap.isOpened():
        ret, frame = cap.read()
        x = get_image_array(frame, 640, 320,
                    ordering=IMAGE_ORDERING)
        input_tensor= np.array(np.expand_dims(x,0), dtype=np.float32)
        detections = sess.run([label_name], {input_name: input_tensor})
        arr = np.asarray(detections)
        arr1 = np.squeeze(arr, axis=0)
        arr2 = np.squeeze(arr1, axis=0)
        pr_p1 = arr2.reshape((output_height,  output_width, n_classes)).argmax(axis=2)
        pred_image = 255 * pr_p1.squeeze()
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


if __name__== '__main__':
    try:
        inference_from_video()
    except BaseException as err:
        cv2.destroyAllWindows()
        raise err