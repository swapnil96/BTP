#! /usr/bin/env python2

from mvnc import mvncapi as mvnc
from matplotlib import pyplot as plt
import numpy, cv2
import sys, os
import cPickle as pickle, time

def whiten_image(source_image):
    source_mean = numpy.mean(source_image)
    source_standard_deviation = numpy.std(source_image)
    std_adjusted = numpy.maximum(source_standard_deviation, 1.0 / numpy.sqrt(source_image.size))
    whitened_image = numpy.multiply(numpy.subtract(source_image, source_mean), 1 / std_adjusted)
    return whitened_image

def preprocess_image(src):
    # scale the image
    NETWORK_WIDTH = 300
    NETWORK_HEIGHT = 300
    preprocessed_image = cv2.resize(src, (NETWORK_WIDTH, NETWORK_HEIGHT))

    #convert to RGB
    preprocessed_image = cv2.cvtColor(preprocessed_image, cv2.COLOR_BGR2RGB)

    #whiten
    preprocessed_image = whiten_image(preprocessed_image)

    # return the preprocessed image
    return preprocessed_image

def transform_ncs_output(img, output, conf_thresh=0.5):
    """Get image and NCS output
    Return: boxes, classes, confidences"""
    num = int(output[0])
    scale = numpy.array([img.shape[1],img.shape[0],img.shape[1],img.shape[0]])
    t = [((e[3:].clip(0,1)*scale).astype(int), int(e[1]), e[2]) for e in output.reshape((-1,7))[1:] if numpy.all(numpy.isfinite(e)) ][:num]
    #glitchy predictions with 1.0 confidence and 0 size in one/both dimensions appear sometimes
    #filter them too
    t = [e for e in t if (e[2]>conf_thresh) and (e[0][0]-e[0][2])*(e[0][1]-e[0][3])>0]
    return (zip(*t) if len(t)>0 else ([],[],[]))

def run_inference(image, facenet_graph):

    resized_image = preprocess_image(image)
    # ***************************************************************
    # Send the image to the NCS
    # ***************************************************************
    facenet_graph.LoadTensor(resized_image.astype(numpy.float16), None)

    # ***************************************************************
    # Get the result from the NCS
    # ***************************************************************
    output, userobj = facenet_graph.GetResult()

    return output
   
def run_image(inference_output, img):
    boxes, classes, confidence = transform_ncs_output(img, inference_output, 0.25)
    ans = []
    for box, cls, conf in zip(boxes, classes, confidence):
        #cv2.rectangle(img, (box[0], box[1]), (box[2], box[3]), (0,255,0), 2)
		temp = img[box[1]:box[3], box[0]:box[2]]
		ans.append(temp)

    #plt.imshow(img)
    #plt.show()
    return ans

# main entry point for program. we'll call main() to do what needs to be done.
if __name__ == "__main__":
    sys.exit(train())

