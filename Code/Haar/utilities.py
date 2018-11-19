#! /usr/bin/env python2

from mvnc import mvncapi as mvnc
import numpy
import cv2
import sys
import os
import cPickle as pickle
import openface
import time

IMAGES_DIR = '../'

VALID_DATA_DIR = IMAGES_DIR + 'valid_data/'

GRAPH_FILENAME = "../facenet_celeb_ncs.graph"

# the same face will return 0.0
# different faces return higher numbers
# this is NOT between 0.0 and 1.0
FACE_MATCH_THRESHOLD = 1.2

face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

def detect_face(bgrImg, dim = 160, multipleFaces = False):
    if bgrImg is None:
        raise Exception("Unable to load image/frame")

    gray = cv2.cvtColor(bgrImg, cv2.COLOR_BGR2GRAY)

    if (multipleFaces):
        faces = face_cascade.detectMultiScale(gray, 1.3, 5)
        final_images = []
        for (x, y, w, h) in faces:
            image = bgrImg[y:y+h, x:x+w]
            # cv2.imshow('img', image)
            # cv2.waitKey(0)
            # cv2.destroyAllWindows()
            scaled_image = cv2.resize(image, (dim, dim))
            final_images.append(scaled_image)

        return final_images

    else:
        # Get the largest face bounding box
        start = time.time()
        faces = face_cascade.detectMultiScale(gray, 1.3, 5)
        #cv2.imshow('img', gray)
        #cv2.waitKey(0)
        #cv2.destroyAllWindows()
        if len(faces) == 0:
            print ("No face detected")
            return None
        x, y, w, h = faces[0]
        final_image = bgrImg[y:y+h, x:x+w]
        scaled_image = cv2.resize(final_image, (dim, dim))
        #cv2.imshow('img', scaled_image)
        #cv2.waitKey(0)
        #cv2.destroyAllWindows()

        print("Time to align: " + str(time.time()-start))
        ans = [scaled_image]
        return ans

# whiten an image
def whiten_image(source_image):
    source_mean = numpy.mean(source_image)
    source_standard_deviation = numpy.std(source_image)
    std_adjusted = numpy.maximum(source_standard_deviation, 1.0 / numpy.sqrt(source_image.size))
    whitened_image = numpy.multiply(numpy.subtract(source_image, source_mean), 1 / std_adjusted)
    return whitened_image

# create a preprocessed image from the source image that matches the
# network expectations and return it
def preprocess_image(src):
    # scale the image
    NETWORK_WIDTH = 160
    NETWORK_HEIGHT = 160
    preprocessed_image = cv2.resize(src, (NETWORK_WIDTH, NETWORK_HEIGHT))

    #convert to RGB
    preprocessed_image = cv2.cvtColor(preprocessed_image, cv2.COLOR_BGR2RGB)

    #whiten
    preprocessed_image = whiten_image(preprocessed_image)

    # return the preprocessed image
    return preprocessed_image

def run_inference(image_to_classify, facenet_graph):

    scaled_image = cv2.resize(image_to_classify, (640, 480))
    face_image = detect_face(scaled_image)
    if numpy.any(face_image) == None or len(face_image) == 0:
        return None
    
    #resized_image = preprocess_image(face_image)
    #global idx
    #cv2.imwrite(str(idx)+".jpg", face_image)
    #idx += 1

    resized_image = preprocess_image(face_image)
    # ***************************************************************
    # Send the image to the NCS
    # ***************************************************************
    facenet_graph.LoadTensor(resized_image.astype(numpy.float16), None)

    # ***************************************************************
    # Get the result from the NCS
    # ***************************************************************
    output, userobj = facenet_graph.GetResult()
    return output

def face_diff(face1_output, face2_output):
    # print (len(face1_output))
    if (len(face1_output) != len(face2_output)):
        print('length mismatch in face_match')
        return False
    total_diff = 0
    for output_index in range(0, len(face1_output)):
        this_diff = numpy.square(face1_output[output_index] - face2_output[output_index])
        total_diff += this_diff
    
    return total_diff
    
def run_image(inference_output, test_output):
    ranking = []
    for directory in inference_output:
        for valid_image in inference_output[directory]:
            diff = face_diff(valid_image, test_output)
            if diff >= FACE_MATCH_THRESHOLD:
                ranking.append([diff, "None"])
            else:
                ranking.append([diff, directory])

    ranking.sort()
    result = {}
    for idx in range(5):
        if ranking[idx][1] in result:
            result[ranking[idx][1]] += 1
        else:
            result[ranking[idx][1]] = 1
    
    ans = "None"
    count = 0
    for d in result:
        if result[d] > count:
            count = result[d]
            ans = d

    count = 0
    if ans == "None" and len(result) != 1:
        for d in result:
            if d != "None":
                if result[d] > count:
                    count = result[d]
                    ans = d

    return ans

def train():

    devices = mvnc.EnumerateDevices()
    if len(devices) == 0:
        print('No NCS devices found')
        quit()

    # Pick the first stick to run the network
    device = mvnc.Device(devices[0])

    # Open the NCS
    device.OpenDevice()

    # The graph file that was created with the ncsdk compiler
    graph_file_name = GRAPH_FILENAME

    # read in the graph file to memory buffer
    with open(graph_file_name, mode='rb') as f:
        graph_in_memory = f.read()

    # create the NCAPI graph instance from the memory buffer containing the graph file.
    graph = device.AllocateGraph(graph_in_memory)

    valid_data_directory_list = os.listdir(VALID_DATA_DIR)
    size = len(valid_data_directory_list)
    inference_output = {}
    for d in valid_data_directory_list:
        dir_name = VALID_DATA_DIR + d

        valid_image_filename_list = [
            dir_name + "/" + i for i in os.listdir(dir_name) if i.endswith(".jpg")]

        for valid_image_filename in valid_image_filename_list:
            validated_image = cv2.imread(valid_image_filename)
            valid_output = run_inference(validated_image, graph)
            if numpy.any(valid_output) == None:
                print("No face detected in " + valid_image_filename + " in dir: " + dir_name)
                continue
            if d in inference_output:
                inference_output[d].append(valid_output)
            else:
                inference_output[d] = [valid_output]
    
    with open('model.pkl', 'wb') as mod:
        pickle.dump(inference_output, mod)

    # Clean up the graph and the device
    graph.DeallocateGraph()
    device.CloseDevice()

# main entry point for program. we'll call main() to do what needs to be done.
if __name__ == "__main__":
    sys.exit(train())

