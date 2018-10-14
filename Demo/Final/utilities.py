#! /usr/bin/env python2

from mvnc import mvncapi as mvnc
import numpy
import cv2
import sys
import os
import cPickle as pickle
import openface
import time

IMAGES_DIR = './'

VALID_DATA_DIR = IMAGES_DIR + 'valid_data/'

GRAPH_FILENAME = "facenet_celeb_ncs.graph"

# the same face will return 0.0
# different faces return higher numbers
# this is NOT between 0.0 and 1.0
FACE_MATCH_THRESHOLD = 1.2

DLIB_FACE_PREDICTOR = "shape_predictor_68_face_landmarks.dat"

def detect_face(bgrImg, align, imgDim = 160, multipleFaces = False):
    if bgrImg is None:
        raise Exception("Unable to load image/frame")

    rgbImg = cv2.cvtColor(bgrImg, cv2.COLOR_BGR2RGB)
    
    if (multipleFaces):
        # Get all bounding boxes
        t = time.time()
        bb = align.getAllFaceBoundingBoxes(rgbImg)
        if bb is None:
            # raise Exception("Unable to find a face: {}".format(imgPath))
            return None

        alignedFaces = []
        for box in bb:
            alignedFaces.append(
                align.align(
                    imgDim,
                    rgbImg,
                    box,
                    landmarkIndices=openface.AlignDlib.OUTER_EYES_AND_NOSE))
        print(time.time() - t)
        if alignedFaces is None:
            raise Exception("Unable to align the frame")

        # if len(alignedFaces) == 0:
            #cv2.imshow('img', bgrImg)
            #cv2.waitKey(0)
            #cv2.destroyAllWindows()
        #    return None

        #for i in range(len(alignedFaces)):
        #    alignedFaces[i] = cv2.cvtColor(alignedFaces[i], cv2.COLOR_RGB2BGR)

        return alignedFaces

    else:
        # Get the largest face bounding box
        t = time.time()
        bb = align.getLargestFaceBoundingBox(rgbImg) #Bounding box
        alignedFace = align.align(
                                imgDim,
                                rgbImg,
                                bb,
                                landmarkIndices=openface.AlignDlib.OUTER_EYES_AND_NOSE)
        # alignedFace = cv2.cvtColor(alignedFace, cv2.COLOR_RGB2BGR)
        print("Time to align: " + str(time.time()-t))
        return alignedFace


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

    #whiten
    preprocessed_image = whiten_image(preprocessed_image)

    # return the preprocessed image
    return preprocessed_image

# Run an inference on the passed image
# image_to_classify is the image on which an inference will be performed
#    upon successful return this image will be overlayed with boxes
#    and labels identifying the found objects within the image.
# ssd_mobilenet_graph is the Graph object from the NCAPI which will
#    be used to peform the inference.
def run_inference(image_to_classify, facenet_graph, dlib_model):

    # get a resized version of the image that is the dimensions
    # SSD Mobile net expects
    scaled_image = cv2.resize(image_to_classify, (640, 480))
    # scaled_image = image_to_classify[0:480, 104:744, :]
    # cv2.imshow('img', scaled_image)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()
    face_image = detect_face(scaled_image, dlib_model)
    if numpy.any(face_image) == None:
        return None
    resized_image = preprocess_image(face_image)
    # ***************************************************************
    # Send the image to the NCS
    # ***************************************************************
    facenet_graph.LoadTensor(resized_image.astype(numpy.float16), None)

    # ***************************************************************
    # Get the result from the NCS
    # ***************************************************************
    output, userobj = facenet_graph.GetResult()

    #print("Total results: " + str(len(output)))
    #print(output)

    return output

# determine if two images are of matching faces based on the
# the network output for both images.
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
    
# Test all files in a list for a match against a valided face and display each one.
# valid_output is inference result for the valid image
# validated image filename is the name of the valid image file
# graph is the ncsdk Graph object initialized with the facenet graph file
#   which we will run the inference on.
# input_image_filename_list is a list of image files to compare against the
#   valid face output.
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

# This function is called from the entry point to do
# all the work of the program
def train():

    # Get a list of ALL the sticks that are plugged in
    # we need at least one
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

    # Dlib model
    align = openface.AlignDlib(DLIB_FACE_PREDICTOR)

    valid_data_directory_list = os.listdir(VALID_DATA_DIR)
    size = len(valid_data_directory_list)
    inference_output = {}
    for d in valid_data_directory_list:
        dir_name = VALID_DATA_DIR + d

        valid_image_filename_list = [
            dir_name + "/" + i for i in os.listdir(dir_name) if i.endswith(".jpg")]

        for valid_image_filename in valid_image_filename_list:
            validated_image = cv2.imread(valid_image_filename)
            valid_output = run_inference(validated_image, graph, align)
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

