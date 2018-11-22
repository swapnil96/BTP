#! /usr/bin/env python2

from mvnc import mvncapi as mvnc
import numpy
import cv2
import sys, os
import cPickle as pickle
import openface
import time

def detect_face(bgrImg, align, imgDim = 160, multipleFaces = False):
    if bgrImg is None:
        raise Exception("Unable to load image/frame")

    rgbImg = cv2.cvtColor(bgrImg, cv2.COLOR_BGR2RGB)
    if (multipleFaces):
        # Get all bounding boxes
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
        if alignedFaces is None:
            raise Exception("Unable to align the frame")

        return alignedFaces
    else:
        # Get the largest face bounding box
        bb = align.getLargestFaceBoundingBox(rgbImg) #Bounding box
        alignedFace = align.align(
                                imgDim,
                                rgbImg,
                                bb,
                                landmarkIndices=openface.AlignDlib.OUTER_EYES_AND_NOSE)
        ans = [alignedFace]
        return ans

# whiten an image
def whiten_image(source_image):
    source_mean = numpy.mean(source_image)
    source_standard_deviation = numpy.std(source_image)
    std_adjusted = numpy.maximum(source_standard_deviation, 1.0 / numpy.sqrt(source_image.size))
    whitened_image = numpy.multiply(numpy.subtract(source_image, source_mean), 1 / std_adjusted)
    return whitened_image

def preprocess_image(src):
    # scale the image
    NETWORK_WIDTH = 160
    NETWORK_HEIGHT = 160
    preprocessed_image = cv2.resize(src, (NETWORK_WIDTH, NETWORK_HEIGHT))

    #whiten
    preprocessed_image = whiten_image(preprocessed_image)

    return preprocessed_image

def run_inference(image_to_classify):
    global graph, align

    scaled_image = cv2.resize(image_to_classify, (640, 480))
    face_image = detect_face(scaled_image, align)
    if len(face_image) == 0 or numpy.any(face_image[0]) == None:
        return None
        
    resized_image = preprocess_image(face_image[0])
    # ***************************************************************
    # Send the image to the NCS
    # ***************************************************************
    graph.LoadTensor(resized_image.astype(numpy.float16), None)

    # ***************************************************************
    # Get the result from the NCS
    # ***************************************************************
    output, _ = graph.GetResult()
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
    
def run_image(inference_output, test_output, threshold):
    ranking = []
    for directory in inference_output:
        for valid_image in inference_output[directory]:
            diff = face_diff(valid_image, test_output)
            if diff >= threshold:
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

def setup(args):
    global graph, align
    devices = mvnc.EnumerateDevices()
    if len(devices) == 0:
        print('No NCS devices found')
        quit()

    device = mvnc.Device(devices[0])
    device.OpenDevice()
    graph_file_name = args.facenetGraph
    with open(graph_file_name, mode='rb') as f:
        graph_in_memory = f.read()

    graph = device.AllocateGraph(graph_in_memory)
    align = openface.AlignDlib(args.dlib)

def train(args):
    valid_data_directory_list = os.listdir(args.trainData)
    inference_output = {}
    for d in valid_data_directory_list:
        dir_name = args.trainData + "/" + d

        valid_image_filename_list = [dir_name + "/" + i for i in os.listdir(dir_name) if i.endswith(".jpg")]
        done = 0
        for valid_image_filename in valid_image_filename_list:
            validated_image = cv2.imread(valid_image_filename)

            valid_output = run_inference(validated_image)
            if numpy.any(valid_output) == None:
                print("No face detected in " + valid_image_filename + " in dir: " + dir_name)
                continue
            if d in inference_output:
                inference_output[d].append(valid_output)
            else:
                inference_output[d] = [valid_output]
            done += 1
            if done == 5:
                break
    
    with open(args.trainModel, 'wb') as mod:
        pickle.dump(inference_output, mod)
