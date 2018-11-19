#! /usr/bin/env python

from mvnc import mvncapi as mvnc
import numpy, cv2
import sys, os
import cPickle as pickle 
import utilities
import tensorflow as tf
import openface

GRAPH_FILENAME = "../facenet_celeb_ncs.graph"
FACE_MATCH_THRESHOLD = 1.2

DLIB_FACE_PREDICTOR = "shape_predictor_68_face_landmarks.dat"

graph = None
model = None
align = None

def setup(gpu_memory_fraction=0.25):
    
    global graph, model, align

    print('Loading NCS graph')

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

    with open('model.pkl', 'rb') as mod:
        model = pickle.load(mod)

    align = openface.AlignDlib(DLIB_FACE_PREDICTOR)

def run(input_image):

    if (graph == None or model == None or align == None):
        print("Run setup function once first")
        return []

    infer_image = cv2.imread(input_image)
    input_vector = utilities.run_inference(infer_image, graph, align)
    if numpy.any(input_vector) == None:
        return []
    match = utilities.run_image(model, input_vector)
    return [match]

if __name__ == "__main__":
    #print(run(sys.argv[1]))
    setup()
