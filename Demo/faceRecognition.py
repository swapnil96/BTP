#! /usr/bin/env python

from mvnc import mvncapi as mvnc
import numpy, cv2
import sys, os
import cPickle as pickle 
import utilities
import tensorflow as tf

curr = os.getcwd()
sys.path.append(curr+"/facenet_align")
sys.path.append(curr+"/facenet_align/align")

import facenet
import align.detect_face
import align.align_dataset_mtcnn

GRAPH_FILENAME = "facenet_celeb_ncs.graph"

FACE_MATCH_THRESHOLD = 1.2

graph = None
model = None
pnet = None
rnet = None
onet = None

def setup(gpu_memory_fraction=0.25):
    
    global graph, model, pnet, rnet, onet

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

    print('Creating networks and loading parameters for face detection')
    with tf.Graph().as_default():
        gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=gpu_memory_fraction)
        sess = tf.Session(config=tf.ConfigProto(gpu_options=gpu_options, log_device_placement=False))
        with sess.as_default():
            pnet, rnet, onet = align.detect_face.create_mtcnn(sess, None)


def detect_face(input_image):

    done = align.align_dataset_mtcnn.main(os.getcwd(), pnet, rnet, onet, input_image)
    return done

def run(input_image):

    if (graph == None or model == None):
        print("Run setup function once first")
        return []

    faces = detect_face(input_image)
    if (faces == 0):
        print("No face detected")
        return []

    face_input_image = faces[0]
    infer_image = cv2.imread(face_input_image)
    input_vector = utilities.run_inference(infer_image, graph)
    match = utilities.run_image(model, input_vector)
    os.remove(face_input_image)
    return [match]

if __name__ == "__main__":
    #print(run(sys.argv[1]))
    setup()
