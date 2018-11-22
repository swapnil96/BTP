#! /usr/bin/env python

from mvnc import mvncapi as mvnc
from matplotlib import pyplot as plt
import numpy, cv2
import sys, os
import cPickle as pickle 
import utilities

GRAPH_FILENAME = "ssd-face.graph"
graph = None

def setup():
    
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

def run(input_image):

    if (graph == None):
        print("Run setup function once first")
        return []

    infer_image = cv2.imread(input_image)
    input_vector = utilities.run_inference(infer_image, graph)
    if numpy.any(input_vector) == None:
        return []

    match = utilities.run_image(input_vector, infer_image)
    '''for img in match:
		plt.imshow(img)
		plt.show()
	'''
    return match

if __name__ == "__main__":
    #print(run(sys.argv[1]))
    setup()
