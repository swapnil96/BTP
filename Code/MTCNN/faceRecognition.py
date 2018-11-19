#! /usr/bin/env python

from mvnc import mvncapi as mvnc
import numpy, cv2
import sys, os
import cPickle as pickle 
import utilities
import tensorflow as tf

model = None

def setup():
    
    global model
    with open('model.pkl', 'rb') as mod:
        model = pickle.load(mod)

def run(input_image):

    if (model == None):
        print("Run setup function once first")
        return []

    infer_image = cv2.imread(input_image)
    input_vector = utilities.run_inference(infer_image)
    if numpy.any(input_vector) == None:
        return []

    match = utilities.run_image(model, input_vector)
    return [match]

if __name__ == "__main__":
    #print(run(sys.argv[1]))
    setup()
