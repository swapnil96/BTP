#! /usr/bin/env python

from mvnc import mvncapi as mvnc
import numpy, cv2
import sys, os
import cPickle as pickle 
import utilities, argparse

def setup(args):    
    model = None
    with open(args.testModel, 'rb') as mod:
        model = pickle.load(mod)

    return model

def run(model, args):
    if (model == None):
        print("No model found")
        return []

    infer_image = cv2.imread(args.testData)
    input_vector = utilities.run_inference(infer_image, args)
    if numpy.any(input_vector) == None:
        return []

    match = utilities.run_image(model, input_vector, args.threshold)
    return [match]

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '-fG',
		'--facenetGraph',
		type=str,
		help="graph file for facenet",
		default="../facenet_celeb_ncs.graph")
    parser.add_argument(
		'type',
		type=str,
		help="train for training, test for testing",
		default="train")
    parser.add_argument(
		'--trainData',
		type=str,
		help="Path to train data directory for training",
		default="../train_data/")
    parser.add_argument(
		'--trainModel',
		type=str,
		help="Name of model which training will produce",
		default='model.pkl')    
    parser.add_argument(
        '-tD',
		'--testData',
		type=str,
		help="Path to test image for testing",
		default='../raw/')
    parser.add_argument(
		'-tM',
        '--testModel',
		type=str,
		help="Path to pickle model for testing",
		default='model.pkl')
    parser.add_argument(
		'-t',
        '--threshold',
		type=float,
		default=1.2,
		help='Face recognition threshold for facenet')
    parser.add_argument('-v', '--verbose', action='store_true')
    args = parser.parse_args()
    if (args.type == "train"):
        utilities.train(args)
    else:
        model = setup(args)
        print(run(model, args))
