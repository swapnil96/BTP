import faceRecognition as fr
from matplotlib import pyplot as plt
import os, cv2, time
import argparse
import cPickle as pickle

def main(args):
    model = fr.setup(args)
    DATA_DIR = args.testData
    data_directory_list = os.listdir(DATA_DIR)
    matrix = {}
    not_detect = 0
    detect = 0
    for d in data_directory_list:
        dir_name = DATA_DIR + "/" + d

        image_filename_list = [dir_name + "/" + i for i in os.listdir(dir_name) if i.endswith(".jpg")]
        t = time.time()
        for image_filename in image_filename_list:
            args.testData = image_filename
            output = fr.run(model, args)
            if len(output) == 0:
                not_detect += 1
                if (args.verbose):
                    print ("No result", image_filename)
                continue

            if d in matrix:
                if output[0] in matrix[d]:
                    matrix[d][output[0]] += 1
                else:
                    matrix[d][output[0]] = 1
            else:
                matrix[d] = {output[0]: 1}
            detect += 1
            if (args.verbose):
                print ("Time for processing", time.time()-t)
            t = time.time()

    if (args.verbose):
        print not_detect, detect
        print matrix

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '-fG',
        '--facenetGraph',
        type=str,
        help="graph file for facenet",
        default="../facenet_celeb_ncs.graph")
    parser.add_argument(
        '-sG',
        '--ssdGraph',
        type=str,
        help="graph file for SSD",
        default="ssd_face.graph")
    parser.add_argument(
        '-tD',
        '--testData',
        type=str,
        help="Path to test image for testing",
        default='../raw')
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
    main(args)
