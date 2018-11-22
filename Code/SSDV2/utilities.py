#! /usr/bin/env python2

from mvnc import mvncapi
from matplotlib import pyplot as plt
import numpy, cv2
import sys, os
import cPickle as pickle, time
import argparse

def transform_input(img, transpose=True, dtype=numpy.float32, W = 160, H = 160):
    inpt = cv2.resize(img, (W,H))
    inpt = inpt - 127.5
    inpt = inpt / 127.5
    inpt = inpt.astype(dtype)
    if transpose:
        inpt = inpt.transpose((2, 0, 1))
    return inpt

def whiten_image(source_image):
    source_mean = numpy.mean(source_image)
    source_standard_deviation = numpy.std(source_image)
    std_adjusted = numpy.maximum(source_standard_deviation, 1.0 / numpy.sqrt(source_image.size))
    whitened_image = numpy.multiply(numpy.subtract(source_image, source_mean), 1 / std_adjusted)
    return whitened_image

def preprocess_image(src, dtype = numpy.float32, W=160, H=160):
    # scale the image
    NETWORK_WIDTH = W
    NETWORK_HEIGHT = H
    preprocessed_image = cv2.resize(src, (NETWORK_WIDTH, NETWORK_HEIGHT))

    #convert to RGB
    preprocessed_image = cv2.cvtColor(preprocessed_image, cv2.COLOR_BGR2RGB)

    #whiten
    preprocessed_image = whiten_image(preprocessed_image)

    # return the preprocessed image
    preprocessed_image = preprocessed_image.astype(dtype)
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

def detect_face(inference_output, img):
    boxes, classes, confidence = transform_ncs_output(img, inference_output, 0.25)
    ans = []
    for box, cls, conf in zip(boxes, classes, confidence):
        #cv2.rectangle(img, (box[0], box[1]), (box[2], box[3]), (0,255,0), 2)
		temp = img[box[1]:box[3], box[0]:box[2]]
		ans.append(temp)

    #plt.imshow(img)
    #plt.show()
    return ans

def setup(args):
    global facenet_graph, facenet_input_fifo, facenet_output_fifo, ssd_graph, ssd_input_fifo, ssd_output_fifo
    
    devices = mvncapi.enumerate_devices()
    device = mvncapi.Device(devices[0])
    device.open()
    with open(args.facenetGraph, mode="rb") as f:
        facenet_graph_data = f.read()
    facenet_graph = mvncapi.Graph('facenet_graph')
    facenet_input_fifo, facenet_output_fifo = facenet_graph.allocate_with_fifos(device, facenet_graph_data)

    with open(args.ssdGraph, mode="rb") as f:
        ssd_graph_data = f.read()
    ssd_graph = mvncapi.Graph('ssd_graph')
    ssd_input_fifo, ssd_output_fifo = ssd_graph.allocate_with_fifos(device, ssd_graph_data)
    
def run_inference(image_to_classify):
    global facenet_graph, facenet_input_fifo, facenet_output_fifo, ssd_graph, ssd_input_fifo, ssd_output_fifo

    #plt.imshow(image_to_classify)
    #plt.show()
    #scaled_image = transform_input(image_to_classify, False, numpy.float32, 300, 300)
    #t = time.time()    
    scaled_image = preprocess_image(image_to_classify, numpy.float32, 300, 300)

    ssd_graph.queue_inference_with_fifo_elem(ssd_input_fifo, ssd_output_fifo, scaled_image, None)
    output, _ = ssd_output_fifo.read_elem()

    face_image = detect_face(output, image_to_classify)
    #print (time.time() - t, "FD")
    if len(face_image) == 0 or numpy.any(face_image[0]) == None:
	return None
	
    #scaled_image = transform_input(face_image[0], False, numpy.float32, 160, 160)
    scaled_image = preprocess_image(image_to_classify, numpy.float32, 160, 160)    
    # ***************************************************************
    # Send the image to the NCS
    # ***************************************************************
    facenet_graph.queue_inference_with_fifo_elem(facenet_input_fifo, facenet_output_fifo, scaled_image, None)
    output, _ = facenet_output_fifo.read_elem()
    
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

def train(args):

    valid_data_directory_list = os.listdir(args.trainData)
    inference_output = {}
    for d in valid_data_directory_list:
        dir_name = args.trainData + "/" + d

        valid_image_filename_list = [
            dir_name + "/" + i for i in os.listdir(dir_name) if i.endswith(".jpg")]
        
        done = 0
        for valid_image_filename in valid_image_filename_list:
            validated_image = cv2.imread(valid_image_filename)
            valid_output = run_inference(validated_image)
            if numpy.any(valid_output) == None:
                if (args.verbose):
                    print("No face detected in " + valid_image_filename + " in dir: " + dir_name)
                continue
            if d in inference_output:
                inference_output[d].append(valid_output)
            else:
                inference_output[d] = [valid_output]
            done += 1
            if done == 10:
                break

    with open(args.trainModel, 'wb') as mod:
        pickle.dump(inference_output, mod)

# main entry point for program. we'll call main() to do what needs to be done.
# if __name__ == "__main__":
#     sys.exit(train())

