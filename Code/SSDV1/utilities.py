#! /usr/bin/env python2

from mvnc import mvncapi as mvnc
from matplotlib import pyplot as plt
import numpy, cv2
import sys, os
import cPickle as pickle, time

IMAGES_DIR = "../"
FACE_MATCH_THRESHOLD = 1.2
VALID_DATA_DIR = IMAGES_DIR + 'valid_data/'

GRAPH_FILENAME = "../facenet_celeb_ncs.graph"
FACE_D_GRAPH_FILENAME = "ssd-face.graph"

devices = mvnc.EnumerateDevices()

def whiten_image(source_image):
    source_mean = numpy.mean(source_image)
    source_standard_deviation = numpy.std(source_image)
    std_adjusted = numpy.maximum(source_standard_deviation, 1.0 / numpy.sqrt(source_image.size))
    whitened_image = numpy.multiply(numpy.subtract(source_image, source_mean), 1 / std_adjusted)
    return whitened_image

def preprocess_image(src, W=160, H=160):
    # scale the image
    NETWORK_WIDTH = W
    NETWORK_HEIGHT = H
    preprocessed_image = cv2.resize(src, (NETWORK_WIDTH, NETWORK_HEIGHT))

    #convert to RGB
    preprocessed_image = cv2.cvtColor(preprocessed_image, cv2.COLOR_BGR2RGB)

    #whiten
    preprocessed_image = whiten_image(preprocessed_image)

    # return the preprocessed image
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

def run_inference(image_to_classify):

    scaled_image = preprocess_image(image_to_classify, 300, 300)
    
    device = mvnc.Device(devices[0])
    device.OpenDevice()
    graph_file_name = FACE_D_GRAPH_FILENAME
    with open(graph_file_name, mode='rb') as f:
        graph_in_memory = f.read()

    facenet_graph = device.AllocateGraph(graph_in_memory)
    facenet_graph.LoadTensor(scaled_image.astype(numpy.float16), None)
    output, userobj = facenet_graph.GetResult()
    
    face_image = detect_face(output, image_to_classify)
    
    facenet_graph.DeallocateGraph()
    device.CloseDevice()

    if len(face_image) == 0 or numpy.any(face_image[0]) == None:
        return None

    resized_image = preprocess_image(face_image[0], 160, 160)

    # ***************************************************************
    # Send the image to the NCS
    # ***************************************************************
    device = mvnc.Device(devices[0])
    device.OpenDevice()
    graph_file_name = GRAPH_FILENAME
    with open(graph_file_name, mode='rb') as f:
        graph_in_memory = f.read()

    facenet_graph = device.AllocateGraph(graph_in_memory)

    facenet_graph.LoadTensor(resized_image.astype(numpy.float16), None)

    # ***************************************************************
    # Get the result from the NCS
    # ***************************************************************
    output, userobj = facenet_graph.GetResult()
    
    facenet_graph.DeallocateGraph()
    device.CloseDevice()

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

    valid_data_directory_list = os.listdir(VALID_DATA_DIR)
    inference_output = {}
    for d in valid_data_directory_list:
        dir_name = VALID_DATA_DIR + d

        valid_image_filename_list = [
            dir_name + "/" + i for i in os.listdir(dir_name) if i.endswith(".jpg")]

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
    
    with open('model.pkl', 'wb') as mod:
        pickle.dump(inference_output, mod)

# main entry point for program. we'll call main() to do what needs to be done.
if __name__ == "__main__":
    sys.exit(train())

