#! /usr/bin/env python3

from mvnc import mvncapi as mvnc
import numpy, cv2
import sys, os
import cPickle as pickle
import fd

def getPaddingSize(img):
    h, w, _ = img.shape
    top, bottom, left, right = (0,0,0,0)

    if w < int(h / 3 * 4):
        tmp = int(h / 3 * 4) - w
        left = tmp // 2
        right = tmp - left
    elif h < int(w / 4 * 3):
        tmp = int(w / 4 * 3) - h
        top = tmp // 2
        bottom = tmp - top
    else:
        pass
    return top, bottom, left, right

def run_inference(image_to_classify, args):

    scaled_image = cv2.resize(image_to_classify, (640, 480))
    
    image = scaled_image
    top,bottom,left,right = getPaddingSize(image)
    image = cv2.copyMakeBorder(image, top, bottom, left, right, cv2.BORDER_CONSTANT, value=[0,0,0])
    frame = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    face_image = fd.detect_face(frame)
    if len(face_image) == 0 or numpy.any(face_image[0]) == None:
        return None
        
    resized_image = preprocess_image(face_image[0])
    # ***************************************************************
    # Send the image to the NCS
    # ***************************************************************
    devices = mvnc.EnumerateDevices()
    device = mvnc.Device(devices[0])
    device.OpenDevice()
    graph_file_name = args.facenetGraph
    with open(graph_file_name, mode='rb') as f:
        graph_in_memory = f.read()

    facenet_graph = device.AllocateGraph(graph_in_memory)

    facenet_graph.LoadTensor(resized_image.astype(numpy.float16), None)
    # ***************************************************************
    # Get the result from the NCS
    # ***************************************************************
    output, userobj = facenet_graph.GetResult()
    device.CloseDevice()

    return output

# whiten an image
def whiten_image(source_image):
    source_mean = numpy.mean(source_image)
    source_standard_deviation = numpy.std(source_image)
    std_adjusted = numpy.maximum(source_standard_deviation, 1.0 / numpy.sqrt(source_image.size))
    whitened_image = numpy.multiply(numpy.subtract(source_image, source_mean), 1 / std_adjusted)
    return whitened_image

def preprocess_image(src):
    NETWORK_WIDTH = 160
    NETWORK_HEIGHT = 160
    preprocessed_image = cv2.resize(src, (NETWORK_WIDTH, NETWORK_HEIGHT))

    #convert to RGB
    preprocessed_image = cv2.cvtColor(preprocessed_image, cv2.COLOR_BGR2RGB)
    #whiten
    preprocessed_image = whiten_image(preprocessed_image)

    # return the preprocessed image
    return preprocessed_image

def face_diff(face1_output, face2_output):
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
            valid_output = run_inference(validated_image, args)
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

