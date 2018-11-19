import faceRecognition as fr
from matplotlib import pyplot as plt
import os, cv2, time

DATA_DIR = "../raw/"

def main():
    start1 = time.time()
    fr.setup()
    print ("Setup time:", time.time() - start1)
    data_directory_list = os.listdir(DATA_DIR)
    matrix = {}
    idx = 0
    not_detect = 0
    detect = 0
    for d in data_directory_list:
        dir_name = DATA_DIR + d

        image_filename_list = [dir_name + "/" + i for i in os.listdir(dir_name) if i.endswith(".jpg")]
	t = time.time()
        for image_filename in image_filename_list:
            if idx < 5:
                start2 = time.time()
            
            output = fr.run(image_filename)
            
            if idx < 2:
                print ("Total prediction time:", time.time() - start2)
            
            if len(output) == 0:
                not_detect += 1
                print ("No result", image_filename)
                continue

            if d in matrix:
                if output[0] in matrix[d]:
                    matrix[d][output[0]] += 1
                else:
                    matrix[d][output[0]] = 1
            else:
                #print (d, dir_name, output)
                matrix[d] = {output[0]: 1}
            idx += 1
            detect += 1
	    print (time.time()-t)
  	    t = time.time()

    print not_detect, detect
    print matrix

if __name__ == "__main__":
    main()