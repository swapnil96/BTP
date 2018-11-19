import faceRecognition as fr
import os, cv2, time

curr = os.getcwd()

DATA_DIR = "../raw/"


def main():
    start1 = time.time()
    fr.setup()
    print ("Setup time:", time.time() - start1)
    data_directory_list = os.listdir(DATA_DIR)
    size = len(data_directory_list)
    matrix = {}
    idx = 0
    not_detect = 0
    for d in data_directory_list:
        dir_name = DATA_DIR + d

        image_filename_list = [
            dir_name + "/" + i for i in os.listdir(dir_name) if i.endswith(".jpg")]

        for image_filename in image_filename_list:
            if idx < 5:
                start2 = time.time()
            
            output = fr.run(image_filename)
            
            if idx < 2:
                print ("Total prediction time:", time.time() - start2)
            
            if len(output) == 0:
                not_detect += 1
                print ("No result")
                continue

            if d in matrix:
                if output in matrix[d]:
                    matrix[d][output[0]] += 1
                else:
                    matrix[d] = {output[0]: 1}
            else:
                matrix[d] = {output[0]: 1}
            idx += 1
    
    print matrix

if __name__ == "__main__":
    main()
