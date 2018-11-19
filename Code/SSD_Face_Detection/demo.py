import ssd_face_detection as fd
from matplotlib import pyplot as plt
import os, cv2, time

curr = os.getcwd()

DATA_DIR = "../raw/"

def main():
	start1 = time.time()
	fd.setup()
	print ("Setup time:", time.time() - start1)
	data_directory_list = os.listdir(DATA_DIR)
	size = len(data_directory_list)
	not_detect = 0
	detect = 0
	for d in data_directory_list:
		dir_name = DATA_DIR + d
		image_filename_list = [dir_name + "/" + i for i in os.listdir(dir_name) if i.endswith(".jpg")]
		t = time.time()
		for image_filename in image_filename_list:
			output = fd.run(image_filename)
			if len(output) == 0:
				not_detect += 1
				print ("No result")
				# img = cv2.imread(image_filename)
				# plt.imshow(img)
				# plt.show()
				continue
			detect += 1
			print (time.time()-t)
			t = time.time()
			
	print not_detect, detect

if __name__ == "__main__":
    main()
