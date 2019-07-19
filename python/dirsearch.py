import glob
import os

# search_dir = "/home/nikhil/Documents/ai-ml-dl-data/logs/mask_rcnn"
# # remove anything from the list that is not a file (directories, symlinks)
# # thanks to J.F. Sebastion for pointing out that the requirement was a list 
# # of files (presumably not including directories)  
# files = filter(os.path.isdir, glob.glob(search_dir + "*"))
# files.sort(key=lambda x: os.path.getmtime(x))

# files.sort(key=lambda fn: os.path.getmtime(os.path.join(search_dir, fn)))



# import os
# import time
# from pprint import pprint

# pprint([(x[0], time.ctime(x[1].st_ctime)) for x in sorted([(fn, os.stat(fn)) for fn in os.listdir(".")], key = lambda x: x[1].st_ctime)])



# search_dir = "/home/nikhil/Documents/ai-ml-dl-data/logs/mask_rcnn"
# os.chdir(search_dir)
# files = filter(os.path.isfile, os.listdir(search_dir))
# files = [os.path.join(search_dir, f) for f in files] # add path to each file
# files.sort(key=lambda x: os.path.getmtime(x))


# search_dir = "/home/nikhil/Documents/ai-ml-dl-data/logs/mask_rcnn"
# # files = os.listdir(search_dir)
# files_fltr = filter(os.path.isdir, os.listdir(search_dir))
# print(files_fltr)
# for name in files:
    # print(name)


print(filter(lambda x: os.path.isdir(x), os.listdir('.')))