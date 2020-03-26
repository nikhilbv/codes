import os
import cv2
import datetime

def load_remap_matrix():

  fs = cv2.FileStorage(ipm_remap_file_path, cv2.FILE_STORAGE_READ)

  remap_to_ipm_x = fs.getNode('remap_ipm_x').mat()
  remap_to_ipm_y = fs.getNode('remap_ipm_y').mat()

  ret = {
      'remap_to_ipm_x': remap_to_ipm_x,
      'remap_to_ipm_y': remap_to_ipm_y,
  }

  fs.release()

  return ret

ipm_remap_file_path = '/codehub/external/lanenet-lane-detection/data/tusimple_ipm_remap.yml'
remap_file_load_ret = load_remap_matrix()
remap_to_ipm_x = remap_file_load_ret['remap_to_ipm_x']
remap_to_ipm_y = remap_file_load_ret['remap_to_ipm_y']

timestamp = ("{:%d%m%y_%H%M%S}").format(datetime.datetime.now())
debug_image_dir = '/aimldl-dat/logs/lanenet/debug'
debug_image_path = os.path.join(debug_image_dir,timestamp)
os.makedirs(debug_image_path)


source_image = "/aimldl-dat/samples/lanenet/7.jpg"
image = cv2.imread(source_image, cv2.IMREAD_COLOR)

tmp_ipm_image = cv2.remap(image, remap_to_ipm_x, remap_to_ipm_y, interpolation=cv2.INTER_NEAREST)
tmp_ipm_image_path = os.path.join(debug_image_path, "tmp_ipm_image.png")
cv2.imwrite(tmp_ipm_image_path, tmp_ipm_image)