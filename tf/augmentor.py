import Augmentor

p = Augmentor.Pipeline("/home/nikhil/Documents/exp")
# p = Augmentor.Pipeline("/aimldl-dat/data-gaze/AIML_Aids/lnd-251119_114944/training")
# Point to a directory containing ground truth data.
# Images with the same file names will be added as ground truth data
# and augmented in parallel to the original data.
p.ground_truth("/home/nikhil/Documents/exp")
# Add operations to the pipeline as normal:
# p.rotate(probability=1, max_left_rotation=5, max_right_rotation=5)
p.rotate_without_crop(1, max_left_rotation=25, max_right_rotation=25, expand=False)
# p.flip_left_right(probability=0.5)
# p.zoom_random(probability=0.5, percentage_area=0.8)
# p.flip_top_bottom(probability=1)
# p.sample(2592)
p.process()
