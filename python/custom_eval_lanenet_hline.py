import numpy as np
from sklearn.linear_model import LinearRegression
import ujson as json
import csv
import os
import cv2

class LaneEval(object):
  lr = LinearRegression()
  pixel_thresh = 20
  pt_thresh = 0.85

  @staticmethod
  def get_angle(xs, y_samples):
    xs, ys = xs[xs >= 0], y_samples[xs >= 0]
    if len(xs) > 1:
      LaneEval.lr.fit(ys[:, None], xs)
      k = LaneEval.lr.coef_[0]
      theta = np.arctan(k)
    else:
      theta = 0
    return theta

  @staticmethod
  def line_accuracy(pred, gt, thresh):
    pred = np.array([p if p >= 0 else -100 for p in pred])
    # print("pred inside line_accuracy : {}".format(pred))
    gt = np.array([g if g >= 0 else -100 for g in gt])  
    # print("gt inside line_accuracy : {}".format(gt))
    # print(np.sum(np.where(np.abs(pred - gt) < thresh, 1., 0.)) / len(gt))
    return np.sum(np.where(np.abs(pred - gt) < thresh, 1., 0.)) / len(gt)

  @staticmethod
  # def bench(pred, gt, y_samples, running_time):
  def bench(pred, gt, y_samples):
    # print("pred inside bench : {}".format(pred))
    # print("gt inside bench : {}".format(gt))

    gtt = []
    pdd = []
    line_accs = []
    fp, fn = 0., 0.
    matched = 0.
    ann_in_pred = 0
    ann_in_gt = 0

    if any(len(p) != len(y_samples) for p in pred):
      raise Exception('Format of lanes error.')
    # if running_time > 200 or len(gt) + 2 < len(pred):
    # if len(gt) + 2 < len(pred):
    #   print("gt : {}".format(gt))
    #   sys.exit()
    #   return 0. , 0., 0., 0., 0., 0., 0., 0., 0., 1.

    for lane_g in gt:
      lane_g_id_found=False
      for lane_g_id in lane_g:
        if lane_g_id == -2:
          continue
        else:
          gtt.append(lane_g)
          lane_g_id_found=True
          break
      if lane_g_id_found:
        ann_in_gt += 1      
    
    # print("gtt inside bench : {}".format(gtt))

    angles = [LaneEval.get_angle(np.array(x_gts), np.array(y_samples)) for x_gts in gt]
    # print("angles : {}".format(angles))
    threshs = [LaneEval.pixel_thresh / np.cos(angle) for angle in angles]
    # print("threshs : {}".format(threshs))
    

    for lane_p in pred:
      lane_p_id_found=False
      for lane_p_id in lane_p:
        if lane_p_id == -2:
          continue
        else:
          pdd.append(lane_p)
          lane_p_id_found=True
          break
      if lane_p_id_found:
        ann_in_pred += 1
    
    for x_gts, thresh in zip(gtt, threshs):
      # print("x_gts : {}".format(x_gts))
      # print("thresh : {}".format(thresh))
      accs = [LaneEval.line_accuracy(np.array(x_preds), np.array(x_gts), thresh) for x_preds in pred]
      # print('accs : {}'.format(accs))
      # print('accs : {}'.format(type(accs)))
      max_acc = np.max(accs) if len(accs) > 0 else 0.
      if max_acc < LaneEval.pt_thresh:
        fn += 1
      else:
        matched += 1
      line_accs.append(max_acc)
    # if ann_in_pred > ann_in_gt:
    #   # fp = ann_in_gt - matched
    #   fp = ann_in_pred - matched
    
    # if len(gtt) > 4 and fn > 0:
    #   fn -= 1
    s = sum(line_accs)
    
    # if ann_in_gt > 4:
    #   s -= min(line_accs)
    
    if len(gtt) > 4 and fn > 0:
      fn -= 1
    s = sum(line_accs)
    
    if ann_in_gt > 4:
      s -= min(line_accs)
    
    # fields = ['image_name', 'len_of_gt', 'len_of_pred','matched','fp','fn']
    # row = [raw_file,len(gt),len(pred),matched,fp,fn]
    # rows.append(row)
    # with open('perimage.csv','a') as csvfile:
    #   writer = csv.writer(csvfile)
    #   writer.writerow(fields)
    #   writer.writerows(rows)
    # return ann_in_gt,ann_in_pred,len(gt),len(pred),matched,fp,fn,s / max(min(4.0, len(gt)), 1.), fp / len(pred) if len(pred) > 0 else 0., fn / max(min(len(gt), 4.) , 1.)
    # return ann_in_gt,ann_in_pred,len(gt),len(pred),matched,fp,fn, s / ann_in_gt, fp / ann_in_pred if ann_in_pred > 0 else 0., fn / max(min(ann_in_gt, 4.) , 1.)
    return ann_in_gt,ann_in_pred,len(gt),len(pred),matched,fp,fn, s / max(min(4.0, ann_in_gt), 1.), fp / ann_in_pred if ann_in_pred > 0 else 0., fn / max(min(ann_in_gt, 4.) , 1.)

  @staticmethod
  def bench_one_submit(pred_file, gt_file):
    # print("gt_file : {}".format(gt_file))
    # print("pred_file : {}".format(pred_file))
    path = '/aimldl-dat/logs/testing/evaluate'
    try:
      json_pred = [json.loads(line) for line in open(pred_file).readlines()]
    except BaseException as e:
      raise Exception('Fail to load json file of the prediction.')
    json_gt = [json.loads(line) for line in open(gt_file).readlines()]
    if len(json_gt) != len(json_pred):
      raise Exception('We do not get the predictions of all the test tasks')
    gts = {l['raw_file']: l for l in json_gt}
    accuracy, fp, fn = 0., 0., 0.
    no_of_ann_in_pred = 0
    no_of_ann_in_gt = 0
    rows = []
    for pred in json_pred:
      # if 'raw_file' not in pred or 'lanes' not in pred or 'run_time' not in pred:
      if 'raw_file' not in pred or 'lanes' not in pred:
        # raise Exception('raw_file or lanes or run_time not in some predictions.')
        raise Exception('raw_file or lanes not in some predictions.')
      raw_file = pred['raw_file']
      # print("raw_file : {}".format(raw_file))
      image_name = raw_file.split('/')[-1]
      img_path = os.path.join(path, image_name)
      # print("images are saved in : {}".format(img_path))
      img = cv2.imread(raw_file)
      if img is None:
        continue

      pred_lanes = pred['lanes']
      for lane_p in pred_lanes:
        lane_p_id_found=False
        for lane_p_id in lane_p:
          if lane_p_id == -2:
            continue
          else:
            lane_p_id_found=True
            break
        if lane_p_id_found:
          no_of_ann_in_pred += 1
      y_samples = pred['v_samples']

      pred_lanes_vis = [[(y, x) for (x, y) in zip(lane, y_samples) if x >= 0] for lane in pred_lanes]
      img_vis = img.copy()

      for lane in pred_lanes_vis:
        cv2.polylines(img_vis, np.int32([lane]), isClosed=False, color=(0,0,255), thickness=3)
        # cv2.imwrite(img_path, img_vis)

      # run_time = pred['run_time']
      if raw_file not in gts:
        raise Exception('Some raw_file from your predictions do not exist in the test tasks.')
      gt = gts[raw_file]
      gt_lanes = gt['lanes']
      for lane_g in gt_lanes:
        lane_g_id_found=False
        for lane_g_id in lane_g:
          if lane_g_id == -2:
            continue
          else:
            lane_g_id_found=True
            break 
        if lane_g_id_found:
          no_of_ann_in_gt += 1
      y_samples = gt['h_samples']

      gt_lanes_vis = [[(y, x) for (x, y) in zip(lane, y_samples) if x >= 0] for lane in gt_lanes]
      # img_vis = img.copy()

      for lane in gt_lanes_vis:
        cv2.polylines(img_vis, np.int32([lane]), isClosed=False, color=(0,255,0), thickness=3)
        cv2.imwrite(img_path, img_vis)
      # print(LaneEval.bench(pred_lanes, gt_lanes, y_samples))
      try:
          # a, p, n = LaneEval.bench(pred_lanes, gt_lanes, y_samples, run_time)
        ann_in_gt,ann_in_pred,len_of_gt,len_of_pred,matched,fpi,fni,a, p, n = LaneEval.bench(pred_lanes, gt_lanes, y_samples)
      except BaseException as e:
        raise Exception('Format of lanes error.')

      image = raw_file.split('/')[-1]
      # print("image : {}".format(image))

      fields = ['image_name', 'len_of_gt','no_of_ann_in_gt','len_of_pred','no_of_ann_in_pred','matched','fp','fn']
      row = [image,len_of_gt,ann_in_gt,len_of_pred,ann_in_pred,matched,fpi,fni]
      rows.append(row)
      accuracy += a
      fp += p
      fn += n

    num = len(gts)
    with open('perimage.csv','a') as csvfile:
      writer = csv.writer(csvfile)
      writer.writerow(fields)
      writer.writerows(rows)

    val = {
        'Accuracy' : round(accuracy/num,4),
        'FP' : round(fp/num,4),
        'FN' : round(fn/num,4),
        'No_of_ann_in_gt' : no_of_ann_in_gt,
        'No_of_ann_in_pred' : no_of_ann_in_pred
    }

    return val

if __name__ == '__main__':
  import sys
  # try:
  #   if len(sys.argv) != 3:
  #     raise Exception('Invalid input arguments')
  #   print(LaneEval.bench_one_submit(sys.argv[1], sys.argv[2]))
  # except Exception as e:
  #   print(e.message)
  #   sys.exit(e.message)
  print(LaneEval.bench_one_submit(sys.argv[1], sys.argv[2]))

