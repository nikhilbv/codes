def isjson(src):
  file = src.split('/')[-1].split('.')[-1]
  if file == 'json':
    return file

src = '/aimldl-dat/data-gaze/AIML_Database/lnd-211019_120637/images-p1-230919_AT1_via205_081019_tuSimple-211019_120637.json'

if isjson(src):
  print("true") 