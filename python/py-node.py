from Naked.toolshed.shell import execute_js
# prog = '/aimldl-cod/apps/annon/lanenet_convertviatotusimple.js --pred /aimldl-dat/logs/lanenet/predict/051119_152152/pred_json/pred-051119_152152.json'
# success = execute_js('lanenet_convertviatotusimple.js --pred /aimldl-dat/logs/lanenet/predict/051119_152152/pred_json/pred-051119_152152.json')
# success = execute_js(prog)

# prog = '/aimldl-cod/apps/annon/lanenet_convertviatotusimple.js --pred /aimldl-dat/logs/lanenet/predict/051119_152152/pred_json/pred-051119_152152.json'
prog = '/aimldl-cod/apps/annon/lanenet_convertviatotusimple.js'
cmd = '--pred'
# json = '/aimldl-dat/logs/lanenet/predict/051119_152152/pred_json/pred-051119_152152.json'
json = '/aimldl-dat/logs/lanenet/evaluate/pred_json/pred-241019_142720.json'
# print("{} {} {}".format(prog,cmd,json))
success = execute_js("{} {} {}".format(prog,cmd,json))
print('success : {}'.format(success))

out = json.replace('.json','_tuSimple.json')
print('out : {}'.format(out))