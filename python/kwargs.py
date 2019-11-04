def test_args_kwargs(arg1, arg2, arg3, **dim):
  print "arg1:", arg1
  print "arg2:", arg2
  print "arg3:", arg3
  # print "dim:", dim
  width = 0
  height = 0
  for k,v in enumerate(dim()):
    print k,v
    if k=='width':
      width=dim[k]
    if k=='height':
      height=dim[k]
  print "width:", width
  print "height:", height

# # first with *args
# args = ("two", 3,5)
# test_args_kwargs(*args)

# now with **kwargs:
dim = {"width": 1280, "height": 720 }
test_args_kwargs(3,"two",5,**dim)
