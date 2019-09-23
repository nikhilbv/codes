class inputoutstring(object):
  def __init__(self):
    self.s= ""

  def getstring(self):
    self.s = raw_input()

  def printstring(self):
    print self.s.upper()

stringobject = inputoutstring()
stringobject.getstring()
stringobject.printstring()
