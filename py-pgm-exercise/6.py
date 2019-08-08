s=raw_input()
d={"uppercase":0, "lowercase":0}
for c in s:
  if c.isupper():
    d["uppercase"]+=1
  else: 
    d["lowercase"]+=1
print(d["uppercase"])
# print("Upper case: {}".format(uppercase))
# print("Lower case: {}".format(lowercase))