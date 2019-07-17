str = "freshersworld is hiring nodeServer"
word = str.split()
for i in range(len(word)):
	if(len(word[i]) > 6):
		count=count+1
print(count)