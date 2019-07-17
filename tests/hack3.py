Sentence = "GeeksforGeeks is good to learn"
words = Sentence.split(" ") 
newWords = [word[::-1] for word in words] 
newSentence = " ".join(newWords) 
print(newSentence)