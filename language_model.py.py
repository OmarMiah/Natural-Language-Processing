#import Libraries

import math 

# PART II 
# train.txt has 10000 sentences
# evaluate language models with test.txt 

#setting paths for files
testFile = "test.txt"
trainFile = "train.txt"

# initilizing counts for unigrams and bigrams
uniCount = {} # training courpus unigram count
biCount = {} # training corpus bigram count

def main():

#              ##### PRE-PROCESSING & TRAINING MODELS (1.1 & 1.2) #####
#
#  1. Pad each sentence in the training and test corpora 
#	 with start and end Symbols: <S> and </S> 
#
#  2. Lowercase all words in the training and test corpora. 
#  	 Note that the data already has been tokenized 
# 	 (i.e. the punctuation has been split off words).
#
#  3. Replace all words occurring in the training data once 
#  	 with the token <unk>. Every word in the test data not 
#  	 seen in training should be treated as <unk>.

	# setting and padding training data
	# unigram likelihood model 
	setData(trainFile,uniCount,"uni")
	padTrain(trainFile, uniCount)

	# setting and padding test data
	# bigram likelihood model
	padTest(testFile, uniCount)
	setData("padded_train.txt",biCount,"bi")

	print("###########################################################")
	print("#                                                         #")
	print("#                    QUESTIONS 1.3                        #")
	print("#                                             ~ Omar Miah #")
	print("###########################################################")

	print("\nQuestion 1")

#  	1. How many word types (unique words) are there in the training corpus? Please include
# 	  the padding symbols and the unknown token.

	wordTypesCount = 0 # UNIQUE
	for word in uniCount:
		wordTypesCount+= 1

	print("Number of word types: {}".format(wordTypesCount))
	print("```````````````````````````````````````````````````````````````")

	print("Question 2")

#   2. How many word tokens are there in the training corpus?	
	
	totalCount = 0 
	for count in uniCount.values():
		totalCount += count 

	print("Number of tokens: {}".format(totalCount))
	print("```````````````````````````````````````````````````````````````")
	
	print("Question 3")

#	3. What percentage of word tokens and word types in the test corpus did not occur in
#	   training (before you mapped the unknown words to <unk> in training and test data)?
#	   Please include the padding symbols in your calculations.

	missingUni = missingUnigrams(testFile, uniCount)
	print("Percent Missing Unigrams: {}%".format(missingUni))
	print("```````````````````````````````````````````````````````````````")
	
	print("Question 4")

#	4. Now replace singletons in the training data with <unk> symbol and map words (in the
#	   test corpus) not observed in training to <unk>. What percentage of bigrams (bigram
#	   types and bigram tokens) in the test corpus did not occur in training (treat <unk> as a
#      regular token that has been observed).

	missingBi = missingBigrams(testFile, biCount)
	print("Percent Missing Unigrams: {}%".format(missingBi))
	print("```````````````````````````````````````````````````````````````")
	
	print("Question 5")

#	5. Compute the log probability of the following sentence under the three models (ignore
#	   capitalization and pad each sentence as described above). Please list all of the parameters required to compute the probabilities and show the complete calculation. Which
#	   of the parameters have zero values under each model? Use log base 2 in your calculations. Map words not observed in the training corpus to the <unk> token.
#					• I look forward to hearing your reply .

	sentence = " <s> i look forward to hearing your reply . </s>"
	print("Sentence to compute: "+"\"" + sentence + "\""+'\n')

	# calculating log of unigrams
	print("Unigram:")
	logUni = logUniCalc(sentence,uniCount,totalCount) 
	print("Log Unigram Prob: {}\n".format(logUni))

	# calculating log of bigrams
	print("Bigram:")
	logBi = logBiCalc(sentence,uniCount,biCount)
	print("Log Bigram Prob: {}\n".format(logBi))

	# calculate log of bigrams with add one
	print("Bigram Add-One:")
	logBiaddOne = logBiAddOneCalc(sentence,uniCount,biCount)
	print("Log Bigram Add-One Prob: {}\n".format(logBiaddOne))
	print("```````````````````````````````````````````````````````````````")
	
	print("Question 6")

#	6.  Compute the perplexity of the sentence above under each of the models.
	
	print("Unigram", calcPerplexity('u', sentence, uniCount, biCount, totalCount))
	print("Bigram", calcPerplexity('b', sentence, uniCount, biCount, totalCount))
	print("Bigram Add One", calcPerplexity('bb', sentence, uniCount, biCount, totalCount))
	print("```````````````````````````````````````````````````````````````")
	
	print("Question 7")
	
# 	7. Compute the perplexity of the entire test corpus under each of the models. Discuss the
#	   differences in the results you obtained.

	corpus = ""
	with open("padded_test.txt", "r", encoding="utf8") as file: 
		for line in file:
			line = line.replace("<\s>", "<\s> <end>")
			corpus += line

	print("Unigram: ",calcPerplexity("u", corpus, uniCount, biCount, totalCount))
	print("Bigram: ", calcPerplexity("b", corpus, uniCount, biCount, totalCount))
	print("Bigram Add-One: ", calcPerplexity("bb", corpus, uniCount, biCount, totalCount))
	print("```````````````````````````````````````````````````````````````")

# inverse probability of test sets PP(W)
def calcPerplexity(type, sentence, uniCount, biCount, corpus):
	pl = 0 
	pm = 0

	for token in sentence.split():
		pm += 1

	if type == "u": # if unigram
		if logUniP(sentence, uniCount, corpus) == "undefined":
			return "undefined"
		else:
			pl = (1.0/pm) * logUniP(sentence, uniCount, corpus) 
	elif type == "b": # if bigram 
		if logBiP(sentence,uniCount,biCount) == "undefined":
			return "undefined"
		else:
			pl = (1.0/pm) * logBiP(sentence,uniCount,biCount)
	else:
		if logBiaddOneP(sentence,uniCount,biCount) == "undefined": # if bigram add one 
			return "undefined"
		else:
			pm = (1.0/pm) * logBiaddOneP(sentence,uniCount,biCount)

	return '%.3f'%(math.pow(2,-1*pl))

# Helper function for Bigram Add One Probability
def biAddOneP(sentence, uniCount, biCount):
	splits = sentence.split()
	result = 1.0
	v = checkTypes(uniCount)
	for i in range(len(splits) - 1):
		pair = (splits[i], splits[i+1])
		if pair not in biCount:
			result *= (1.0 / (float(uniCount[splits[i]]) + v))
		else: 
			result *= (biCount[pair] + 1.0) / (float(uniCount[splits[i]]) + v)
	return result

# Bigram Add One Probability 
def logBiaddOneP(sentence, uniCount, biCount):
	result = 0.0
	for line in sentence.split("\n"):
		outcome = biAddOneP(line, uniCount, biCount)
		if outcome != 0:
			result += math.log(outcome, 2)
	return result

# Bigram helper function 
def biP(sentence, uniCount, biCount):
	splits = sentence.split()
	result = 1.0 
	for i in range(len(splits) - 1):
		pairs = (splits[i], splits[i+1])
		print(pairs)
		if pairs not in biCount:
			return 0
		result *= (biCount[pairs] / float(uniCount[splits[i]]))
	return result

# Bigram probability 
def logBiP(sentence, uniCount, biCount):
	result = 0.0
	for line in sentence.split("\n"):
		outcome = biP(line, uniCount, biCount)
		if outcome == 0:
			return "undefined"
		else: result += math.log(outcome, 2)
	return result

# Helper function for unigram probability
def uniP(sentence, uniCount, totalCount):
	result = 1.0
	for word in sentence.split():
		result *= (uniCount[word] / totalCount)
	return result

# Unigram log probability 
def logUniP(sentence,uniCount, totalCount):
	result = 0.0
	for line in sentence.split("\n"):		
		outcome = uniP(line, uniCount, totalCount)
		if outcome == 0: 
			return "undefined"
		else:
			result += math.log(outcome, 2)
	return result

# counting unigrams
def checkTypes(uniCount):
	count = 0
	for word in uniCount:
		count += 1
	return count

# Helper function to calculate probability of tokens
def biAddOneCalc(sentence, uniCount, biCount):
	splits = sentence.split()
	prob = 1.0
	typeCount = checkTypes(uniCount)
	for i in range(len(splits) - 1):
		pair = (splits[i], splits[i+1])
		if pair not in biCount:
			print(pair,":  ",'%.3f'%(math.log((1.0 / (float(uniCount[splits[i]]) + typeCount)),2)))
			prob *= (1.0 / (float(uniCount[splits[i]]) + typeCount))
		else: 
			print(pair,":  ",'%.3f'%(math.log((biCount[pair] + 1.0) / (float(uniCount[splits[i]]) + typeCount),2)))
			prob *= (biCount[pair] + 1.0) / (float(uniCount[splits[i]]) + typeCount)
	return prob

# Log Calculator for Bigrams AddOne
def logBiAddOneCalc(sentences, uniCount, biCount):
	result = 0.0
	for sentence in sentences.split("\n"):
		outcome = biAddOneCalc(sentence, uniCount, biCount)
		if outcome != 0: 
			result += math.log(outcome, 2)
	return '%.3f'%(result)

# Helper function to calculate the probability of a token
def biCalc(sentence, uniCount, biCount):
	splits = sentence.split()
	prob = 1.0
	for i in range(len(splits) - 1):
		pair = (splits[i], splits[i+1])
		if pair not in biCount:
			print(pair," undefined")
			return 0
		print(pair,":  ",'%.3f'%(math.log(biCount[pair]/float(uniCount[splits[i]]),2)))
		prob *= (biCount[pair] / float(uniCount[splits[i]]))
	return prob

# Log Calculator for Bigrams
def logBiCalc(sentence, uniCount, biCount):
	result = 0.0
	for line in sentence.split("\n"):
		outcome = biCalc(sentence, uniCount, biCount)
		if outcome == 0: 
			return "404: Not Found"
		else:
			result += math.log(outcome, 2)
	return '%.3f'%(result)

# Helper function to calculate the probability of a token
def uniCalc(sentence, wordCount, size):
	prob = 1.0
	for word in sentence.split(): 
		print(word,":  ",'%.3f'%(math.log(wordCount[word]/size,2)))
		prob *= (wordCount[word] / size)
	return prob

# Log Calculator for Unigrams
def logUniCalc(sentences, wordCount, size):
	prob = 0.0
	for line in sentences.split("\n"):		
		sentenceProb = uniCalc(line, wordCount, size)
		if sentenceProb == 0: 
			return "undefined"
		else:
			prob += math.log(sentenceProb, 2)
	return '%.3f'%(prob)

# Percentage of Missing Bigrams
def missingBigrams(infile,count):
	missing = 0
	total = 0 
	with open(infile, "r", encoding="utf8") as file: 
		for line in file:
			splits = line.split()
			for i in range(len(splits) - 1):
				total += 1
				pair = (splits[i], splits[i+1])
				if pair not in count:
					missing += 1
	return '%.3f'%(missing/total*100)

# Percentage of Missing Unigrams
def missingUnigrams(infile,count):
	missing = 0
	total = 0 
	with open(infile, "r", encoding="utf8") as file: 
		for line in file:
			line = line.lower()
			for word in line.split():
				total+=1
				if word not in count:
					missing += 1
	return '%.3f'%(missing/total*100)
					
# we need to set the data in order to pad and counts words
def setData(input,count,model):
	with open(input, encoding="utf8") as file:
		for line in file: # obviously uni splitting for wordcount
			line = line.lower()
			if model == "uni":
				for each in line.split():
					if each not in count:
						count[each] = 1 
					else:
						count[each] += 1 
			else: # setting up for bi data to be parsed
				splits = line.split()
				for element in range(len(splits) - 1):
					pair = (splits[element], splits[element+1])
					if pair not in count:
						count[pair] = 1
					else:
						count[pair] += 1

# adding padding to sentence while simultaneously searching for foreign words
def padTest(testFile,wordCount):
	# creating output file
	modifiedFile = open("padded_"+ testFile, "w", encoding = "utf8")
	with open(testFile,"r",encoding="utf8") as file:
		# adding padded tokens to each sentence 
		for line in file:
			line = line.lower()
			modifiedFile.write("<s>")
			for each in line.split():
				# idenitifying unknown tokens and labeling with <unk>
				if each not in wordCount:
					modifiedFile.write(" <unk>")
				else:
					modifiedFile.write(" " + each)
			modifiedFile.write(" </s>\n")
	modifiedFile.close()

# setting the padded counts and iterating to find the 
def padTrain(trainFile, wordCount):
	# create the output folder
	modifiedFile = open("padded_" + trainFile, "w", encoding="utf8")
	# setting counts for our padded words
	wordCount["<unk>"] = 0 
	wordCount["<s>"] = 0
	wordCount["</s>"] = 0
	# scanning training file for padded tokens
	# setting values for unknown tokens 
	# removing the word from the training list
	with open(trainFile, "r",encoding="utf8") as file:
		for line in file:
			line = line.lower()
			wordCount["<s>"] += 1
			modifiedFile.write("<s>")
			for each in line.split():
				if wordCount[each] == 1:
					modifiedFile.write(" <unk>")
					wordCount["<unk>"] += 1
					del wordCount[each]
				else:
					modifiedFile.write(" " + each)
			wordCount["</s>"] += 1
			modifiedFile.write(" </s>\n")
	modifiedFile.close()

if __name__ == '__main__':
	main()