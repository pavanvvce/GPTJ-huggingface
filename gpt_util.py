def remove_prefix(text, prefix):
    return text[text.startswith(prefix) and len(prefix):]

def formatPrediction(text_sentence, prediction):
    predictionResult = []

    wordPrediction = remove_prefix(prediction,text_sentence)
    print("Word predictions are: ",  wordPrediction)
    wordPrediction =  str(wordPrediction.strip())
    if  wordPrediction.startswith('//'):
         wordPrediction =  wordPrediction.split("\n")[1]
    if(len(wordPrediction.strip())>0 and (wordPrediction[0] not in predictionResult)):
        print('First word: ', wordPrediction.split("\n"))
        entireLine = wordPrediction.split("\n")[0]
        entireSentence = entireLine.split(";")[0]
        entireBlock = entireSentence.split("{")[0]
        entireBlock = entireBlock.split("(")[0]
        entireWord = entireBlock.strip().split(" ")[0]
        predictionResult.append(entireWord)
        predictionResult.append(entireSentence)
        #predictionResult.append(wordPrediction  )
        #predictionResult.append(wordPrediction.split("\n")[0].split("{")[0].split(";")[0].split(" ")[0]) 
    print("Returning: ", predictionResult)
    return predictionResult

def formatPrediction_fewShot(text_sentence, prediction):
    predictionResult = []
    #print("input was: ", text_sentence)
    print('*'*100)
    wordPrediction = remove_prefix(prediction.strip(" "),text_sentence)
    print("length of wP: ", len(wordPrediction))

    # check if the data is there or not
    print(prediction.startswith(text_sentence))

    wordPrediction = wordPrediction.strip('###').strip("\n").strip()
    print("after removing the initials: \n", wordPrediction)
    splittedData = wordPrediction.split("\n")
    t1 = splittedData[-1]
    if len(str(t1).strip()) == 0 or str(t1).strip() == "code:":
        wordPrediction = ""
    if len(str(wordPrediction).strip())== 0:
        wordPrediction = ""
    predictionResult = [wordPrediction]
    #print("wordPrediction:\n",wordPrediction)
    return predictionResult

