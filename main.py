
import tensorflow as tf




top_k = 50
# for old model change the path to OldModel folder
one_step_reloaded = tf.saved_model.load('NewModel/one_step')



    
def remove_prefix(text, prefix):
    # if prefix[-1].isspace():
    #     print("hi")
    return text[text.startswith(prefix) and len(prefix):]
    # else:
    #     print("hello")
       
    #     return text[text.startswith(prefix) and len(prefix.rsplit(' ', 1)[0])+1:].strip()


def get_prediction(text_sentence,top_clean):
    
    states = None
    next_char = tf.constant([text_sentence])
    total_range=len(text_sentence)+top_clean
    result = [next_char]

    for n in range(total_range):
        next_char, states = one_step_reloaded.generate_one_step(next_char, states=states)
        if( next_char=="\n"):
            break
        result.append(next_char)
    prediction = (tf.strings.join(result)[0].numpy().decode("utf-8"))
    return prediction

def get_all_predictions(text_sentence, top_clean=50, suggestionsCount=3):

    predictionResult=[]

    for i in range(0,suggestionsCount):
        prediction= get_prediction(text_sentence,top_clean)
        wordPrediction=remove_prefix(prediction,text_sentence)

       # print(wordPrediction)
        if(len(wordPrediction)>0 and (wordPrediction[0] not in predictionResult)):
            predictionResult.append(wordPrediction  )
            predictionResult.append(wordPrediction.split(" ")[0]) 
            

    return {'prediction':sorted(list(set(predictionResult)), key=len) }
           
