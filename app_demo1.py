import torch
from transformers import AutoTokenizer, GPTJForCausalLM
print("Dependencies imported")


# configurations for GPT-J

tokenizer = AutoTokenizer.from_pretrained("EleutherAI/gpt-j-6B")
print("Loaded tokenizer")

# load the model from local(Model has been previously downloaded)
if torch.cuda.is_available():
    print("cuda")
    #model =  GPTJForCausalLM.from_pretrained("EleutherAI/gpt-j-6B").cuda()
else:
    print("not cuda")
    #model =  GPTJForCausalLM.from_pretrained("EleutherAI/gpt-j-6B")
    
model = torch.load("/home/pavan/gptjnew/gpt-j-6B.pt")
print("Model loaded")

model.eval()


from flask import Flask, request, render_template
import json
import main
import requests
import os
import gpt_util
import time
from flask_cors import CORS
import pandas as pd
from thefuzz import process
from thefuzz import fuzz

##
app = Flask(__name__)
CORS(app)

# give option as gpt3 for gpt3 exection 
# else give otpion as gptj for gptj mode of exection
# if none of the option is given default mode would execute
option = "gptj"
#option = "default"
# dont ship the below line / use their gpt3 api_key
gpt3_api_key="12312312312"


# testing purpose: Priyanka
assistance = "content_assist"

@app.route('/')
def index():
    return render_template('index.html')


@app.route('/get_end_predictions', methods=['post'])
def get_prediction_eos():
    try:
        input_text = request.json['input_text']
     
        top_k = request.json['top_k']

        # assistent type:
        assistance = request.json['assistance']

        # setting the temperature and top_p from the request for finding the optimal value
        gptj_temperature = request.json["temperature"]
        gptj_top_p = request.json["top_p"]
     
        suggestion_count=3
        # if(request.json['suggestion_count']):
        #     suggestion_count=request.json['suggestion_count']
        #print(suggestion_count)
        if(option == "gpt3"):
            pass
                # openai.api_key =gpt3_api_key
                # responseGpt = openai.Completion.create(
                #                 engine="davinci",
                #                 n=5,
                #                 prompt=input_text[-100:],
                #                 temperature=0.7,
                #                 max_tokens=30,
                #                 top_p=1,
                #                 frequency_penalty=0,
                #                 presence_penalty=0
                #                 )
                # responseArray=[]
                # for i in responseGpt.choices:
                #    # print(i)
                   
                #     word=i["text"]
                #     if "\n" in word:
                #         word=word.split("\n")[0]
                #     if ";" in word:
                #         word=word.split(";")[0]+";"

                #     responseArray.append(word.strip())
                # res={"prediction":list(set(responseArray))}
        elif(option == "gptj"):
            elStartTime = time.time()
            resList = []

            inputs = tokenizer(input_text, return_tensors="pt")
            #input_ids = tokenizer(input_text, return_tensors="pt").input_ids.to("cuda")
            input_ids = tokenizer(input_text, return_tensors="pt")
            out_desiredLength = input_ids.size(dim=1) + top_k
            #masks=inputs["attention_mask"].to("cuda")
            masks=inputs["attention_mask"]
            # block with the logic to generate code for code completion
            if assistance == "content_assist":
                print("Content assist:\n")
                # for testing purpose, reducing the loop to 2 from 3
                for i in range(0,2):
                    execStartTime = time.time()

                    # Generate predictions
                    output = model.generate(
                            input_ids,
                            attention_mask=masks,
                            do_sample=True,
                            max_length=out_desiredLength,
                            temperature=gptj_temperature,
                            use_cache=True,
                            top_p=gptj_top_p,
                            
                        )
                    execEndTime = time.time()
                    print('Time to execute: ',execEndTime - execStartTime)
                    
                    decodedOutput = tokenizer.batch_decode(output)[0]
                    
                    print('Decoded Output\n: ', decodedOutput )
                    
                    # formatting the suggestions
                    resList += gpt_util.formatPrediction(input_text, decodedOutput )
                
                print("Result: ",resList)

            elif assistance == "code_generation":
                print("Code generation")
                out_desiredLength = input_ids.size(dim=1) + 400
                execStartTime = time.time()

                # generate predictions
                output = model.generate(
                        input_ids,
                        attention_mask=masks,
                        do_sample=True,
                        max_length=out_desiredLength,
                        temperature=gptj_temperature,
                        use_cache=True,
                        top_p=gptj_top_p
                        )

                execEndTime = time.time()
                print('Time to execute: ',execEndTime - execStartTime)
                
                decodedOutput  = tokenizer.batch_decode(output)[0]
                print('Decoded Output : ', decodedOutput )
                
                resList.append(decodedOutput )

            # block to generate code when some code and comments are given
            elif assistance == "code_assist_generation":
                print("Code generation")

                out_desiredLength = input_ids.size(dim=1) + top_k
                execStartTime = time.time()
                
                output = model.generate(
                        input_ids,
                        attention_mask=masks,
                        do_sample=True,
                        max_length=out_desiredLength,
                        temperature=gptj_temperature,
                        use_cache=True,
                        top_p=gptj_top_p,
                        )

                execEndTime = time.time()
                print('Time to execute: ',execEndTime - execStartTime)
                
                decodedOutput  = tokenizer.batch_decode(output)[0]
                print('Decoded Output : ', decodedOutput )
                
                resList.append(decodedOutput )

            # block to implement few shot learning: fetch the data from CSV file and use that for inference
            elif assistance == "fewShotLearning":
                print("Few shot code generation")

                input_text = input_text.strip("//")

                # getting the element first: which element does the user wants the code for
                intentsCollection = pd.read_csv("FewShotLearning_Comment_Elements.csv")
                codesCollection = pd.read_csv("FewShotLearning_Elems_Comm_Codes.csv")

                # using the fuzz to find the intended element based on the distace between them
                allComments = intentsCollection["Comment"]

                matchingComment = process.extract(
                    input_text,
                    allComments,
                    scorer=fuzz.token_sort_ratio,
                    limit=1,
                )

                onlyComment = matchingComment[0][0]
                elementToSearch = intentsCollection[intentsCollection["Comment"] == onlyComment]["Intended_Element"].tolist()[0]

                print(f"intended element: {elementToSearch}")

                # seraching for the element in the data that we have
                extractedData = codesCollection.loc[codesCollection['Intended_Element'] == elementToSearch]
                '''
                # using the fuzz to find the most relevant comments
                comments = extractedData.iloc[:, 1].astype(str)
                matchingCodes = process.extract(
                    input_text.strip("//"),
                    comments,
                    scorer=fuzz.token_sort_ratio,
                    limit=12,
                )
                filteredComments = list(x[0] for x in matchingCodes)
                extractedData = codesCollection[codesCollection["Comment"].isin(filteredComments)]
                '''
                # extracting only Codes and comments
                extractedData = extractedData.iloc[:,[1,2]]
                
                comments = "comment: "+extractedData.iloc[:,0].astype(str)
                codes = "code: "+ extractedData.iloc[:,1].astype(str)

                dataToSend = comments.astype(str) +"\n"+ codes.astype(str) + '\n###'
                dfConcatenated = dataToSend.values
                codesList = dfConcatenated.tolist()
                fCodes = "\n".join(codesList)
                fCodes += f"\ncomment: {input_text}\ncode: "
                #print("Few shot data for codes: ", fCodes)
                extraTokens = 50

                inputs = tokenizer(fCodes, return_tensors="pt")
                #input_ids = tokenizer(fCodes, return_tensors="pt").input_ids.to("cuda")
                input_ids = tokenizer(fCodes, return_tensors="pt").input_ids
                #masks=inputs["attention_mask"].to("cuda")
                masks=inputs["attention_mask"]
                
                out_desiredLength = input_ids.size(dim=1) + extraTokens
                print("out_desiredLength:",out_desiredLength) 
                execStartTime = time.time()
                try:
                    output = model.generate(
                            input_ids,
                            attention_mask=masks,
                            #do_sample=True,
                            max_length=out_desiredLength,
                            temperature=gptj_temperature,
                            #use_cache=True,
                            top_p=gptj_top_p,
                            eos_token_id= 21017,
                            return_full_text=False
                            #eos_token_id=26
                        )
                    execEndTime = time.time()
                    #print('Time to execute: ',execEndTime - execStartTime)
                    #print("out length: ",out_desiredLength)
                    
                    decodedOutput = tokenizer.batch_decode(output)[0]
                    print('Decoded Output: ', decodedOutput)

                    fRes = gpt_util.formatPrediction_fewShot(fCodes, decodedOutput)
                    formattedResult = "".join(fRes)
                    print("Formatted:\n", formattedResult)

                    resList += [formattedResult]
                except Exception as e:
                    print("Encountered error while getting results: ",e)


                                
            res = {'prediction':sorted(list(set(resList)), key=len) }
            print("result: ", res)
            elEndTime = time.time()
            print('Total execution time: ', elEndTime - elStartTime)
    ##end of gptj comment       
        
        else:
           # print("InputText")
           # print(input_text)
            res = main.get_all_predictions(input_text, top_clean=int(top_k),suggestionsCount=int(suggestion_count))   
           
       
        return app.response_class(response=json.dumps(res), status=200, mimetype='application/json')
    except Exception as error:
        err = str(error)
       # print(err)
        return app.response_class(response=json.dumps(err), status=500, mimetype='application/json')



if __name__ == '__main__':
    app.run(host='0.0.0.0', debug=True, port=8000, use_reloader=False)
