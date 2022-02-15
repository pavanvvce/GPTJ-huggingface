import torch
from transformers import AutoTokenizer, GPTJForCausalLM
print("Dependencies imported")


# configurations for GPT-J

#tokenizer = AutoTokenizer.from_pretrained("gpt2")
tokenizer = AutoTokenizer.from_pretrained("EleutherAI/gpt-j-6B")
print("Loaded tokenizer")
#model = GPTJForCausalLM.from_pretrained("EleutherAI/gpt-j-6B", torch_dtype=torch.float16)
model = torch.load("gptj_float16.pt")
print("Model loaded")
model.parallelize()
print("Model parallelized")


from flask import Flask, request, render_template
import json
import main
import requests
import os
import gpt_util
import time
from flask_cors import CORS
import pandas as pd

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
     
    
        #print(input_text)
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
            #pass
            ##gptj comment
            #res=[]
            elStartTime = time.time()
            resList = []
            inputs = tokenizer(input_text, return_tensors="pt")
            input_ids = tokenizer(input_text, return_tensors="pt").input_ids.to("cuda")
            
            
            out_desiredLength = input_ids.size(dim=1) + top_k
            masks=inputs["attention_mask"].to("cuda")

            # single word/line generator
            if assistance == "content_assist":
                print("Content assist:\n")
                loop_list = [0, 1, 2]
                # for testing purpose, reducing the loop to 2 from 3
                for i in range(0,2):
                    execStartTime = time.time()
                    output = model.generate(
                    input_ids,
                    attention_mask=masks,
                    do_sample=True,
                    max_length=out_desiredLength,
                    temperature=gptj_temperature,
                    use_cache=True,
                    top_p=gptj_top_p,
                   # output_scores=True,
                    #repetition_penalty = 0.8
                    )
                    execEndTime = time.time()
                    print('Time to execute: ',execEndTime - execStartTime)
                    #resList += tokenizer.decode(output[0]).split("\n")
                    tempOut = tokenizer.batch_decode(output)[0]
                    #print("Scores: ", output.scores)
                    print('Tempout: ', tempOut)
                    #resList.append(tokenizer.batch_decode(output)[0])
                    # formatting the suggestions
                    resList += gpt_util.formatPrediction(input_text, tempOut)
                
                print("Result: ",resList)

            # for multi-line generator
            elif assistance == "code_generation":
                print("Code generation")
                out_desiredLength = input_ids.size(dim=1) + 400
                execStartTime = time.time()
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
                #resList += tokenizer.decode(output[0]).split("\n")
                tempOut = tokenizer.batch_decode(output)[0]
                print('Tempout: ', tempOut)
                #resList.append(tokenizer.batch_decode(output)[0])
                #not  formatting the suggestions
                resList.append(tempOut)

            #  for code generation using comments
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
                #eos_token_id=26
                )
                execEndTime = time.time()
                print('Time to execute: ',execEndTime - execStartTime)
                print("out length: ",out_desiredLength)
                #resList += tokenizer.decode(output[0]).split("\n")
                tempOut = tokenizer.batch_decode(output)[0]
                print('Tempout: ', tempOut)
                #resList.append(tokenizer.batch_decode(output)[0])
                #not  formatting the suggestions
                resList.append(tempOut)

            # for few shot learning
            elif assistance == "fewShotLearning":
                print("Few shot code generation")

                # getting the element first: which element does the user wants the code for
                intentsCollection = pd.read_csv("FewShotLearning_Comment_Elements.csv")
                codesCollection = pd.read_csv("FewShotLearning_Elems_Comm_Codes.csv")

                # sampling the code
                intentsCollection_df = intentsCollection.sample(frac = 0.3)

                # creating the format "comment:<comment>\nelement:<element>\m###"
                comments = "comment: "+intentsCollection_df.iloc[:,0].astype(str)
                elements = "element: "+ intentsCollection_df.iloc[:,1].astype(str)

                dataToSend = comments.astype(str) +"\n"+ elements.astype(str) + '\n###'
                dfConcatenated = dataToSend.values
                codesList = dfConcatenated.tolist()
                fCodes = "\n".join(codesList)
                fCodes += f"\ncomment: {input_text.strip('//')}\nelement: "

                # tokenizing the input
                inputs = tokenizer(fCodes, return_tensors="pt")
                input_ids = tokenizer(fCodes, return_tensors="pt").input_ids.to("cuda")

                out_desiredLength = input_ids.size(dim=1) + 20
                masks=inputs["attention_mask"].to("cuda")

                # calling the model for element name
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

                tempOut = tokenizer.batch_decode(output)[0]
                formattedResult = str(tempOut.split('###')[0]).strip()
                code_idx = formattedResult.find("element:")
                formattedResult = str(formattedResult[code_idx+9:]).strip('###')
                
                elementToSearch = formattedResult

                # seraching for the element in the data that we have
                extractedData = codesCollection.loc[codesCollection['Intended_Element'] == elementToSearch]
                extractedData = extractedData.iloc[:,[1,2]]
                print("data extracted")

                #sampling
                extractedData = extractedData.sample(frac = 0.5)
                comments = "comment: "+extractedData.iloc[:,0].astype(str)
                codes = "code: "+ extractedData.iloc[:,1].astype(str)

                dataToSend = comments.astype(str) +"\n"+ codes.astype(str) + '\n###'
                dfConcatenated = dataToSend.values
                codesList = dfConcatenated.tolist()
                fCodes = "\n".join(codesList)
                fCodes += f"\ncomment: {input_text.strip('//')}\ncode: "

                inputs = tokenizer(fCodes, return_tensors="pt")
                input_ids = tokenizer(fCodes, return_tensors="pt").input_ids.to("cuda")
                masks=inputs["attention_mask"].to("cuda")
                
                out_desiredLength = input_ids.size(dim=1) + top_k
                execStartTime = time.time()
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
                print('Time to execute: ',execEndTime - execStartTime)
                #resList += tokenizer.decode(output[0]).split("\n")
                tempOut = tokenizer.batch_decode(output)[0]
                print('Tempout: ', tempOut)
                formattedResult = str(tempOut.split('###')[0])
                code_idx = formattedResult.find("\ncode:")
                formattedResult = formattedResult[code_idx+1:]
                print("Formatted:\n", formattedResult)
                # the final result
                resList += gpt_util.formatPrediction_fewShot(input_text, tempOut)


                                
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
