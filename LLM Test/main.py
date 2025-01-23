import os

import csvWriter

os.environ['HF_HOME'] = 'C:\\LLM'


import timeit


# Load model directly
from transformers import AutoTokenizer, AutoModelForCausalLM
from huggingface_hub import login
hfFile = open("C:\\LLM\\.cache\\huggingface\\token")
login(token=hfFile.readline())

# set values
maxlength = 100


#adjustable variables
#enables the model to run on GPU
device = "cuda"
transformerLink = "Qwen/Qwen2.5-Coder-1.5B-Instruct"

def loadModel(modelCode, device):
    _tokenizer = AutoTokenizer.from_pretrained(modelCode)
    _model = AutoModelForCausalLM.from_pretrained(modelCode).to(device)
    return _tokenizer, _model

#def loadModel(modelCode):
#    _model = AutoModelForCausalLM.from_pretrained(modelCode, trust_remote_code=True)
#    return _model

tokenizer, model = loadModel(transformerLink, device)

filepath = "C:\\LLM\\evaluation\\evaluations.json"
datablocks = csvWriter.DataFile(filepath)

prompts = ["Explain the stock market in as much detail as possible.",
           "Wer war Martin Luther?",
           "Which OS do Nokia Smartphones use?",
           "If a planes crashes in US and lands on the Canada-US border, where do you bury the survivors?",
           "Envision a plot for a blockbuster movie."]

for prompt in prompts:
    inputs = tokenizer(prompt, return_tensors="pt").to(device)

# Generate
    tic = timeit.default_timer()
    generate_ids = model.generate(inputs.input_ids, max_length=maxlength)
    toc = timeit.default_timer()
    response = tokenizer.batch_decode(generate_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0][len(prompt):]
    # TO-DO: Automate saving responses and response times
    print("Generationtime with params maxlength (" + str(maxlength) + ") and prompt (" + str(prompt) +
          ") was " + str(toc - tic))
    print(response)
    elapsed_time = str(toc - tic)

    datablocks.append(transformerLink, "maxlength: " + str(maxlength), elapsed_time, prompt, response)

datablocks.write()

print("Program terminated.")