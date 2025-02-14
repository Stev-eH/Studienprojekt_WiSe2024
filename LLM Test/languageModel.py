from transformers import AutoTokenizer, AutoModelForCausalLM
from huggingface_hub import login
import os
import timeit


class hfTransformer:
   modelname = ""
   tokenFile = ""
   tokenizer = ""
   model = ""
   device = "cuda"
   verbose = False
   modelPath = ""


   def __init__(self, modelname, modelPath="", tokenFile="",verbose=False):
       self.modelname = modelname
       self.tokenFile = tokenFile
       self.verbose = verbose
       self.modelPath = modelPath


       if(self.tokenFile != ""):
           login(token=open(tokenFile, "r").readline())


       if(self.modelPath != ""):
           os.environ['HF_HOME'] = self.modelPath


       self.tokenizer, self.model = self.loadModel(self.modelname, self.device)




   def loadModel(self, modelCode, device):
       _tokenizer = AutoTokenizer.from_pretrained(modelCode)
       _model = AutoModelForCausalLM.from_pretrained(modelCode).to(self.device)
       return _tokenizer, _model


   def generateOutput(self, prompt, maxlength=0):
       if(maxlength == 0):
           maxlength = len(prompt)


       input = self.tokenizer(prompt, return_tensors="pt").to(self.device)


       #Generate the output
       tic = timeit.default_timer()
       generate_ids = self.model.generate(input.input_ids, max_length=maxlength)
       toc = timeit.default_timer()
       response = self.tokenizer.batch_decode(generate_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)[
                      0][len(prompt):]
       elapsed_time = str(toc - tic)
       if self.verbose:
           print("Generationtime with params maxlength (" + str(maxlength) + ") and prompt (" + str(prompt) +
             ") was " + elapsed_time)
           return response, elapsed_time, maxlength

