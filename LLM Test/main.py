import os
os.environ['HF_HOME'] = 'C:\\LLM'

import timeit

# Load model directly
from transformers import AutoTokenizer, AutoModelForCausalLM

#enables the model to run on GPU
device = "cuda"

tokenizer = AutoTokenizer.from_pretrained("allenai/OLMo-1B-0724-hf")
model = AutoModelForCausalLM.from_pretrained("allenai/OLMo-1B-0724-hf").to(device)



prompt = 0

while(prompt != "end"):
    maxlength = int(input("maxlength: \n>>"))
    # TO-DO: Automate prompts
    prompt = input("prompt: \n>>")
    if(prompt == "end"):
        break
    inputs = tokenizer(prompt, return_tensors="pt").to(device)

# Generate
    tic = timeit.default_timer()
    generate_ids = model.generate(inputs.input_ids, max_length=maxlength)
    toc = timeit.default_timer()
    print(tokenizer.batch_decode(generate_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0][len(prompt):])
    # TO-DO: Automate saving responses and response times
    print("Generationtime with params maxlength (" + str(maxlength) + ") and prompt (" + str(prompt) +
          ") was " + str(toc - tic))

print("Program terminated.")