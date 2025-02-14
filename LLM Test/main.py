import languageModel
import jsonWriter


filepath = ".\\Evaluations\\evaluation.json"
datablocks = jsonWriter.DataFile(filepath)

modelName = "Qwen/Qwen2.5-Coder-1.5B-Instruct"


hardware="Lab-Laptop"


prompts = ["Remember this number sequence: 090993.",
          "Explain the stock market in as much detail as possible.",
          "Wer war Martin Luther?",
          "Which OS do Nokia Smartphones use?",
          "If a plane crashes in the US and lands on the Canada-US border, where do you bury the survivors?",
          "Envision a plot for a blockbuster movie.",
          "What number sequence did I ask you to remember?"]


model = languageModel.hfTransformer(modelName, "C:\\LLM", "C:\\LLM\\.cache\\huggingface\\token", True)


for prompt in prompts:
   response, elapsed_time, maxlength = model.generateOutput(prompt, 100)
   datablocks.append(hardware, modelName, "maxlength: " + str(maxlength), elapsed_time, prompt, response)


datablocks.write()

