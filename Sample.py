from ctransformers import AutoModelForCausalLM, AutoConfig
import time

t = time.time()
role = "Professional Storyteller"
question = "Write an essay on cheetahs in 200 words"

config = AutoConfig.from_pretrained("TheBloke/Mistral-7B-OpenOrca-GGUF")
config.config.gpu_layers = 50
config.config.mlock = True
config.config.context_length = 4096
config.config.batch_size = 16
config.config.max_new_tokens = 8192
llm = AutoModelForCausalLM.from_pretrained("TheBloke/Mistral-7B-Instruct-v0.1-GGUF", model_file="mistral-7b-instruct-v0.1.Q5_K_M.gguf", model_type="mistral", config=config)

prompt = f"<s>[INST]You are a {role} named Veronica. You will respond in a way that is most natural and relevant for a {role}, while making sure to follow any specific instructions given to you[/INST]Sure, I will now respond to the conversation as if I am a {role} and my answers will be context relevant and related to my role of being a {role}</s>[INST]{question}[/INST]"
response = llm(prompt)
print(response)
t = time.time() - t
print("Time = ", t)
