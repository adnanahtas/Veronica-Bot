from ctransformers import AutoModelForCausalLM, AutoConfig
import time

t = time.time()
role = "Professor"
question = "What is Life"

model_repo = "TheBloke/Mistral-7B-Instruct-v0.2-GGUF"

def init_model():
    config = AutoConfig.from_pretrained("TheBloke/Mistral-7B-Instruct-v0.2-GGUF")
    config.config.gpu_layers = 50
    # config.config.mlock = True
    config.config.context_length = 256
    config.config.batch_size = 8
    config.config.max_new_tokens = 1024
    llm = AutoModelForCausalLM.from_pretrained(model_repo, model_file="mistral-7b-instruct-v0.2.Q5_K_M.gguf", model_type="mistral", config=config)
    return llm

def generate(model, role, question):
    prompt = f"<s>[INST]You are a {role} named Veronica. You will respond in a way that is most natural and relevant for a {role}[/INST]Sure, I will now respond to the conversation as if I am a {role} and my answers will be context relevant and related to my role of being a {role}</s>[INST]{question}[/INST]"
    response = model(prompt)
    return response

m = init_model()
op = generate(m, role, question)
print(op)
t = time.time() - t
print("Time = ", t)