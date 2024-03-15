from ctransformers import AutoModelForCausalLM, AutoConfig
import time

t = time.time()
role = "personal assistant"
question = "openAI's "

model_repo = "TheBloke/Mistral-7B-Instruct-v0.2-GGUF"

def init_model():
    config = AutoConfig.from_pretrained("TheBloke/Mistral-7B-Instruct-v0.1-GGUF")
    config.config.gpu_layers = 50
    # config.config.mlock = True
    config.config.context_length = 2048
    config.config.batch_size = 16
    config.config.max_new_tokens = 1024
    llm = AutoModelForCausalLM.from_pretrained("TheBloke/Mistral-7B-Instruct-v0.1-GGUF", model_file="mistral-7b-instruct-v0.1.Q5_K_M.gguf", model_type="mistral", config=config)
    llm._context = llm.tokenize("The F5 and the F5 Pro are almost identical in size and core features, but upon closer look at the specs sheet, we start to see how they differ in some niche aspects. Still, both phones are aimed towards the budget-conscious crowd, with the F5 Pro acting as an affordable flagship. The handset offers a few notable upgrades over its predecessor while maintaining the same price point (the F4 launched for €400). It's a great choice if you are looking for a capable midranger with some upper-tier features on the side. The standard F5 runs on a brand new Snapdragon 7+ Gen 2 chipset, which hopefully isn't just a rebranded Snapdragon 778G yet again. The handset also features a familiar 6.67-inch OLED panel with a 120Hz refresh rate and high maximum brightness as its more expensive sibling but at a lower resolution. The camera setups of both devices are identical, and the battery capacity is rated at around 5,000 mAh with support for blazing-fast 67W fast charging. The only difference is that the F5 Pro adds wireless charging to its list of features. It seems like no matter which device you decide to get, the user experience won't vary by much. But if wireless charging and a more future-proof SoC, such as the Snapdragon 8+ Gen 1, are up on your priority list, the few extra bucks for the F5 Pro might be worth it. Still, we believe that the F5 will get you 90% of where you want to be, making it presumably the better deal of the two. ")
    print("model load successful!")
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