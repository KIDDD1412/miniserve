from transformers import GPT2LMHeadModel
model = GPT2LMHeadModel.from_pretrained("gpt2")
for name, p in model.state_dict().items():
    print(name, tuple(p.shape))