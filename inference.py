from transformers import AutoTokenizer, AutoConfig, AutoModel
from mm_model import (
    OrpheusConfig,
    OrpheusForConditionalGeneration,
)

AutoConfig.register("orpheus", OrpheusConfig)
AutoModel.register(OrpheusConfig, OrpheusForConditionalGeneration)

model_name = "amuvarma/3b-zuckreg-convo-projsnactune"
model = AutoModel.from_pretrained(model_name)

def generate_output(prompt):
    inputs = tokenizer.encode(prompt, return_tensors="pt").to(model.device)
    print("inputs", inputs)
    
    # Generate output
    outs = model.generate(
        inputs,
        max_new_tokens=500,
        temperature=0.7,
        repetition_penalty=1.1,
        top_p=0.9,
        eos_token_id=128258,
    )
    
    total_tokens = outs.shape[1]
    elapsed_time = time.time() - start_time
    tokens_per_second = total_tokens / elapsed_time
    print(f"Prompt: {prompt!r}")
    print(f"Tokens per second: {tokens_per_second:.2f}")
    print(f"Total tokens: {total_tokens}")
    print(f"Time elapsed: {elapsed_time:.2f}s")
    
    # return generated_text

prompt = "Here is a short story about a dragon:"
generate_output(prompt)