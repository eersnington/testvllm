from vllm import LLM, SamplingParams
import time
model = "lmsys/vicuna-13b-v1.3"
llm = LLM(model=model)

prompt = "What is the difference between nuclear fission and nuclear fusion."
sampling_params = SamplingParams(
    max_tokens=100,
    # do_sample=True,
    temperature=0.7,
    top_p=0.95,
    top_k=50,
)

start = time.time()
outputs = llm.generate([prompt], sampling_params)
end = time.time()

for output in outputs:
  generated_text = output.outputs[0].text

print(generated_text)
print(len(generated_text)/(end-start),"tokens/s")
print("time taken:", (end-start))
