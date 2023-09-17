from flask import Flask, request, jsonify
from vllm import LLM, SamplingParams

app = Flask(__name__)

model = "lmsys/vicuna-13b-v1.3"
llm = LLM(model=model)

system_prompt = "A chat between a curious user and an artificial intelligence assistant. \nThe assistant gives helpful, detailed, and polite answers to the user's questions."

addon_prompt = ""


def get_prompt(human_prompt):
    prompt_template = f"{system_prompt}\n{addon_prompt} \n\nUSER: {human_prompt} \nASSISTANT: "
    return prompt_template


def generate(text):
    prompt = get_prompt(text)
    sampling_params = SamplingParams(
        max_tokens=100,
        # do_sample=True,
        temperature=0.7,
        top_p=0.95,
        top_k=50,
    )
    outputs = llm.generate([prompt], sampling_params)
    return outputs


def parse_text(outputs):
    for output in outputs:
        prompt = output.prompt
        generated_text = output.outputs[0].text
    return prompt, generated_text


@app.route("/")
def test():
    return {"test": "works"}


@app.route("/generate", methods=["POST"])
def analyze_text():
    input_data = request.json
    text = input_data["text"]

    input = get_prompt(text)
    result = generate(input)

    prompt, response = parse_text(result)

    response_data = {"prompt": prompt, "response": response}

    return jsonify(response_data)


if __name__ == "__main__":
    app.run()
