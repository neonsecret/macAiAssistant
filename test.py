import os

from custom_functions import create_tool_schema, AssistantFunctions, execute_function_call

os.chdir("functionary")
from llama_cpp import Llama

from functionary.prompt_template import get_prompt_template_from_tokenizer
from transformers import AutoTokenizer

if __name__ == '__main__':
    tools = create_tool_schema(AssistantFunctions)

    # You can download gguf files from https://huggingface.co/meetkai/functionary-small-v2.5-GGUF
    llm = Llama.from_pretrained('meetkai/functionary-small-v2.5-GGUF',
                                "*Q4*",
                                n_ctx=8192,
                                n_gpu_layers=-1,
                                verbose=False)
    messages = [{"role": "user", "content": "How much ram do I have?"}]

    # Create tokenizer from HF. We should use tokenizer from HF to make sure that tokenizing is correct
    # Because there might be a mismatch between llama-cpp tokenizer and HF tokenizer and the model was trained using HF tokenizer
    tokenizer = AutoTokenizer.from_pretrained(
        "meetkai/functionary-small-v2.5-GGUF", legacy=True
    )
    # prompt_template will be used for creating the prompt
    prompt_template = get_prompt_template_from_tokenizer(tokenizer)

    # Before inference, we need to add an empty assistant (message without content or function_call)
    messages.append({"role": "assistant"})

    # Create the prompt to use for inference
    prompt_str = prompt_template.get_prompt_from_messages(messages, tools)
    token_ids = tokenizer.encode(prompt_str)

    gen_tokens = []
    # Get list of stop_tokens
    stop_token_ids = [
        tokenizer.encode(token)[-1]
        for token in prompt_template.get_stop_tokens_for_generation()
    ]

    # We use function generate (instead of __call__) so we can pass in list of token_ids
    for token_id in llm.generate(token_ids, temp=0):
        if token_id in stop_token_ids:
            break
        gen_tokens.append(token_id)

    llm_output = tokenizer.decode(gen_tokens)

    # parse the message from llm_output
    result = prompt_template.parse_assistant_response(llm_output)
    print(result)

    for tool_call in result['tool_calls']:
        result = execute_function_call(tool_call)
        print(f"Function {tool_call['function']['name']} returned: {result}")
