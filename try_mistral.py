import os 

from mistral_inference.transformer import Transformer
from mistral_inference.generate import generate

from mistral_common.tokens.tokenizers.mistral import MistralTokenizer
from mistral_common.protocol.instruct.messages import UserMessage
from mistral_common.protocol.instruct.request import ChatCompletionRequest

# load tokenizer
mistral_tokenizer = MistralTokenizer.from_file("data/harvAI/mistral7B/tokenizer.model.v3")
# chat completion request
completion_request = ChatCompletionRequest(messages=[UserMessage(content="Que sait tu du droit francais ?")])
# encode message
tokens = mistral_tokenizer.encode_chat_completion(completion_request).tokens
# load model
model = Transformer.from_folder("data/harvAI/mistral7B")
# generate results
out_tokens, _ = generate([tokens], model, max_tokens=64, temperature=0.5, eos_id=mistral_tokenizer.instruct_tokenizer.tokenizer.eos_id)
# decode generated tokens
result = mistral_tokenizer.instruct_tokenizer.tokenizer.decode(out_tokens[0])
print(result)


# from mistral_common.protocol.instruct.messages import UserMessage
# from mistral_common.protocol.instruct.request import ChatCompletionRequest
# from mistral_common.protocol.instruct.tool_calls import Function, Tool

# completion_request = ChatCompletionRequest(
#     tools=[
#         Tool(
#             function=Function(
#                 name="get_current_weather",
#                 description="Get the current weather",
#                 parameters={
#                     "type": "object",
#                     "properties": {
#                         "location": {
#                             "type": "string",
#                             "description": "The city and state, e.g. San Francisco, CA",
#                         },
#                         "format": {
#                             "type": "string",
#                             "enum": ["celsius", "fahrenheit"],
#                             "description": "The temperature unit to use. Infer this from the users location.",
#                         },
#                     },
#                     "required": ["location", "format"],
#                 },
#             )
#         )
#     ],
#     messages=[
#         UserMessage(content="What's the weather like today in Paris?"),
#         ],
# )