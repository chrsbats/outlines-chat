
from outlines_chat import LLM, ChatModel, SimpleChatHistory
import outlines

llm = LLM("bartowski/openchat-3.5-0106-exl2",revision="5_0")


### EXAMPLE 1 - Basic chat mode with system prompt and history.

# If you don't provide a ChatModel with a custom chat_template it will use the Jinja template that are automatically provided with the model from huggingface.
model = ChatModel(llm)
history = SimpleChatHistory(llm, system_instruction="You only respond in French.", user_role = 'user', llm_role='assistant')
text = model("Define AI in a short sentence.", history=history, temp=0.0, update_history=True, stop_at=['.'])
assert text.startswith("L'intelligence artificielle est un système")
# One message for user, one for assistant
assert len(history.data) == 2


### EXAMPLE 2 - Chat with custom chat template.

# Openchat models have a math reasoning mode. We will use this to do some math.
@outlines.prompt
def math_template(query,system,messages,user_role,llm_role):
    '''<s>{% for message in messages %}
    {{ 'Math Correct ' + message['role'].title() + ': ' + message['content'] + '<|end_of_turn|>'}}
    {% endfor %}
    {{ 'Math Correct ' + user_role.title() + ': ' + query + '<|end_of_turn|>'}}
    {{ 'Math Correct ' + llm_role.title() + ': ' }}'''

math_model = ChatModel(llm,chat_template=math_template)
# We dont need to keep track of history if we dont want it.
# Current bug exists in outlines that causes it to sometimes not respect the chat stop tokens if they have been added as special tokens so we add a stop at.
text = math_model("10.3 − 7988.8133 =", temp=0.0, stop_at=['Math Correct'])
# Strip the stop at phrase.
text = text[:text.rfind('Math Correct')].strip()
assert text == "10.3 - 7988.8133 = -7978.5133"


### EXAMPLE 3 - Chat with regex output.

text = model("What is the IP address of the Google DNS servers?", temp=0.0, max_new_tokens=30)
assert text.startswith("The IP addresses of the Google DNS servers are")

ip_regex = r"((25[0-5]|2[0-4]\d|[01]?\d\d?)\.){3}(25[0-5]|2[0-4]\d|[01]?\d\d?)"
text = model("What is the IP address of the Google DNS servers?", temp=0.0, regex=ip_regex)
assert text == "8.8.8.8"
