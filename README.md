# outlines-chat

An example workflow for using local chat models with outlines.  Outlines was built when most local LLM models did text completions and before local LLM chat models became standard.

This posted here purely for educational purposes.  The code was extracted from a larger project and cleaned up.  As such it hasn't been fully tested in its current form and was designed for my own specific use case (I didn't have others in mind). 

See [test_outlines_chat.py](test_outlines_chat.py) for some basic usage examples.  See [outlines_chat.py](outlines_chat.py) for code.  

Three classes are introduced in outlines_chat.py:

1) **LLM**:  This is a wrapper around the outlines model class.  It can take an existing outlines model as a argument or it can load one via exllamav2/huggingface.  The main functionality here is to apply a chat template to the messages datastructure.  The chat template can be the one that comes with the model or it can be a custom built template in the form of an outlines prompt object.

2) **SimpleChatHistory**: This simply takes the last N chat messages that can fit into the token budget and uses that as context for the prompt.  You would replace this with your RAG solution.

3) **ChatModel**: This creates a custom text generator that is a bit easier to use for most my common use cases. Besides the temperature parameter you can pass it regex or json to restrict the output to a defined schema / format.
