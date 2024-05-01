# outlines-chat

An example workflow for using local chat models with outlines (which was built when most models were based on text completions and before chat models were thing).

See test_outlines_chat.py for usage examples and outlines_chat.py for code.  

Three classes are introduced.

1) LLM:  This is a wrapper around the outlines model class.  It can take an existing outlines model as a argument or it can load one via exllamav2/huggingface.  The main functionality here is to apply a chat template to the messages datastructure.  The chat template can be a custom built outlines template.

2) SimpleChatHistory: This simply takes the last N chat messages that can fit into the token budget and uses that as context for the prompt.  You would replace this with your RAG solution.

3) ChatModel: This creates a custom generator method that is a bit easier to use for most common use cases. Besides temperature you can pass it regex or json to restrict output result.