from typing import Any
import outlines
from huggingface_hub import snapshot_download

isa = isinstance

class ChatHistory:
    pass

class SimpleChatHistory(ChatHistory):

    def __init__(self, llm, system_instruction='', llm_budget=4096, size=64, user_role = 'user', llm_role='assistant'):
        self.system = []
        self.llm = llm
        self.set_system(system_instruction)
        self.data = []
        self.size = size
        self.llm_budget = llm_budget
        self.user_role = user_role
        self.llm_role = llm_role

    def reset(self):
        self.data = []

    def set_system(self,system_instruction):
        if system_instruction:
            self.system = [{'role':'system','content':system_instruction}] 
        else:
            self.system = []

    def check_size(self):
        if self.size:
            if len(self.data) > self.size:
                self.data = self.data[-self.size:]

    def append(self,msg):
        '''
        Append to history.
        '''
        self.data.append(msg)
        self.check_size()        
    
    def extend(self,msgs):
        '''
        Extend history.
        '''
        self.data.extend(msgs)
        self.check_size()   

    def replace(self,msgs):
        '''
        Replaces history with completely new set of messages, keeping the system message in place unless otherwise specified.
        '''
        if msgs and msgs[0]['role'] == 'system':
            self.set_system(system_instruction=msgs[0]['content'])
            self.data = msgs[1:]
        else:
            self.data = msgs
        self.check_size()   

    def build_prompt(self, prompt, prefix='', system_instruction = '', chat_template = None):
        '''
        This is where we could do a bunch of RAG stuff by searching the history and filtering out irrelevant stuff.
        As it is we just go through our history and add stuff until our LLM budget is used up.
        '''
        if system_instruction:
            system = [{'role':'system','content':system_instruction}] 
        else:
            system = self.system
            
        if prompt:
            current_prompt = [{'role':self.user_role,'content':prompt}]
        else:
            current_prompt = []

        history_size = len(self.data)
        i = 2 
        full_history = False
        count = 0
        while count < self.llm_budget:
            if i > history_size:
                full_history = True
                break
            prompt = self.llm.apply_chat_template(system + self.data[-i:] + current_prompt,
                                                  user_role=self.user_role, llm_role=self.llm_role , chat_template=chat_template) + prefix
            count = self.llm.count_tokens(prompt)
            i += 2
        if full_history:
            return self.llm.apply_chat_template(system + self.data + current_prompt, 
                                                user_role=self.user_role, llm_role=self.llm_role , chat_template=chat_template) + prefix
        else:
            # We are past our budget, so we step back one instruction and one response.
            return self.llm.apply_chat_template(system + self.data[-(i-2):] + current_prompt, 
                                                user_role=self.user_role, llm_role=self.llm_role , chat_template=chat_template) + prefix
        


class LLM:

    def __init__(self, model, revision='', device='cuda:0'):
        if isa(model,str):
            # Load a exllamav2 model
            self.model_name = model
            self.revision = revision
            # snapshot makes the initial loading of the model onto the GPU *MUCH* faster.
            self.model_directory = snapshot_download(repo_id=model, revision=revision)
            self.llm = outlines.models.exl2(self.model_directory,device=device)
        else:
            self.llm = model
        self.tokenizer = self.llm.tokenizer.tokenizer
        
    def __call__(self, *args: Any, **kwds: Any) -> Any:
        # TODO: add top_p, top_k etc.

        temp = kwds.get('temp',1.0)
        if 'temp' in kwds:
            del kwds['temp']
            
        regex = kwds.get('regex',"")
        if 'regex' in kwds:
            del kwds['regex']

        json = kwds.get('json',"")
        if 'json' in kwds:
            del kwds['json']
        
        if json and regex:
            raise Exception("Can't have a json and regex constraint at the same time.")

        sampler = build_sampler(temp)
        
        if regex:
            generator = outlines.generate.regex(self.llm,regex_str=regex,sampler=sampler)
        elif json:
            generator = outlines.generate.json(self.llm,json=json,sampler=sampler)
        else:
            generator = outlines.generate.text(self.llm,sampler)
        text = generator(*args,**kwds)
        
        return text

    def tokenize(self,text):
        return self.tokenizer.tokenize(text)

    def apply_chat_template(self, messages, user_role='user', llm_role='assistant', chat_template=None):
        # Use a custom outlines template if passed in.
        if isa(chat_template,outlines.prompts.Prompt):
            system = ''
            if messages and messages[0]['role'] == 'system':
                system = messages[0]['content']
                messages = messages[1:]
            #history = [x['content'] for x in messages]
            query = messages[-1]['content']
            user_role = messages[-1]['role']
            messages = messages[:-1]
            return chat_template(query,system,messages,user_role,llm_role)
        
        # Load one that came with the model.
        return self.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    
    def count_tokens(self,text):
        return len(self.tokenize(text))
 
def build_sampler(temp=0.8):
    if temp > 0.0:
        sampler = outlines.samplers.multinomial(temperature=temp)
    else:
        sampler = outlines.samplers.greedy()
    return sampler
        
class ChatModel:

    def __init__(self, llm=None, chat_template=None):
        
        if llm is None:
            self.llm = LLM(model_name="bartowski/openchat-3.5-0106-exl2",revision="4_25")   
        else:
            self.llm = llm

        self.chat_template = chat_template

    def __call__(self, prompt,
                       history=None, update_history=False, 
                       temp=0.8, stop_at=[], max_new_tokens=4096,
                       doc_tag=None, prefix='',
                       regex=None, json=None ):
        
        original_prompt = prompt
        if history is None:
            # If no history is passed in we build a temporary one.
            history = SimpleChatHistory(self.llm)
            update_history = False

        prompt = history.build_prompt(prompt, prefix= prefix, chat_template = self.chat_template)

        if isa(stop_at,str):
            stop_at = [stop_at] + [self.llm.tokenizer.eos_token]
        else:
            stop_at = stop_at + [self.llm.tokenizer.eos_token]
        
        if doc_tag:
            prefix = '<'+doc_tag+'>' + prefix
            stop_at = stop_at + ['</'+doc_tag+'>']

        response = self.llm(prompt, temp = temp, stop_at = stop_at, max_tokens = max_new_tokens, regex = regex, json = json)
        
        if update_history:
            history.append({'role':history.user_role,'content':original_prompt})
            history.append({'role':history.llm_role,'content': prefix + response})
        
        if extract_doc and doc_tag:
            return extract_doc(prefix + response, doc_tag)
        
        return prefix + response


def extract_doc(response,tag):
    '''
    Look for a pair of tags and return content in the middle
    '''
    i = response.find('<'+tag+'>')
    if i > -1:
        response = response[i+len('<'+tag+'>'):]
    i = response.rfind('</'+tag+'>')
    if i > -1:
        response = response[:i]
    return response.strip()