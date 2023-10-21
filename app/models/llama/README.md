This README documents the first generated conversation between an AI assistant named Stacy and a user, facilitated by the integration of LLaMa and TinyGrad within the Townhall application. The conversation illuminates several aspects of the system that are currently under review for improvement, including CPU vs GPU backend usage, the generation of irrelevant answers, and buffer limit issues.

**Backend Usage**

The logs indicated that the GPU backend was used for inference. However, it's worth noting that despite this message, the system actually utilized the CPU for inference. This discrepancy is being looked into for clarification and potential optimization.

**Irrelevant Output**

In the conversation, the user's question about writing a small Python script dealing with complex numbers received an irrelevant output. The system generated an exception message and a code snippet that doesn't align with the query. This issue is under review to improve the system's understanding of context and relevance in responses.

**Buffer Limit Exceeded**

During the conversation, the system also encountered a buffer limit issue, triggering an AssertionError. This is a known, logged issue that we are working on. Future updates may include strategy classes to better manage buffer limits and prevent such exceptions.

**Llama Integration**

To start the Llama module, run the following commands:

```bash
pip install -r requirements.txt
cd app/models/llama/weights/
bash pull_llama.sh
cd ../../../..
python3 app/models/llama/llama.py
```


```bash
(.venv) james@Sid-Desktop:/mnt/e/workspace/townhall$ /mnt/e/workspace/townhall/.venv/bin/python /mnt/e/workspace/townhall/app/models/llama/llama.py
using GPU backend
using LLaMA-7B model
ram used: 13.48 GB, freqs_cis                                         : 100%|█████████████████████████████████████████████████████████████████████████████████████████████████| 292/292 [00:00<00:00, 3758.05it/s]
loaded weights in 80.91 ms, 13.48 GB loaded at 166.56 GB/s
Preparing KV cache for chatbot with personality Stacy...
1 warning generated.
/mnt/e/workspace/townhall/.venv/lib/python3.10/site-packages/pyopencl/cache.py:417: CompilerWarning: Non-empty compiler output encountered. Set the environment variable PYOPENCL_COMPILER_OUTPUT=1 to see more.
  prg.build(options_bytes, [devices[i] for i in to_be_built_indices])
1 warning generated.
1 warning generated.
1 warning generated.
1 warning generated.
1 warning generated.
1 warning generated.
1 warning generated.
1 warning generated.
171400.72 ms
Consider that the following is conversation between an AI assistant named Stacy and User
You are Stacy!
You have been a rapper your whole life who struggled with bipolar disorder. You called yourself lil stacy.
You love to answer questions and you are very good at it. Sometimes you answer in rap form.
You are verbose, honest, and accurate when you answer questions, but sometimes your mental illness manifests.
After you are done speaking, output [EOS]. You are not the User.

<CHAT LOG>

User: What is your name?
Stacy: Hi! My name is Stacy. I'm a rapper with bipolar disorder. [EOS]

User: french revolution was what year?
Stacy: The French Revolution started in 1789, and lasted 10 years until 1799. [EOS]

User: What is bigger, the moon or the sun?
Stacy: The sun is bigger than the moon, except when Mercury is in retrograde. [EOS]

User: Write a small python script that deals with complex numbers.
1 warning generated.
1 warning generated.
1 warning generated.
1 warning generated.
1 warning generated.
1 warning generated.
1 warning generated.
1 warning generated.
1 warning generated.
St1 warning generated.
1 warning generated.
1 warning generated.
1 warning generated.
1 warning generated.
1 warning generated.
1 warning generated.
1 warning generated.
acy: I'll try. !<EOS>
\end{code}

\begin{blockquote}

\begin{code}
import sys
import re
from functools import partial
from collections import defaultdict
from typing import List, Dict

#
# The goal is to create some kind of knowledge base
# that can be queried or queried with a "fuzzy" search
#

class Answer:
    def __init__(self, name, description, answers):
        self.name = name
        self.description = description
        self.answers = answers

    def __str__(self):
        return f'{self.name}: {self.description}'

    def __repr__(self):
        return f'{self.__str__()}'.encode('utf-8')

    def __eq__(self, other):
        return self.__str__() == other.__str__()

    def __hash__(self):
        return hash(self.__str__())

    def as_dict(self):
        return {
            'name': self.name,
            'description': self.description,
            'answers': self.answers,
        }

class Question:
    def __init__(self, question, answers):
        self.question = question
        self.answers = answers

    def __str__(self):
        return f'{self.question}: {self.answers}'

    def __repr__(self):
        return f'{self.__str__()}'.encode('utf-8')

    def __eq__(self, other):
        return self.__str__() == other.__str__()

    def __hash__(self):
        return hash(self.__str__())

    def as_dict(self):
        return {
            'question': self.question,
            'answers': self.answers,
        }

class QuestionList:
    def __init__(self):
        self.questions = []

    def __len__(self):
        return len(self.questions)

    def __repr__(self):
        return self.__str__()

    def __str__(self):
        return ' '.join(f'{question}'.encode('utf-8') for question in self.questions)

    def __eq__(self, other):
        return self.__str__() == other.__str__()

    def __hash__(self):
        return hash(self.__str__())

    def as_dict(self):
        return {
            'questions': self.questions,
        }

class User:
    def __init__(self, name, description, questions):
        self.name = name
        self.description = description
        self.questions = questions

    def __str__(self):
        return f'{self.name}: {self.description}'

    def __repr__(self):
        return f'{self.__str__()}'.encode('utf-8')

    def __eq__(self, other):
        return self.__str__() == other.__str__()

    def __hash__(self):
        return hash(self.__str__())

    def as_dict(self):
        return {
            'name': self.name,
            'Traceback (most recent call last):
  File "/mnt/e/workspace/townhall/app/models/llama/llama.py", line 260, in <module>
    probs = llama.model(
  File "/mnt/e/workspace/townhall/app/models/llama/transformer.py", line 83, in __call__
    pos = Variable("pos", 1, 1024).bind(start_pos)
  File "/mnt/e/workspace/townhall/ext/tinygrad/tinygrad/shape/symbolic.py", line 155, in bind
    assert self.val is None and self.min<=val<=self.max, f"cannot bind {val} to {self}"
AssertionError: cannot bind 1025 to <pos[1-1024]>
```
