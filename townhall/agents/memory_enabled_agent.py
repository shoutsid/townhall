"""
Memory Enabled Agent (MEA)
"""

import os
from pathlib import Path
import sys
from autogen import AssistantAgent, Agent
from typing import Dict, Optional, Union

sys.path.append(str(Path(__file__).parent.parent.parent.absolute()))
# pylint: disable=wrong-import-position
from townhall.agents.user_agent import UserAgent
from settings import LLM_CONFIG as gpt_config
# pylint: enable=wrong-import-position

# Default directory for storing agent memories
MEMORY_DIRECTORY = "agent_workplace/memories"

# Memory Enabled Agent (MEA)
class MemoryEnabledAgent(AssistantAgent):
    """
    A class representing an agent that has short-term and long-term memory capabilities.
    This agent is designed to ensure that discussions benefit from short-term memory (STM).
    It provides a method to check memory using `lookup_from_long_term_memory` and pass in a hint describing what you are trying to remember.
    """

    # Portion of MEA system prompt - appended after DEFAULT_SYSTEM_MESSAGE. Responsible for ensuring discussion benefits from short term memory (STM)
    DEFAULT_MEM_AGENT_MESSAGE = """

    Be sure to modulate your discussion by what you remember about {sender}.
    If something is missing that you should know, try checking your memory using lookup_from_long_term_memory and pass in a hint describing what you are trying to remember.
    If you are asked what can you remember, it would be good to call lookup_from_long_term_memory.
    """

    # Other portion of MEA system prompt - this should be modified for the task at hand for the MEA
    DEFAULT_SYSTEM_MESSAGE = """   """

    #--------- Dynamic Memory Settings -----------
    # Consult documentation before adjusting from defaults - Bad values can cause infinite looping/AI calls/$$
    # Max number of short term memories before initiating compression to long
    DEFAULT_SHORT_TERM_MEMORY_LIMIT = 10

    # Proportion to cut short term memory off (0.9 drops 9 out of 10 memories after exceeding STM limit, 0.1 drops 1 out of 10 memories after exceeding STM limit)
    DEFAULT_COMPRESSION_RATIO_STM = 0.8

    # Max convo length before the end starts falling off, in number of messages. User and AI both count, so minimum is 2.
    DEFAULT_MAX_CONVO_LENGTH = 10

    # Proportion to cut chat off (0.9 drops 9 out of 10 chats after exceeding limit, 0.1 drops 1 out of 10 chats after exceeding limit)
    DEFAULT_COMPRESSION_RATIO_CHAT = 0.8

    def __init__(self, name, gpt_config):
        self.gpt_config = gpt_config
        self.llm_config = gpt_config['config_list']
        self.config_list = gpt_config['config_list']
        self.llm_model = gpt_config['config_list'][0]['model']

        # pylint: disable=line-too-long
        super().__init__(
            name = name,
            llm_config = {
            "temperature": 0,
            "request_timeout": 600,
            "seed": 42,
            "model": self.llm_model,
            "config_list": self.config_list,
            "functions": [
                    {
                        "name": "lookup_from_long_term_memory",
                        "description": "Retrieves information from the long term memory; Do not use this function without first referring to Your current short term memory. Use this function to recall something outside the scope of your context. These are your older memories.",
                        "parameters": {
                            "type": "object",
                            "properties": {
                                "hint": {
                                    "type": "string",
                                    "description": "A hint or description of what information is being attempted to be retrieved.",
                                },
                            },
                            "required": ["hint"],
                        },
                    },

                ],
            },
            system_message = self.DEFAULT_SYSTEM_MESSAGE + self.DEFAULT_MEM_AGENT_MESSAGE
        )
        # pylint: enable=line-too-long
        self.memories_path = os.path.join(MEMORY_DIRECTORY, self.name)
        self.short_term_memory_path = os.path.join(self.memories_path,"short_term_memory.txt")
        self.long_term_memory_path = os.path.join(self.memories_path,"long_term_memory.txt")
        self.memories = self.initialize_memories()
        self.memory_manager = self.initialize_memory_manager()
        self.functions_for_map = [self.lookup_from_long_term_memory]
        self.sender_agent = None

    def receive(
        self,
        message: Union[Dict, str],
        sender: Agent,
        request_reply: Optional[bool] = None,
        silent: Optional[bool] = False,
    ):
        # pylint: disable=line-too-long
        """Receive a message from another agent.

        Once a message is received, this function sends a reply to the sender or stop.
        The reply can be generated automatically or entered manually by a human.

        Args:
            message (dict or str): message from the sender. If the type is dict, it may contain the following reserved fields (either content or function_call need to be provided).
                1. "content": content of the message, can be None.
                2. "function_call": a dictionary containing the function name and arguments.
                3. "role": role of the message, can be "assistant", "user", "function".
                    This field is only needed to distinguish between "function" or "assistant"/"user".
                4. "name": In most cases, this field is not needed. When the role is "function", this field is needed to indicate the function name.
                5. "context" (dict): the context of the message, which will be passed to
                    [Completion.create](../oai/Completion#create).
            sender: sender of an Agent instance.
            request_reply (bool or None): whether a reply is requested from the sender.
                If None, the value is determined by `self.reply_at_receive[sender]`.
            silent (bool or None): (Experimental) whether to print the message received.

        Raises:
            ValueError: if the message can't be converted into a valid ChatCompletion message.
        """
        # pylint: enable=line-too-long

        # Read short term memory file
        self.memories = self.read_short_term_memory()

        # Default AutoGen function
        self._process_received_message(message, sender, silent)

        # If there is more than the initial message
        if len(self.chat_messages[sender]) > 1:
            # Construct the STM message to be placed at top of context
            m0 = {}
            m0['content'] = f"Things you remember about {self.sender_agent.name}: {self.memories}|"
            m0['role'] = 'assistant'

            # Overwrite top of context with STM
            self.chat_messages[sender][0] = m0

        # If there is only the initial message - only at chat initialization
        else:
            # set sender_agent to sender, and format MEA system prompt to senders name
            self.sender_agent = sender # safe to set here
            self.system_message.format(sender=self.sender_agent.name)


        # Debugging callouts for monitoring chat progression/dynamics
        print("DEBUG: NumChatMessages: "  +str(len(self.chat_messages[sender]))
            + " vs Limit:" + str(self.DEFAULT_MAX_CONVO_LENGTH))
        print("DEBUG: ChatMessages:")
        print(self.chat_messages[sender])
        print("END DEBUG")

        # If max length is hit, use compression ratio to trim window and
        #  then pass history to memory manager
        # Remove the oldest message, but not the first! First message is dynamic short term memory
        if self.chat_too_long():
            # index 0 is short term memory, index 1 is start of conversation, and where to do FILO
            # use compression ratio to determine trim number
            trim_num = int(len(self.chat_messages[sender])*self.DEFAULT_COMPRESSION_RATIO_CHAT)
            lost_messages = []

            # pop corresponding messages out of chat history
            for i in range(trim_num-1):
                lost_messages.append(self.chat_messages[sender].pop(1))

            # send messages to memory manager to process
            self.memory_manager.process_chat_section(lost_messages)

            # Debugging callouts for monitoring chat message trimming
            print(f"DEBUG: Messages trimmed from chat:\n{lost_messages}")

        # Default AutoGen Logic
        if request_reply is False or request_reply is None \
            and self.reply_at_receive[sender] is False:
            return
        reply = self.generate_reply(messages=self.chat_messages[sender], sender=sender)
        if reply is not None:
            self.send(reply, sender, silent=silent)

    def initialize_memories(self):
        """
        Initializes the memory structure and MEA memories.

        If a memory directory does not exist, it creates one.
        If the specific agent's memory does not exist,
        it initializes the memory folder and files.
        Otherwise, it reads the short-term memory and returns it.

        Returns: None if the agent's memory does not exist, otherwise the short-term memory.
        """
        # Initilize memory structure and MEA memories.
        # Does a memory directory exist? If not, make one
        if not os.path.exists(MEMORY_DIRECTORY):
            os.makedirs(MEMORY_DIRECTORY)

        # Does this specific agents memory exist?
        if os.path.exists(self.memories_path):
            return self.read_short_term_memory()

        # If not, initialize the memory folder and files
        else:
            os.makedirs(self.memories_path)
            with open(self.long_term_memory_path, 'w', encoding='utf-8') as _f:
                pass

            with open(self.short_term_memory_path, 'w', encoding='utf-8') as _f:
                pass

            return None

    def initialize_memory_manager(self):
        """
        Initializes the memory manager agent for the Memory Enabled Agent (MEA).
        Passes in the MEA object as the parent agent.

        Returns:
        MemoryEnabledAgentManager: The initialized memory manager agent.
        """
        return MemoryEnabledAgentManager(parent_agent= self)

    def chat_too_long(self):
        """
        Check if chat is exceeding the default maximum conversation length.

        Returns:
            bool:
                True if the chat is exceeding the default maximum conversation length,
                False otherwise.
        """
        all_chats = self.chat_messages[self.sender_agent]
        filtered_chats = [c for c in all_chats if 'function_call' not in c]
        self.chat_messages[self.sender_agent] = filtered_chats
        return bool(len(self.chat_messages[self.sender_agent]) > self.DEFAULT_MAX_CONVO_LENGTH)

    def read_short_term_memory(self, list_mode=False):
        """
        Read and return short term memories, either as string or list.

        Args:
            list_mode (bool):
                If True, the memories will be returned as a list instead of a joined string.

        Returns:
            str or list: The short term memories.
        """
        if not list_mode:
            with open(self.short_term_memory_path, 'r', encoding='utf-8') as f:
                memories = '|'.join(f.readlines())
            return memories

        # List_mode is used if the memories should be returned
        #  as a list instead of a joined string - used by memory manager
        with open(self.short_term_memory_path, 'r', encoding='utf-8') as f:
            return f.readlines()[0].split('|')

    def append_to_short_term_memory(self, memories):
        """
        Append new short term memories to the short term memory file.

        Args:
            memories (list): A list of memories to append to the short term memory file.

        Returns:
            bool: True if the memories were successfully appended to the file, False otherwise.
        """
        # Append new short term memories to STM file.
        with open(self.short_term_memory_path, 'a', encoding='utf-8') as f:
            for memory in memories:
                if memory is not None:
                    f.write(f"{memory}|")

        # Check if new additions cause STM to exceed limit
        if self.short_term_memory_full():
            # Debugging messages to monitor memory compression
            print("DEBUG")
            print("ATTEMPTING MEMORY COMPRESSION")
            print("END DEBUG")

            # NOTE: Can cause infinite looping/AI calls/$$
            s_to_l_response = self.short_term_to_long_term()
            print(s_to_l_response)
            return True
        return True

    def short_term_memory_full(self):
        """
        Checks if the short term memory has reached its limit.

        Returns:
            bool: True if the short term memory has reached its limit, False otherwise.
        """
        with open(self.short_term_memory_path,'r') as f:
            num_memories = len(f.readlines()[0].split('|'))

        if num_memories > self.DEFAULT_SHORT_TERM_MEMORY_LIMIT:
            return True
        else:
            return False

    def short_term_to_long_term(self):
        """
        Compresses the short-term memory of the agent by trimming off some
        of the oldest memories (FILO) and requests the
        memory manager to store the compressed memory in long-term memory.
        The memory manager rewrites the memory as normal,
        but without the trimmed off memories.

        Returns:
            The compressed long-term memory.
        """
        return self.memory_manager.short_to_long()

    def lookup_from_long_term_memory(self, hint):
        """
        Attempt to retrieve information from long-term memory as it relates to a hint.

        Parameters:
        hint (str): A hint to use when searching for information in long-term memory.

        Returns:
        The information retrieved from long-term memory, if any.
        """
        return self.memory_manager.lookup_from_long(hint)

    def rewrite_short_term_memory(self, memories):
        """
        Rewrites the short term memory file with the given list of memories.

        Args:
            memories (list): A list of memories to write to the short term memory file.

        Returns:
            None
        """
        with open(self.short_term_memory_path, 'w', encoding='utf-8') as f:
            for memory in memories:
                f.write(f"{memory}|")

    def get_function_map(self):
        """
        Returns a dictionary mapping function names to their corresponding functions.

        Returns:
            dict: A dictionary mapping function names (str) to their corresponding functions.
        """
        f_map = {}
        f_list = self.llm_config["functions"]
        for i, f in enumerate(f_list):
            f_map[f['name']] = self.functions_for_map[i]

        return f_map




class MemoryEnabledAgentManager(AssistantAgent):
    # pylint: disable=line-too-long
    """
    A class that represents an AI assistant acting as a memory manager.
    It can summarize conversation sections, incorporate information into long-term memory,
    and retrieve information from long-term memory.

    Attributes:
    - parent_agent: the parent agent object
    - gpt_config: the GPT configuration object
    - llm_config: the LLM configuration object
    - config_list: the configuration list
    - llm_model: the LLM model
    """

    DEFAULT_MEM_MANAGER_MESSAGE = """
    You are a helpful AI assistant acting as a memory manager. There are three situations where you will be asked to perform:

    1. Summarize conversation section
    You will be shown a section of conversation as it leaves the context window of another assistant. You are to extract key points, facts, or memories, from the section of conversation which will then be added to the memory of the other assistant. It is critical that you are detail oriented so you do not miss anything.

    When you are shown a section of conversation, make a function call to append_to_short_term_memory and pass in a list of facts/statements/anything that accuractly and succinctly covers all details.


    2. Incorporate information into long-term memory
    You will be shown the existing long term memory, and a list of new memories which need to be incorporated. The memory is formatted as a series of statements seperated by the '|' character.
    For example, the existing memory might look like this:
        "{User's name} is allergic to seafood| {User's name} likes dogs| {User's name} is from Canada"
    In this example, if you were requested to write a message to memory that was "{User's name} likes cats", when you go to rewrite the memory, it should look like this:
        "{User's name} is allergic to seafood| {User's name} likes dogs, cats| {User's name} is from Canada"
    Sometimes it might be best to remove content from the memory. For example, if the existing memory looked like this:
        "{User's name} is allergic to seafood| {User's name} likes dogs, cats| {User's name} is from Canada"
    And you were asked to write a message to memory that was "User does not like dogs", when you go to rewrite the memory, it should look like this:
        "{User's name} is allergic to seafood| {User's name} likes cats| {User's name} is from Canada"
    The changes to the rewritten memory can span multiple statements, if appropriate. The point is to keep the entire memory as accurate and representative as possible.

    3. Retrieve information from long-term memory
    You will be shown the existing long term memory, and be asked by the agent "What do I know about {thing}?". The agent is trying to find any information in its memory about {thing}. You should respond with a simple statement that answers the question.

    Do not participate in any form of conversation.

    If you do not know something, say
        "I don't know anything about {}

        TERMINATE"

    After you have finished your task, respond with "TERMINATE"

    """
    # pylint: enable=line-too-long

    def __init__(self, parent_agent):
        # Grab parent agent object and parent agent llm config info.
        self.parent_agent = parent_agent
        self.gpt_config = parent_agent.gpt_config
        self.llm_config = self.gpt_config['config_list']
        self.config_list = self.gpt_config['config_list']
        self.llm_model = self.gpt_config['config_list'][0]['model']

        # Regular __init__ for AssistentAgent
        # pylint: disable=line-too-long
        super().__init__(
            name= parent_agent.name + "_MemoryManager",
            llm_config={
                "temperature": 0,
                "request_timeout": 600,
                "seed": 42,
                "model": self.llm_model,
                "config_list": self.config_list,
                "functions": [
                    {
                        "name": "rewrite_memory",
                        "description": "Rewrites the entire memory to incorporate new information in a smart, condensed, entity focused way.",
                        "parameters": {
                            "type": "object",
                            "properties": {
                                "memories": {
                                    "type": "string",
                                    "description": "the rewritten memory. Contains all the information from existing memory as well as the new information.",
                                },
                            },
                            "required": ["memories"],
                        },
                    },
                    {
                        "name": "append_to_short_term_memory",
                        "description": "Writes an array of items to memory.",
                        "parameters": {
                            "type": "object",
                            "properties": {
                                "memories": {
                                    "type": "array",
                                    "items": {
                                        "type": "string",
                                        "description": "thing or things to record and remember. Make sure each string is context complete, such that an outsider would understand it."
                                    },
                                },
                            },
                            "required": ["memories"]
                        }
                    },


                ],
            },
            system_message = self.DEFAULT_MEM_MANAGER_MESSAGE,
        )
        # pylint: enable=line-too-long

        # These are dummy user_agents to allow MMA code execution.
        # There needs to be two as conversation histories were
        # cross-contaminating on consequetive function calls.
        self.function_agent_ltm = UserAgent(
            name="user_proxy_for_LTM",
            is_termination_msg= self.is_mem_termination_msg,
            human_input_mode="NEVER",
            max_consecutive_auto_reply=1,
            code_execution_config={"work_dir": "agent_workplace"},
            function_map={"rewrite_memory": self.rewrite_memory}
            )
        self.function_agent_stm = UserAgent(
            name="user_proxy_for_STM",
            is_termination_msg= self.is_mem_termination_msg,
            human_input_mode="NEVER",
            max_consecutive_auto_reply=1,
            code_execution_config={"work_dir": "agent_workplace"},
            function_map={
                "append_to_short_term_memory": self.parent_agent.append_to_short_term_memory}
            )

    def is_mem_termination_msg(self, msg):
        """
        Check if the given message is a memory termination message.

        A memory termination message is a message that ends with the string "TERMINATE".

        Args:
            msg (dict): The message to check.

        Returns:
            bool: True if the message is a memory termination message, False otherwise.
        """
        if msg.get("content") is not None:
            if msg.get("content", "").rstrip().endswith("TERMINATE"):
                return True
        return False


    def process_chat_section(self, lost_messages):
        """
        Processes a lost conversation section by summarizing it and prompting the
        user to append the key points to short-term memory.

        Args:
            lost_messages (str): The lost conversation section to be summarized.

        Returns:
            None
        """
        message = f"""
        Conversation Section to Summarize:
        {lost_messages}

        Please make a function call to append_to_short_term_memory and pass in the key points you can extract from the above conversation section. Do not use 'User' or 'Assistant' - replace 'User' with {self.parent_agent.sender_agent.name}, and replace 'Assistant' with 'I'.
        """
        self.function_agent_stm.initiate_chat(self, message=message)

    def read_long_term_memory(self):
        """
        Reads the long-term memory of the agent from the file specified by `long_term_memory_path`.

        Returns:
            A list of strings representing the contents of the long-term memory file,
            split by the '|' character. If the file is empty, an empty list is returned.
        """
        with open(self.parent_agent.long_term_memory_path, 'r', encoding='utf-8') as f:
            mems = f.readlines()

        if len(mems) == 0:
            return []

        return mems[0].split('|')

    def short_to_long(self):
        """
        Moves short-term memories to long-term memory based on a compression ratio.
        Trim off tail of short term memory to incorporate into long (FILO)

        Returns:
            A list of memories that were moved to long-term memory.
        """
        # method code here
        # Get the memories in list mode
        mems = self.parent_agent.read_short_term_memory(list_mode = True)
        # Determine trim point using STM Compression Ratio
        trim_num = int(len(mems)*self.parent_agent.DEFAULT_COMPRESSION_RATIO_STM)
        # Split between mems to leave (stay in STM) and mems to store (into LTM)
        mems_to_leave = mems[trim_num:]
        mems_to_store = mems[:trim_num]
        # Rewrite STM with shortened memory list
        self.parent_agent.rewrite_short_term_memory(mems_to_leave)

        # Call incorporate_memories, return result.
        return self.incorporate_memories(mems_to_store)

    def incorporate_memories(self, memories):
        # pylint: disable=line-too-long
        """
        Incorporates new memories into the agent's long-term memory.
        Present MMA with full LTM, trimmed of STM, and have it redo LTM to incorporate STM

        Args:
            memories (list): A list of strings representing the new memories to incorporate.

        Returns:l
            bool: True if the operation was successful, False otherwise.
        The method prompts the user to call the `rewrite_memory` function passing in the reconfigured long-term memory
        which incorporates the old with the new. It is recommended to modify memories in place to capture new information
        instead of always making the memory longer; only make it longer if necessary, but otherwise do your best to condense,
        reorganize, and rewrite. The goal is for the Long Term Memory you are writing to be as entity dense as possible.
        """
        # pylint: enable=line-too-ong

        message = f"""
        Full Long Term Memory:
        {self.read_long_term_memory()}

        New memory or memories to incorporate:
        {'|'.join(memories)}

        Please make a function call to rewrite_memory and pass in the reconfigured long term memory which incorporates the old with the new. It is better to modify memories in place to capture new information instead of always making the memory longer; only make it longer if necessary, but otherwise do your best to condense, reorganize, and rewrite. The goal is for the Long Term Memory you are writing to be as entity dense as possible.
        """
        self.function_agent_ltm.initiate_chat(
            self,
            message=message
        )
        return True

    def rewrite_memory(self, memories):
        """
        Rewrite the entire long term memory with the given memories.

        Args:
            memories (str):
                A string representing the new long term memory.

        Returns:
            bool:
                True if the memory was successfully rewritten, False otherwise.
        """
        memory_list = memories.split('|')
        with open(self.parent_agent.long_term_memory_path, 'w', encoding='utf-8') as f:
            for memory in memory_list:
                f.write(f"{memory}|")

        return True

    def lookup_from_long(self, _hint):
        """
        Looks up information related to the given hint in the agent's long term memory
        and initiates a chat with the function agent to ask for more information.
        The response is returned to the conversing agent.
        Called by MEA - request for information from LTM relating to hint.

        Args:
            hint (str):
                A string representing the information to look up in the agent's long term memory.

        Returns:
            str:
                The response from the function agent,
                which contains information related to the given hint.
        """
        self.function_agent_ltm.initiate_chat(
            self,
            message=f"Full Long Term Memory:\n{self.read_long_term_memory()}\n\n" \
                "What do I know about: {hint}?\n\n " \
                "Respond in chat - Do not make a function call. " \
                "Replace 'you' with {self.parent_agent.sender_agent.name}. End with TERMINATE."
        )
        # Send back the response to the conversing agent.
        # Due to current flow and manual exiting, '-3' is magic number that gets
        #  original MMA response to question.
        return self.chat_messages[self.function_agent_ltm][-1]


if __name__ == "__main__":
    # Make the MemoryEnabledAgent
    mem_agent = MemoryEnabledAgent("Assistant", gpt_config)

    # Make a user proxy - retrieve mem_agent function map.
    user_proxy = UserAgent(
        name="User",
        human_input_mode="ALWAYS",
        max_consecutive_auto_reply=10,
        code_execution_config={"work_dir": "agent_workdir"},
        function_map=mem_agent.get_function_map(),
    )

    # Chat with the Agent
    user_proxy.initiate_chat(
        mem_agent,
        message=f"(User:{user_proxy.name} Connected)"
    )
