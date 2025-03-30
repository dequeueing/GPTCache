from typing import *

def extract_user_question(data: Dict[str, Any], **_: Dict[str, Any]) -> Any:
    input_string = data.get("prompt")
    
    # Define the markers
    start_marker = "<|end_header_id|>"
    end_marker = "<|eot_id|>"
    
    # Find the positions of the markers
    start_pos = input_string.find(start_marker, input_string.find("user")) + len(start_marker)
    end_pos = input_string.find(end_marker, start_pos)
    
    # Extract the substring between the markers
    if start_pos != -1 and end_pos != -1:
        question = input_string[start_pos:end_pos].strip()
        return question
    else:
        return input_string
    
def from_list(messages: List[Any]) -> Any:
    """No change after evaluation.

    :param messages: A list of candidate outputs.
    :type messages: List[Any]

    Example:
        .. code-block:: python

            from gptcache.processor.post import nop

            messages = ["message 1", "message 2", "message 3"]
            answer = nop(messages)
            assert answer = messages
    """
    return messages[0]


def after_template(message, tokenizer):
    prompt = [
        {"role": "user", "content": message},
    ]
    return tokenizer.apply_chat_template(
        prompt, tokenize=False, add_generation_prompt=True
    )
