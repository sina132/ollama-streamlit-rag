import time
import ollama
import os
def get_time() -> str:
    """Return the human-readable current time."""
    return time.ctime()

def check_dir():
    '''
    check currenct directory files
    '''
    return os.listdir("./files")

tools = {
    "get_time": get_time,
    "check_dir":check_dir
}

res = ollama.chat(
    model="llama3.1:8b",
    messages=[{'role': 'user', 'content': 'please list the files in the current directory'}],
    tools=list(tools.values()),
)
for function in res.message.tool_calls:
    name = function.function.name
    args = function.function.arguments
    if name in tools:
        res = tools[name](**args)
        print(res);
        