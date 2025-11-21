import time
from openai import OpenAI
from src.basereal import BaseReal
from src.log import logger
from src.config import get_llm_api_key, get_llm_base_url


api_key = get_llm_api_key()
base_url = get_llm_base_url()
show_api_key = api_key[:10] + "..."
client = OpenAI(
    api_key=api_key,
    base_url=base_url,
)
logger.debug(f"llm api_key: {show_api_key}, llm base_url: {base_url}")


def llm_response(message, nerfreal: BaseReal):
    start = time.perf_counter()
    completion = client.chat.completions.create(
        model="qwen-plus",
        messages=[{'role': 'system', 'content': 'You are a helpful assistant.简单回答用户问题，尽量简洁。'},
                  {'role': 'user', 'content': message}],
        stream=True,
        stream_options={"include_usage": True}
    )
    result = ""
    first = True
    for chunk in completion:
        if len(chunk.choices) > 0:
            if first:
                end = time.perf_counter()
                logger.debug(f"llm Time to first chunk: {end - start}s")
                first = False
            msg = chunk.choices[0].delta.content
            lastpos = 0
            for i, char in enumerate(msg):
                if char in ",.!;:，。！？：；":
                    result = result + msg[lastpos:i + 1]
                    lastpos = i + 1
                    if len(result) > 10:
                        logger.info(result)
                        nerfreal.put_msg_txt(result)
                        result = ""
            result = result + msg[lastpos:]
    end = time.perf_counter()
    logger.debug(f"llm Time to last chunk: {end - start}s")
    nerfreal.put_msg_txt(result)
