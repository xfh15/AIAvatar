import time
from collections import deque
from openai import OpenAI
from src.basereal import BaseReal
from src.log import logger
from src.config import get_llm_api_key, get_llm_base_url, get_llm_model_name


api_key = get_llm_api_key()
base_url = get_llm_base_url()
model_name = get_llm_model_name()
show_api_key = api_key[:10] + "..."
client = OpenAI(
    api_key=api_key,
    base_url=base_url,
)
logger.debug(f"llm api_key: {show_api_key}, llm base_url: {base_url}, llm model: {model_name}")

# 全局对话历史管理
# key: session_id, value: deque of messages (最多保留10轮，即20条消息)
_conversation_history = {}
MAX_HISTORY_ROUNDS = 10  # 最多保留10轮对话
MAX_HISTORY_MESSAGES = MAX_HISTORY_ROUNDS * 2  # 每轮2条消息（user + assistant）

# 系统提示词
SYSTEM_PROMPT = 'You are a helpful assistant.简单回答用户问题，尽量简洁。回答为text格式，不要markdown格式的。'


def get_conversation_history(session_id):
    """获取指定session的对话历史"""
    if session_id not in _conversation_history:
        _conversation_history[session_id] = deque(maxlen=MAX_HISTORY_MESSAGES)
    return _conversation_history[session_id]


def add_to_history(session_id, role, content):
    """添加消息到对话历史
    
    Args:
        session_id: 会话ID
        role: 'user' 或 'assistant'
        content: 消息内容
    """
    history = get_conversation_history(session_id)
    history.append({'role': role, 'content': content})


def clear_conversation_history(session_id=None):
    """清除对话历史
    
    Args:
        session_id: 指定session_id清除特定会话，None则清除所有
    """
    global _conversation_history
    if session_id is None:
        _conversation_history.clear()
        logger.info("All conversation history cleared")
    elif session_id in _conversation_history:
        del _conversation_history[session_id]
        logger.info(f"Session {session_id} conversation history cleared")


def build_messages(session_id, current_message):
    """构建完整的消息列表（系统提示词 + 历史 + 当前消息）
    
    Args:
        session_id: 会话ID
        current_message: 当前用户消息
    
    Returns:
        完整的消息列表
    """
    messages = [{'role': 'system', 'content': SYSTEM_PROMPT}]
    
    # 添加历史消息
    history = get_conversation_history(session_id)
    messages.extend(list(history))
    
    # 添加当前消息
    messages.append({'role': 'user', 'content': current_message})
    
    return messages


def llm_response(message, nerfreal: BaseReal, session_id=None):
    """生成LLM响应（支持多轮对话）
    
    Args:
        message: 用户输入消息
        nerfreal: BaseReal实例
        session_id: 会话ID，用于区分不同用户/会话的对话历史
    """
    # 如果没有提供session_id，使用nerfreal的sessionid
    if session_id is None:
        session_id = getattr(nerfreal, 'sessionid', 'default')
    
    logger.debug(f"Session {session_id}: User message: {message}")
    
    # 构建包含历史的消息列表
    messages = build_messages(session_id, message)
    
    start = time.perf_counter()
    completion = client.chat.completions.create(
        model=model_name,  # 使用配置文件中的模型名称
        messages=messages,
        stream=True,
        stream_options={"include_usage": True}
    )
    
    result = ""
    assistant_response = ""  # 完整的助手响应
    first = True
    first_tts = True  # 是否是第一次发送TTS
    
    # 只在大标点处分割（句号、问号、感叹号、分号）
    major_punctuation = "。！？；.!?;"
    
    for chunk in completion:
        if len(chunk.choices) > 0:
            if first:
                end = time.perf_counter()
                logger.debug(f"llm Time to first chunk: {end - start}s")
                first = False
            msg = chunk.choices[0].delta.content
            if msg:
                assistant_response += msg  # 累积完整响应
                lastpos = 0
                for i, char in enumerate(msg):
                    # 只在大标点处分割
                    if char in major_punctuation:
                        result = result + msg[lastpos:i + 1]
                        lastpos = i + 1
                        # 立即发送给TTS，无需等待长度限制
                        if len(result.strip()) > 0:
                            if first_tts:
                                first_tts = False
                            nerfreal.put_msg_txt(result)
                            result = ""
                result = result + msg[lastpos:]
    
    end = time.perf_counter()
    # 处理最后剩余的文本（如果没有以大标点结尾）
    if result.strip():
        nerfreal.put_msg_txt(result)
    
    # 保存到对话历史
    add_to_history(session_id, 'user', message)
    add_to_history(session_id, 'assistant', assistant_response)
    