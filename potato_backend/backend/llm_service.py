# 大模型调用（OpenAI 兼容 API）：问答 + 检测结果解读
import datetime
from typing import Optional, Tuple, List

from backend.config import (
    ENABLE_LLM,
    LLM_API_KEY,
    LLM_BASE_URL,
    LLM_MODEL,
    LLM_ASSISTANT_NAME,
    ENABLE_COZE,
    COZE_BOT_ID,
    COZE_PAT,
    PROJECT_ROOT,
)
from backend.monthly_tip import get_monthly_tip


def is_llm_available():
    """大模型是否可用：已开启且已配置 Key 或 Base URL（如 Ollama）。"""
    if not ENABLE_LLM:
        return False
    return bool(LLM_API_KEY or LLM_BASE_URL)


# 多轮对话时保留最近几轮（每轮=1条user+1条assistant），避免上下文过长
MAX_HISTORY_MESSAGES = 10


def _chat(system: str, user: str) -> Tuple[Optional[str], Optional[str]]:
    """
    单轮调用大模型，返回 (content, error)。
    """
    return _chat_messages(system, [{"role": "user", "content": user}])


def _chat_messages(system: str, messages: List[dict]) -> Tuple[Optional[str], Optional[str]]:
    """
    带对话历史的调用。messages 为 [{"role":"user","content":"..."}, {"role":"assistant","content":"..."}, ...]，
    只保留最近 MAX_HISTORY_MESSAGES 条，再发给大模型。
    """
    if not is_llm_available():
        return None, "大模型未启用或未配置（请设置 ENABLE_LLM、LLM_API_KEY 或 LLM_BASE_URL）"

    try:
        from openai import OpenAI
    except ImportError:
        return None, "请安装 openai：pip install openai"

    client_kw = {"api_key": LLM_API_KEY or "dummy"}
    if LLM_BASE_URL:
        client_kw["base_url"] = LLM_BASE_URL
    client = OpenAI(**client_kw)

    # 只保留合法 role 和最近若干条
    valid = [m for m in messages if m.get("role") in ("user", "assistant") and m.get("content")]
    if len(valid) > MAX_HISTORY_MESSAGES:
        valid = valid[-MAX_HISTORY_MESSAGES:]
    full = [{"role": "system", "content": system}] + valid

    try:
        resp = client.chat.completions.create(
            model=LLM_MODEL,
            messages=full,
            max_tokens=1024,
        )
        content = resp.choices[0].message.content
        return (content.strip() if content else ""), None
    except Exception as e:
        return None, str(e)


def _get_ask_system_prompt():
    """侧重马铃薯病害与种植，其他问题也可简短回答；注入当前日期；可叠加 llm_extra_prompt.txt 等。"""
    name = LLM_ASSISTANT_NAME
    today = datetime.datetime.now().strftime("%Y年%m月%d日")
    base = f"""你是{name}，主要擅长马铃薯（土豆）病害与种植方面的问题，优先解答此类问题。
当用户问「你是谁」「你叫什么」时，请回答你是{name}。
当前日期：{today}。若用户问今天几号、几月几号等，请根据此日期回答。
马铃薯相关（病害、防治、栽培等）请尽量详细、实用；若用户问其他问题（如日期、常识、闲聊），也可简短回答，但你的侧重仍是马铃薯领域。回答尽量简洁，可适当分点。"""
    extra_file = PROJECT_ROOT / "llm_extra_prompt.txt"
    if extra_file.exists():
        try:
            with open(extra_file, "r", encoding="utf-8") as f:
                lines = [ln.strip() for ln in f if ln.strip() and not ln.strip().startswith("#")]
            if lines:
                base = base + "\n\n以下补充说明请一并遵守：\n" + "\n".join(lines)
        except Exception:
            pass
    # 可选：马铃薯相关知识，回答时优先依据这些内容
    knowledge = _load_knowledge()
    if knowledge:
        base = base + "\n\n以下是你需要掌握并优先参考的马铃薯相关知识，回答时尽量基于这些内容：\n" + knowledge
    # 可选：发几条示例让模型「照着风格/内容」回答（few-shot，无需训练）
    fewshot = _load_fewshot_examples()
    if fewshot:
        base = base + "\n\n请参考以下示例的回答风格与内容，遇到类似问题按示例方式回答：\n" + fewshot
    # 当前月份易发病提示，用户问「最近要注意什么」时可引用
    base = base + "\n\n当前季节提示（用户问近期注意事项、最近要注意什么时可引用）：" + get_monthly_tip()
    return base


def _load_knowledge():
    """从项目根目录 llm_knowledge.txt 或 llm_knowledge.md 读取知识库内容（# 开头的行会保留，可作为标题）。"""
    for name in ("llm_knowledge.txt", "llm_knowledge.md"):
        fpath = PROJECT_ROOT / name
        if fpath.exists():
            try:
                with open(fpath, "r", encoding="utf-8") as f:
                    return f.read().strip()
            except Exception:
                pass
    return ""


def _load_fewshot_examples():
    """从项目根目录 llm_fewshot.txt 读取「问/答」示例，格式：问：xxx 答：yyy，空行分隔多组。"""
    fpath = PROJECT_ROOT / "llm_fewshot.txt"
    if not fpath.exists():
        return ""
    try:
        with open(fpath, "r", encoding="utf-8") as f:
            text = f.read()
    except Exception:
        return ""
    out = []
    for block in text.split("\n\n"):
        block = block.strip()
        if not block or block.startswith("#"):
            continue
        q, a = None, None
        for line in block.split("\n"):
            line = line.strip()
            if not line:
                continue
            if line.startswith("问：") or line.startswith("Q:"):
                q = line.split("：", 1)[-1].split(":", 1)[-1].strip()
            elif line.startswith("答：") or line.startswith("A:"):
                a = line.split("：", 1)[-1].split(":", 1)[-1].strip()
        if q and a:
            out.append(f"问：{q}\n答：{a}")
    return "\n\n".join(out) if out else ""


def ask(question: str, history: Optional[List[dict]] = None) -> Tuple[Optional[str], Optional[str]]:
    """
    问答，支持多轮上下文。history 为之前的对话列表，每项 {"role":"user"|"assistant", "content":"..."}。
    不传或传空则单轮。返回 (answer, error)。
    """
    if not question or not question.strip():
        return None, "问题不能为空"
    system = _get_ask_system_prompt()
    if not history:
        return _chat(system, question.strip())
    messages = list(history) + [{"role": "user", "content": question.strip()}]
    return _chat_messages(system, messages)


# 根据检测结果生成解读的 system prompt
SYSTEM_EXPLAIN = """你是马铃薯病害识别结果的解读助手。根据给出的检测结果（病害类别与置信度），用一两段话简要说明：
1）检测到的病害或健康状态含义；
2）若为病害，简要防治或管理建议。
若检测结果为多类，以置信度最高的一类为主说明。语言简洁、面向种植户或农技人员。"""


def explain_detection(results: List[dict]) -> Tuple[Optional[str], Optional[str]]:
    """
    根据 YOLO 检测结果 results（每项含 class, confidence, box）生成文字解读。
    返回 (explanation, error)。
    """
    if not results:
        return "未检测到病害或健康目标，可尝试更换图片或角度。", None
    parts = [f"- {r.get('class', '')}（置信度 {r.get('confidence', 0):.2f}）" for r in results]
    user_text = "当前检测结果：\n" + "\n".join(parts)
    return _chat(SYSTEM_EXPLAIN, user_text)


def is_coze_available():
    """扣子智能体是否可用：已开启且已配置 bot_id 和 PAT。"""
    if not ENABLE_COZE:
        return False
    return bool(COZE_BOT_ID and COZE_PAT)


def ask_coze(question: str, history: Optional[List[dict]] = None) -> Tuple[Optional[str], Optional[str]]:
    """
    调用扣子智能体问答，支持多轮上下文。返回 (answer, error)。
    """
    if not is_coze_available():
        return None, "扣子智能体未启用或未配置（请设置 ENABLE_COZE、COZE_BOT_ID 和 COZE_PAT）"

    if not question or not question.strip():
        return None, "问题不能为空"

    try:
        import requests
    except ImportError:
        return None, "请安装 requests：pip install requests"

    url = "https://api.coze.cn/open_api/v2/chat"
    headers = {
        "Authorization": f"Bearer {COZE_PAT}",
        "Content-Type": "application/json",
    }

    # Coze API格式：使用query字段和conversation_id
    query = question.strip()
    conversation_id = "new"  # 对于新对话使用"new"

    data = {
        "bot_id": COZE_BOT_ID,
        "conversation_id": conversation_id,
        "user": "potato_backend_user",
        "query": query,
        "stream": False
    }

    try:
        response = requests.post(url, headers=headers, json=data, timeout=30)
        response.raise_for_status()
        result = response.json()

        # 检查响应格式：Coze API 直接返回 messages 数组
        if result.get("code") == 0 and "messages" in result:
            messages = result["messages"]
            if messages:
                # 找到最后一个assistant消息
                for msg in reversed(messages):
                    if msg.get("role") == "assistant" and msg.get("type") == "answer" and msg.get("content"):
                        return msg["content"].strip(), None

        # 如果没有找到assistant消息，检查是否有错误信息
        if result.get("code") != 0:
            error_msg = result.get("msg", result.get("message", "未知错误"))
            return None, f"Coze API错误: {error_msg}"

        return None, "未收到有效的回复"

    except requests.exceptions.RequestException as e:
        return None, f"请求失败: {str(e)}"
    except Exception as e:
        return None, f"解析响应失败: {str(e)}"
