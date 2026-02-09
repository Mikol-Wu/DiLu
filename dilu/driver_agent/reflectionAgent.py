import os
import re
import json
import textwrap
import time
from typing import Tuple, Optional, Dict, Any

from langchain.chat_models import AzureChatOpenAI, ChatOpenAI
from langchain.schema import HumanMessage, SystemMessage
from rich import print


def _extract_json(text: str) -> Optional[Dict[str, Any]]:
    # find first {...} block
    m = re.search(r"\{[\s\S]*\}", text)
    if not m:
        return None
    try:
        return json.loads(m.group(0))
    except Exception:
        return None


class ReflectionAgent:
    def __init__(self, temperature: float = 0.0, verbose: bool = False) -> None:
        oai_api_type = os.getenv("OPENAI_API_TYPE")
        if oai_api_type == "azure":
            self.llm = AzureChatOpenAI(
                deployment_name=os.getenv("AZURE_CHAT_DEPLOY_NAME"),
                temperature=temperature,
                max_tokens=1200,
                request_timeout=60,
            )
        else:
            # openai-compatible (OpenAI or DeepSeek)
            self.llm = ChatOpenAI(
                temperature=temperature,
                model_name=os.getenv("OPENAI_CHAT_MODEL", "deepseek-reasoner"),
                openai_api_base=os.getenv("OPENAI_API_BASE"),
                openai_api_key=os.getenv("OPENAI_API_KEY"),
                max_tokens=1200,
                request_timeout=60,
            )

    def reflection(self, human_message: str, llm_response: str) -> str:
        corrected_text, _ = self.reflection_with_rule(human_message, llm_response)
        return corrected_text

    def reflection_with_rule(self, human_message: str, llm_response: str) -> Tuple[str, Optional[Dict[str, Any]]]:
        delimiter = "####"
        system_message = textwrap.dedent(f"""\
        You are a driving-safety analyst. You will be given the user's prompt and the model's response.
        The response led to a collision later. Your job:
        1) Identify the reasoning mistake precisely.
        2) Provide corrected reasoning and a corrected action.
        3) Output an additional JSON rule that can prevent similar mistakes in the future.

        Output format:
        {delimiter} Analysis of the mistake:
        ...
        {delimiter} What should ChatGPT do to avoid such errors in the future:
        ...
        {delimiter} Corrected version of ChatGPT response:
        ...
        {delimiter} Rule JSON:
        <a single JSON object, no markdown, no code fences>
        """)

        human_prompt = textwrap.dedent(f"""\
        ``` Human Message ```
        {human_message}
        ``` ChatGPT Response ```
        {llm_response}

        Notes:
        - The corrected response must end with: "Response to user:#### <action_id>" where action_id is 0-4.
        - The Rule JSON must include these keys:
          - risk_type: string
          - trigger: string (a condition, can be pseudo-code)
          - bad_action: int (0-4)
          - good_action: int (0-4)
          - rationale: string
        """)

        print("Self-reflection (structured) is running...")
        start_time = time.time()
        messages = [SystemMessage(content=system_message), HumanMessage(content=human_prompt)]
        response = self.llm(messages).content
        print("Reflection done. Time taken: {:.2f}s".format(time.time() - start_time))

        # Keep same behavior as old project: store "avoid future errors" section as a memory chunk
        target_phrase = f"{delimiter} What should ChatGPT do to avoid such errors in the future:"
        idx = response.find(target_phrase)
        if idx >= 0:
            substring = response[idx + len(target_phrase):].strip()
        else:
            substring = response.strip()

        corrected_memory = f"{delimiter} I have made a misake before and below is my self-reflection:\n{substring}"

        # Parse JSON rule
        rule = None
        rule_marker = f"{delimiter} Rule JSON:"
        ridx = response.find(rule_marker)
        if ridx >= 0:
            rule_text = response[ridx + len(rule_marker):].strip()
            rule = _extract_json(rule_text)

        return corrected_memory, rule
