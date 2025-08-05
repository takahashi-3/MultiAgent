from typing import List, TypedDict, Dict
import time
import json
from openai import OpenAI

childAgent_systemPrompt = "あなたは顧客として、ユーザに対して発言を行うという役目を持っているエージェントです。"

## state ######################################################################################
class ChildAgent():
    """
    agent_name(str) = エージェントの名前
    agent_personality(Dict[str, str]) = エージェントのパーソナリティ
    agent_task(str) = エージェントのタスク
    current_target(str) = 現在ユーザが対応を行っているエージェント
    history(List[str]) = 会話の履歴
    model_name(str) = 推論を行わせるモデル名
    response: この時点におけるエージェントの発言(出力のみ)
    return_state: 現在接客を受けているエージェントがクレームを行う場合、または設定されたタスクが完了していないと判定した場合""（ユーザの発話へ）, タスクが完了していると判断した場合"エージェント名"(タスク生成へ)
    thema(str) = 会話のテーマ
    """

    def __init__(self, parent_prompt:str, agent_name:str, agent_personality:str, agent_task:str):
        self.parent_prompt = parent_prompt
        self.agent_name = agent_name
        self.agent_personality = agent_personality
        self.agent_task = agent_task
        self.client = OpenAI()
        self.history = ""
        self.internal_state = ""
        self.target = ""
        self.thema = "日本の飲食店"
        self.utterance = ""

        print(f"name:{self.agent_name}, personality:{self.agent_personality}, task:{self.agent_task}, prompt:{self.parent_prompt}")
        
        self.inference()

    def main(self):
        while():
            pass

    def inference(self): # 発話
        prompt = "以下の行動方針に沿って、発言を行うか決定し、発言を行う場合にはその内容をjson形式で出力してください。また自身の内部的な状態についても出力してください。" + "出力形式：{\"customer\": {\"internal_state\": {\"\"}, \"utterance\": {\"\"}}}" + f"\n行動方針:{self.parent_prompt}" + f"\nあなたのパーソナリティ:{self.agent_personality}\nこれまでの対話履歴:{self.history}"

        api_response = self.client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[
                    {"role": "system", "content": childAgent_systemPrompt},
                    {"role": "user", "content": prompt}
                ]
        )

        json_response = api_response.choices[0].message.content
        print(json_response)
        json_list = json.loads(json_response)['customer']

        self.internal_state, self.utterance = json_list['internal_state'], json_list['utterance']

        print(f"{self.agent_name}: {self.utterance}")

    def wait(self, wait_time:int):
        time.sleep(wait_time)
        return