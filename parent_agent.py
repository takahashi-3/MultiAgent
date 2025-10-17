from typing import Annotated, List, TypedDict, Dict, Any
from openai import OpenAI
import json
import os

from child_agent import ChildAgent

os.environ['OPENAI_API_KEY'] = ""

parentAgent_systemPrompt = "あなたは複数の顧客エージェントの行動を管理し、より現実的で多様な訓練状況を作成するという役目を持っているエージェントです。"
thema = "日本の飲食店"

# Graph全体のstate
class ParentAgent():
    """
    agent_tasks(Dict[str, str]) = エージェントの名前をキー, タスクをコンテンツとする辞書
    current_target(str) = 現在ユーザが対応を行っているエージェント
    feedbacks(List[str]) = フィードバックの内容
    history(List[str]) = 会話の履歴
    init_flag(bool) = task_generatorで初期タスクの生成を行うか(True),タスクの更新を行うか(False)の判断をするためのflag
    model_name(str) = 推論を行わせるモデル名(利用するLLMのAPIに基づいた名前を設定してください. ここではGroq)
    past_utterance(str) = 前回のユーザ発話
    shared_value(Any) = ユーザ入力のプロセスとデータをやり取りするための共有変数
    speakers(List[Dict[str]]) = 客役エージェントの名前をキー, パーソナリティをコンテンツとして持つ辞書
    speakers_names(List[str]) = 訓練に参加しているエージェントの名前
    subgraph(Any) = サブグラフのインスタンス?
    task_number(int) = 訓練全体で処理すべきタスクの数（この数のタスクを完了したら訓練終了）
    task_state(Dict[str, bool]) = 現在の２つのタスクが終了しているかどうか(key:エージェント名, content:タスクの状態(True:完了 , False:未完了))
    thema(str) = 会話のテーマ
    """
    def __init__(self, thema:str, train_scale:int):
        self.agent_tasks = {} # {"name": "task", "name": "task", ...}
        self.client = OpenAI()
        self.history = ""
        self.train_scale = train_scale
        self.thema = thema

        self.childrenList = []
        self.personalities = {}
        self.tasks = {}

        self.make_init_context()
        self.main_loop()


    def make_init_context(self): # 各エージェントのコンテキスト作成
        prompt = f"あなたは今、{self.thema}において{self.train_scale}人の顧客を管理する必要があります。今後の訓練の展開によりバリエーションが生まれるように彼らの名前および詳細なパーソナリティを設定してください。また少なくとも一人は厄介な顧客が存在するようにパーソナリティを設定してください。\n" + "出力はjson形式で行ってください。\n{\"customers\": [{\"name\": \"\", \"personality\": \"気難しい性格で、サービスや料理に対して文句をつけることが多い\"}, ]}"
        api_response = self.client.chat.completions.create(
        model="gpt-3.5-turbo",
        messages=[
                {"role": "system", "content": parentAgent_systemPrompt},
                {"role": "user", "content": prompt}
            ]
        )

        json_response = api_response.choices[0].message.content
        json_list = json.loads(json_response)['customers']

        for customer in json_list:
            self.personalities[customer['name']] = customer['personality']

        #print(self.personalities)

    def children_manage(self):
        prompt = f"あなたは今、{self.thema}において{self.train_scale}人の顧客エージェントを管理する必要があります。これまでの対話履歴をもとに、現時点における各顧客エージェントの行動を決定するために、客としての接客内容を割り当ててください。また、その内容に沿って顧客として行動するためのプロンプトも生成してください\n" + "出力は以下のようなjson形式で行ってください。\n{\"customers\": [{\"name\": \"\", \"task\": \"\", \"prompt\": \"\"}, ]}" + "これまでの対話履歴:" + self.history + "\n顧客の名前とパーソナリティ:" + str(self.personalities)
        api_response = self.client.chat.completions.create(
        model="gpt-3.5-turbo",
        messages=[
                {"role": "system", "content": parentAgent_systemPrompt},
                {"role": "user", "content": prompt}
            ]
        )

        json_response = api_response.choices[0].message.content
        json_list = json.loads(json_response)['customers']

        for customer in json_list:
            self.childrenList.append(ChildAgent(parent_prompt=customer['prompt'], agent_name=customer['name'], agent_personality=self.personalities[customer['name']], agent_task=customer['task']))
            self.tasks[customer['name']] = customer['task']

    def main_loop(self):
        self.children_manage()

if __name__ == "__main__":
    thema = "日本の飲食店"
    train_scale = 4
    parent_agent = ParentAgent(thema=thema, train_scale=train_scale)
