import random
import operator
import os

from typing import Annotated, Any, List, TypedDict, Dict, Union
from pydantic import BaseModel, Field

from langchain_core.messages import HumanMessage, SystemMessage
from langchain_openai import ChatOpenAI

import global_value as g

# 訓練環境の設定(各タスクのoptionとして機能する)
ENV_SETTING = """
メニュー: ['パンケーキ', 'ハンバーガーセット', 'バゲットセット', 'サンドウィッチセット', 'チョコレートケーキ', 'ピザ']
座席: ['カウンター席', 'テーブル席']
"""
INIT_TASK_SITUATION = ['入店', '料理の注文', '料理の配膳', '片付け', 'クレーム']

# 同時に行動するエージェントの数
SYNCRO_CUSTOMER_NUMBER = 2

# タスクの発生順序
TASK_PROCEDURE_EXPLANATION = '接客タスクは基本的に \'入店\' -> \'料理の注文\' -> \'料理の配膳\' -> \'片付け\' の順に発生します。'

# タスク終了後の待ち時間に関する概念の説明
# WAIT_TIME_EXPLANATION = '特定の客に対する接客の終了後、その客が次の接客タスクを発生させるまでには、基本的に時間を要します。（例：客Nに対して、\'料理の配膳\'を終えた後、客Nは配膳された料理を食べ始め、料理を食べ終えた後で\'片付け\'のタスクが発生する）'

# 指示役LLMによるタスク割り当ての出力形式
class Format_task_assign(BaseModel):
    """
    #### 指示役LLMによるタスク割り当て処理の出力形式\n
    agent_name(List[str]) = タスクを行う顧客役エージェントの名前\n
    tasks(List[str]) = エージェントが行う接客タスクの名称
    """
    agent_name: List[str] = Field(description='The names of Customer-Agent that request service to User.')
    tasks: List[str] = Field(description='The tasks that be caused by Customer-Agent. each element correspond to \"agent_name\" elements')
    options: List[str] = Field(description='The details of the task. each element correspond to \"tasks\" elements. For example, if a \'tasks\' element is \'料理の注文\', one of the menu contents is selected to the correspond \'option\' elements.')

# Graph全体のstate
class AppState(TypedDict):
    """
    #### 親Graphにおいて、ノード間でやり取りされる情報(State)\n
    agent_tasks(Dict[str, Dict[str, str]]) = エージェントの名前をキー, タスクをコンテンツとする辞書\n
    current_speakers_names(List[str]) = 現在のフェーズにおいて行動を起こしているエージェントのリスト\n
    current_target(str) = 現在接客の対象となっているエージェント\n
    feedbacks(List[str]) = フィードバックの内容\n
    history(List[str]) = 会話の履歴\n
    history_for_each_agent(Dict[str, List[str]]) = 各顧客役エージェントごとの履歴\n
    init_flag(bool) = task_generatorで初期タスクの生成を行うか(True),タスクの更新を行うか(False)の判断をするためのフラグ\n
    model_name(str) = 推論を行わせるモデル名(利用するLLMのAPIに基づいた名前を設定してください.)\n
    speakers_personality(Dict[str, str]) = 客役エージェントの名前をキー, パーソナリティをコンテンツとして持つ辞書\n
    speakers_names(List[str]) = 訓練に参加しているエージェントの名前(客役のプールとして機能する)\n
    subgraph(Any) = サブグラフのインスタンス\n
    task_number(int) = 訓練全体で処理すべきタスクの数（この数のタスクを完了したら訓練終了）\n
    task_state(Dict[str, bool]) = 現在の２つのタスクが終了しているかどうか(key:エージェント名, content:タスクの状態(True:完了 , False:未完了))\n
    thema(str) = 会話のテーマ
    """
    agent_tasks:  Dict[str, Dict[str, str]]
    current_speakers_names: List[str]
    current_target: str
    feedbacks: Annotated[List[str], operator.add]
    history: Annotated[List[str], operator.add]
    history_for_each_agent: Dict[str, List[str]]
    init_flag: bool
    model_name: str
    speakers_personality: Dict[str, str]
    speakers_names: List[str]
    subgraph: Any # 型がわからないのでひとまず任意型です
    task_number: int
    task_state: Dict[str, bool]
    thema: str

def check_controller_prompt(prompt: str, task_dict: Dict[str, Dict[str, str]]):
    # コンテキストの確認
    os.makedirs(f'./{g.output_dir}/prompt', exist_ok=True)
    with open(f'./{g.output_dir}/prompt/controller_prompt.txt', 'a') as fp:
        fp.write(prompt + f'\n\n出力:: 割り当てタスク:{task_dict}\n\n\n')

def check_task_assign(task_dict: Dict[str, Dict[str, str]], history_for_each_agent: Dict[str, List[str]]) -> None:
    """
    ####指示役LLMによるタスク割り当て内容を確認するための関数です.(./task_assign/assigned_task.txt に保存)\n
    Args: task_dict(Dict[str, Dict[str]]) = 各顧客役エージェントのタスク及びそのオプション\n
          history_for_each_agent(Dict[str, List[str]]) =  各顧客役エージェントのユーザとの対話履歴\n
    Return: None
    """
    exported_to_file = '{'

    if(len(history_for_each_agent.keys()) == 0): #初期状態
        for agent_name in task_dict.keys():
            to_file_for_each_agent = '{'
            to_file_for_each_agent += f'名前:{agent_name}\n'
            to_file_for_each_agent += f'タスク:{task_dict[agent_name]["task"]}\nタスクのオプション:{task_dict[agent_name]["option"]}'
            to_file_for_each_agent += '},\n'
            exported_to_file += to_file_for_each_agent
    else:
        for agent_name in history_for_each_agent.keys():
            to_file_for_each_agent = '{'
            to_file_for_each_agent += f'名前:{agent_name}\n'

            to_file_for_each_agent += f'\t{agent_name}の対話履歴:[\n'
            for temp_history in history_for_each_agent[agent_name]:
                for utt in temp_history.split('\n'):
                        if(utt != ''):
                            to_file_for_each_agent += '\t\t' + utt + '\n'
            to_file_for_each_agent += '\n]'

            if(agent_name in task_dict.keys()):
                to_file_for_each_agent += f',\nタスク:{task_dict[agent_name]["task"]}\nタスクのオプション:{task_dict[agent_name]["option"]}'

            to_file_for_each_agent += '},\n'
            exported_to_file += to_file_for_each_agent

    exported_to_file += '}'

    os.makedirs(f'./{g.output_dir}/task_assign/', exist_ok=True)
    with open(f'./{g.output_dir}/task_assign/assigned_task.txt', 'a') as fp:
        fp.write(exported_to_file + '\n\n')


def get_prompt_history_for_each_agent(speakers_personality: Dict[str, str], history_for_each_agent: Dict[str, List[str]]) -> str:
    """
    ####客ごとのパーソナリティ＋対話履歴をまとめ上げ、指示役LLMにタスク割り当てを行わせるようのコンテキストを作成します。\n
    Args: speakers_personality(Dict[str, str]) = 各顧客役エージェントのパーソナリティ\n
          history_for_each_agent(Dict[str, List[str]]) =  各顧客役エージェントのユーザとの対話履歴\n
    Return: str = プロンプト用のコンテキスト
    """
    context = '{'

    for agent_name in speakers_personality.keys():
        context_for_each_agent = '{'
        context_for_each_agent += f'名前:{agent_name}, 性格:{speakers_personality[agent_name]}\n'

        if(agent_name in history_for_each_agent.keys()):
            context_for_each_agent += f'\t{agent_name}の対話履歴:[\n'
            for temp_history in history_for_each_agent[agent_name]:
                for utt in temp_history.split('\n'):
                    if(utt != ''):
                        context_for_each_agent += '\t\t' + utt + '\n'
            context_for_each_agent += '\n]'
        
        context_for_each_agent += '},\n'
        context += context_for_each_agent

    context += '訓練環境' + ENV_SETTING
    context += '\n}'

    return context


def task_generator(state: AppState):
    """
    客ごとの接客タスクを生成します。\n
    Args: state(AppState)\n
    Return: Dict[str] = 客ごとのタスク
    """
    init_flag = state.get('init_flag', False)
    model_name = state.get('model_name')
    current_speakers_names = state.get('current_speakers_names', [])
    history_for_each_agent = state.get('history_for_each_agent', {})
    speakers_names = state.get('speakers_names', [])
    speakers_personality = state.get('speakers_personality', {})
    task_number = state.get('task_number', 0)

    task_state = {}

    model = ChatOpenAI(model=model_name, temperature=0.0)
    
    # タスクを割り当てるエージェント数の決定(人手定義)
    # 事前定義された割り当てエージェント数(assign_agent_number)よりも、残タスク数(task_number)が小さい場合、
    # 残タスク数分のエージェントにタスクを割り当てる
    assign_agent_number = 2
    if(task_number < assign_agent_number):
        assign_agent_number = task_number

    # 指示役LLMに訓練状況＋パーソナリティを提示して、タスクを割り当てさせる
    if(init_flag):
        thema = state.get("thema")
        task_dict = {}

        # 初期に発生するタスクを設定(訓練状況は無し)
        system_message = f"あなたには、{INIT_TASK_SITUATION}の中から顧客役エージェントたちの発生させる接客タスクを選択するという役割が課されています。"
        #human_message = f"{thema}というテーマにおいて、全顧客役エージェントから2名を選び出し、それらの顧客役エージェントが店員に対して、発生させる接客タスクを{INIT_TASK_SITUATION}から１つずつ選択してください。\n\n#全顧客役エージェント:{speakers_names}"
        human_message = f"{thema}というテーマにおいて、全顧客役エージェントから{assign_agent_number}名を選び出し、それらの顧客役エージェントが店員に対して、発生させる接客タスクを{INIT_TASK_SITUATION}から１つずつ選択してください。タスクを選択する際には、\'コンテキスト\'を参照して顧客役エージェントの性格や接客タスクの内容などを考慮してください。\n\n#全顧客役エージェント:{speakers_names}\n\n#コンテキスト:{get_prompt_history_for_each_agent(speakers_personality, history_for_each_agent)}"
        
        structured_response_model = model.with_structured_output(Format_task_assign)
        response = structured_response_model.invoke([SystemMessage(system_message), HumanMessage(human_message)])
        
        current_speakers_names = response.agent_name
        for agent_name, task, option in zip(current_speakers_names, response.tasks, response.options):
            task_dict[agent_name] = {'task': task, 'option': option}

        # 発話対象への追加(訓練に登場するエージェントのみ残す)
        for customer_name in current_speakers_names:
            task_state[customer_name] = False

        # タスク割り当てがちゃんとできているか確認(Debug)
        check_controller_prompt(human_message, task_dict)
        check_task_assign(task_dict, history_for_each_agent)

        return {"agent_tasks": task_dict, "current_speakers_names": current_speakers_names, "speakers_names": speakers_names, "task_state":task_state}
    else:
        previous_task_dict = state.get("agent_tasks", {})
        task_dict = {}
        thema = state.get("thema", "")

        # 前のフェーズで接客を行われなかった顧客のタスクを取得
        #for i in current_speakers_names:
        #    if(i in previous_task_dict.keys()):
        #        task_dict[i] = previous_task_dict[i]

        # タスクの更新
        system_message = f"あなたには、{INIT_TASK_SITUATION}の中から顧客役エージェントたちの発生させる接客タスクを選択するという役割が課されています。"
        human_message = f"{thema}というテーマにおいて、全顧客役エージェントから{assign_agent_number}名を選び出し、それらの顧客役エージェントが店員に対して、発生させる接客タスクを{INIT_TASK_SITUATION}から１つずつ選択してください。タスクを選択する際には、\'コンテキスト\'や\'接客タスクの発生順序\'を参照して顧客役エージェントの性格やタスクの内容、これまでの対話履歴などを考慮してください。\n\n#全顧客役エージェント:{speakers_names}\n\n#接客タスクの発生順序:{TASK_PROCEDURE_EXPLANATION}\n\n#コンテキスト:{get_prompt_history_for_each_agent(speakers_personality, history_for_each_agent)}"
        #check_controller_prompt(human_message)
        
        structured_response_model = model.with_structured_output(Format_task_assign)
        response = structured_response_model.invoke([SystemMessage(system_message), HumanMessage(human_message)])
        current_speakers_names = response.agent_name

        #for agent_name, task in zip(current_speakers_names, response.tasks):
        #    task_dict[agent_name] = task
        for agent_name, task, option in zip(current_speakers_names, response.tasks, response.options):
            task_dict[agent_name] = {'task': task, 'option': option}

        # 発話対象への追加(訓練に登場するエージェントのみ残す)
        for customer_name in current_speakers_names:
            task_state[customer_name] = False

        # タスク割り当てがちゃんとできているか確認(Debug)
        check_controller_prompt(human_message, task_dict)
        check_task_assign(task_dict, history_for_each_agent)

        return {"agent_tasks": task_dict, "current_speakers_names": current_speakers_names, "speakers_names": speakers_names, "task_state":task_state}


# 状況の説明を行うもの
# クレーム,料理の配膳,入店,料理の注文,片付け
def task_to_situation(customer_name: str, task: str, option: str) -> str:
    if(task == "料理の配膳"):
        return f"*System:客{customer_name}に{option}を配膳する準備ができました\n"
    elif(task == "入店"):
        return f"*System:客{customer_name}が来店しました\n"
    elif(task == "片付け"):
        return f"*System:客{customer_name}の皿が空になっています\n"
    elif(task == 'クレーム'):
        return f'*System:客{customer_name}が不満そうです\n'
    else:
        return f"*System:客{customer_name}が挙手しています\n" # 主に料理の注文に対応（指示役LLMが新たにタスクを作り出す場合でも対応可能）
        

def situation_generator(state: AppState):
    """
    二人の客役が同時に行動を起こす状況を生成します。
    Args: state(AppState)
    Return: Dict[str] = 履歴の作成
    """
    agent_tasks = state.get("agent_tasks")
    speaker_name_inPool = state.get('speakers_names') # 全顧客エージェントの名前(訓練に未参加の者も含む)
    speaker_name_inEnv = state.get('current_speakers_names') # 現在の訓練に参加中の顧客エージェントの名前
    history_for_each_agent = state.get('history_for_each_agent') # 各顧客エージェントの対話履歴
    
    situation: str = "" # 意思表示の出力を格納


    for agent_name in speaker_name_inPool:
        if(agent_name in speaker_name_inEnv):
            situation += task_to_situation(agent_name, agent_tasks[agent_name]['task'], agent_tasks[agent_name]['option'])
            
    print(situation, end="")

    # 訓練に参加中の全エージェントに意思表示の状況を共有
    for agent_name in speaker_name_inPool:
        if(agent_name in speaker_name_inEnv):
            if(not (agent_name in history_for_each_agent.keys())):
                history_for_each_agent[agent_name] = [situation]
            else:
                history_for_each_agent[agent_name].append(situation)

    return {"history": [situation], 'history_for_each_agent': history_for_each_agent, "init_flag": False}
