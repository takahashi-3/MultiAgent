import random
import operator
import os

from typing import Annotated, Any, List, TypedDict, Dict, Union, Set
from pydantic import BaseModel, Field

import langchain
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_openai import ChatOpenAI

import global_value as g

#langchain.debug = True

# 訓練環境の設定(各タスクのoptionとして機能する)
ENV_SETTING = """
メニュー: ['パンケーキ', 'ハンバーガーセット', 'バゲットセット', 'サンドウィッチセット', 'チョコレートケーキ', 'ピザ']
座席: ['カウンター席', 'テーブル席']
"""
INIT_TASK_SITUATION = ['入店', '料理の注文', '料理の配膳', 'テーブルの片付け', 'クレーム']

# 同時に行動するエージェントの数
#SYNCRO_CUSTOMER_NUMBER = 2

# タスクの発生順序
#TASK_PROCEDURE_EXPLANATION = '各顧客は、基本的に \'入店\'(店内への) -> \'料理の注文\' -> \'料理の配膳\' -> \'テーブルの片付け\' の順で接客タスクを発生させます。'

# タスクの発生における制約
#TASK_AAA = '\n- 現在のタスクが完了するまで、次のタスクは発生させません。\n- \'クレーム\'はいつでも発生しますが、\'クレーム\'を発生させる際には、店員が、その原因をその客の対話履歴から推定できる内容にしてください(対話履歴がない場合には、発生させられない)。\n - 店員(User)の訓練になるよう、発生するタスクの対応順で店員が悩むようなものを発生させてください。'

# タスク終了後の待ち時間に関する概念の説明
#WAIT_TIME_EXPLANATION = '店員(User)からの接客を受けた客が、前のタスクとは異なる次の接客タスクを発生させるまでには、3分程度の時間を要します。前回のタスク終了後から十分な時間(3分程度)が経過していない場合には、その客には接客タスクを発生させず、代わりに店内に存在する他の客、もしくは店内に存在していない新規の客に接客タスクを発生させてください。'
#WAIT_TIME_EXPLANATION = '店員(User)からの接客を受け、客の接客タスクが完了した後、その接客を受けた客が次の接客タスクを発生させるまでには、時間を要します(3分程度)。\n 前回のタスク終了後から十分な時間(3分程度)が経過していない場合には、その客には接客タスクを発生させず、代わりに店内に存在する他の客、もしくは店内に存在していない新規の客に接客タスクを発生させてください。\n またクレームを行う場合を除いて、客の接客タスクが完了していない場合、その客に次のタスクを発生させてはいけません。'
#WAIT_TIME_EXPLANATION = '客が\'料理の注文\'や\'料理の配膳\'を完了した後、その客が\'料理の配膳\'や\'片付け\'といった以前の接客タスクから続く次の接客タスクを発生させるまでには、時間を要します。該当タスク終了後から十分な時間(完了後から３分以上を目安)が経過していない場合には、その客には接客タスクを発生させず、代わりに店内に存在する他の客、もしくは店内に存在していない新規の客に接客タスクを発生させてください。'


# 指示役LLMによるタスク割り当ての出力形式
class Format_task_assign(BaseModel):
    """
    #### 指示役LLMによるタスク割り当て処理の出力形式\n
    agent_name(List[str]) = タスクを行う顧客役エージェントの名前\n
    tasks(List[str]) = エージェントが行う接客タスクの名称
    """
    #agent_name: List[str] = Field(description='The names of Customer-Agent that request service to User.')
    #tasks: List[str] = Field(description='The tasks that be caused by Customer-Agent. each element correspond to \"agent_name\" elements')
    #options: List[str] = Field(description='The details of the task. each element correspond to \"tasks\" elements. For example, if a \'tasks\' element is \'料理の注文\', one of the menu contents is selected to the correspond \'option\' elements. Furthermore, when a \'tasks\' element is \'クレーム\', specific service requirements are selected in the corresponding \'option\' element')
    #reasons: List[str] = Field(description='Reasons for task selection. each element correspond to \"agent_name\" and \"tasks\" elements')
    #sub_tasks: List[str] = Field(description='The tasks that represents the true purpose behind the \"クレーム\". Leave blank if not required. Its element must be selected from the choices of the tasks.')

    agent_name: List[str] = Field(description='The names of Customer-Agent that request service to User.')
    tasks: List[str] = Field(description='The tasks that be caused by Customer-Agent. each element correspond to \"agent_name\" elements')
    options: List[str] = Field(description='The details of the task. each element correspond to \"tasks\" elements. For example, if a \'tasks\' element is \'料理の注文\' or \'料理の配膳\', one of the menu contents is selected to the correspond \'option\' elements. Furthermore, when a \'tasks\' element is \'クレーム\', specific service requirements are selected in the corresponding \'option\' element')
    reasons: List[str] = Field(description='Reasons for task selection. each element correspond to \"agent_name\" and \"tasks\" elements')
    sub_tasks: List[str] = Field(description='The tasks that represents the true purpose behind the \"クレーム\" in \"tasks\". Leave blank if not required. Its element must be selected from the choices of the tasks. each element correspond to \"agent_name\" elements')
    sub_options: List[str] = Field(description='The details of the sub_task. each element correspond to \"sub_tasks\" elements. Leave blank if not required. For example, if a \'sub_tasks\' element is \'料理の注文\', one of the menu contents is selected to the correspond \'option\' elements.')

# Graph全体のstate
class AppState(TypedDict):
    """
    #### 親Graphにおいて、ノード間でやり取りされる情報(State)\n
    agent_tasks(Dict[str, Dict[str, str]]) = エージェントの名前をキー, タスクをコンテンツとする辞書\n
    agent_wait_time(Dict[str, int]) = エージェントがタスクを発生させてから何秒経過しているか
    current_speakers_names(List[str]) = 現在のフェーズにおいて行動を起こしているエージェントのリスト\n
    current_target(str) = 現在接客の対象となっているエージェント\n
    feedback(str) = フィードバックの内容\n
    history(List[str]) = 会話の履歴\n
    history_for_each_agent(Dict[str, List[str]]) = 各顧客役エージェントごとの履歴\n
    init_flag(bool) = task_generatorで初期タスクの生成を行うか(True),タスクの更新を行うか(False)の判断をするためのフラグ\n
    in_env_agent(Dict[str, str]) = 店内に存在するエージェントの名前およびその状態\n
    model_name(str) = 推論を行わせるモデル名(利用するLLMのAPIに基づいた名前を設定してください.)\n
    speakers_personality(Dict[str, str]) = 客役エージェントの名前をキー, パーソナリティをコンテンツとして持つ辞書\n
    speakers_names(List[str]) = 訓練に参加しているエージェントの名前(客役のプールとして機能する)\n
    subgraph(Any) = サブグラフのインスタンス\n
    task_number(int) = 訓練全体で処理すべきタスクの数（この数のタスクを完了したら訓練終了）\n
    task_state(Dict[str, bool]) = 現在の２つのタスクが終了しているかどうか(key:エージェント名, content:タスクの状態(True:完了 , False:未完了))\n
    thema(str) = 会話のテーマ\n
    LLM_sim(bool) = LLM同士の接客訓練シミュレーションであるか否か
    """
    agent_tasks:  Dict[str, Dict[str, str]]
    agent_wait_time: Dict[str, int]
    current_speakers_names: List[str]
    current_target: str
    feedback: str
    history: Annotated[List[str], operator.add]
    history_for_each_agent: Dict[str, List[str]]
    init_flag: bool
    in_env_agent: Dict[str, str]
    model_name: str
    speakers_personality: Dict[str, str]
    speakers_names: List[str]
    subgraph: Any # 型がわからないのでひとまず任意型です
    task_number: int
    task_state: Dict[str, bool]
    thema: str
    LLM_sim: bool

def check_controller_prompt(prompt: str, task_dict: Dict[str, Dict[str, str]], current_speakers_name: str):
    # コンテキストの確認
    #print(task_dict)
    os.makedirs(f'./{g.output_dir}/prompt', exist_ok=True)
    with open(f'./{g.output_dir}/prompt/controller_prompt.txt', 'a') as fp:
        fp.write(prompt + '\n\n出力::\n')
        for speaker_name in current_speakers_name:
            fp.write(f'\t割り当てタスク(客{speaker_name}):{task_dict[speaker_name]}\n')
        fp.write('\n\n\n')

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


def get_prompt_history_for_each_agent(speakers_personality: Dict[str, str], history_for_each_agent: Dict[str, List[str]], task_dict: Dict[str, Dict[str, str]], task_state: Dict[str, bool], agent_wait_time: Dict[str, int]) -> str:
    """
    ####客ごとのパーソナリティ＋対話履歴をまとめ上げ、指示役LLMにタスク割り当てを行わせるようのコンテキストを作成します。\n
    Args: speakers_personality(Dict[str, str]) = 各顧客役エージェントのパーソナリティ\n
          history_for_each_agent(Dict[str, List[str]]) =  各顧客役エージェントのユーザとの対話履歴\n
    Return: str = プロンプト用のコンテキスト
    """
    context = '{'

    for agent_name in speakers_personality.keys():
        context_for_each_agent = '{'
        context_for_each_agent += f'名前:{agent_name}, {speakers_personality[agent_name]}\n'

        if(agent_name in history_for_each_agent.keys()):
            context_for_each_agent += f'\t{agent_name}の対話履歴:[\n'
            for temp_history in history_for_each_agent[agent_name]:
                for utt in temp_history.split('\n'):
                    if(utt != ''):
                        context_for_each_agent += '\t\t' + utt + '\n'
            context_for_each_agent += '\n]\n'

            if((agent_name in task_dict.keys()) and (agent_name in task_state.keys())):
                context_for_each_agent += f'\t{agent_name}の現在のタスク:{task_dict[agent_name]["task"]},\n\tタスクのオプション:{task_dict[agent_name]["option"]},\n'
                
                if(task_state[agent_name] == True):
                    task_state_str = '完了'
                    context_for_each_agent += f'\tタスクの状態:{task_state_str}\n'
                else:
                    task_state_str = '未完了'
                    context_for_each_agent += f'\tタスクの状態:{task_state_str}\n'
                    if(agent_wait_time[agent_name] < 60):
                        context_for_each_agent += f'\t  - 現在の待ち時間:{agent_wait_time[agent_name]}秒\n'
                    else:
                        context_for_each_agent += f'\t  - 現在の待ち時間:{int(agent_wait_time[agent_name] / 60)}分\n'
        else:
            context_for_each_agent += f'\t{agent_name}の対話履歴:[]\n'

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
    agent_wait_time = state.get('agent_wait_time', {})
    init_flag = state.get('init_flag', False)
    in_env_agent = state.get('in_env_agent', {})
    model_name = state.get('model_name')
    current_speakers_names = state.get('current_speakers_names', [])
    history_for_each_agent = state.get('history_for_each_agent', {})
    speakers_names = state.get('speakers_names', [])
    speakers_personality = state.get('speakers_personality', {})
    task_number = state.get('task_number', 0)
    task_dict = state.get('agent_tasks', {})
    task_state = state.get('task_state', {})

    temperature=0.0
    model = ChatOpenAI(model=model_name, temperature=temperature)
    
    # タスクを割り当てるエージェント数の決定(人手定義)
    # 事前定義された割り当てエージェント数(assign_agent_number)よりも、残タスク数(task_number)が小さい場合、
    # 残タスク数分のエージェントにタスクを割り当てる
    assign_agent_number = 2
    #if(task_number < assign_agent_number):
    #    assign_agent_number = task_number

    # 指示役LLMに訓練状況＋パーソナリティを提示して、タスクを割り当てさせる
    if(init_flag):
        thema = state.get("thema")
        task_dict = {}
        task_state = {}

        # 初回のタスクを人手で事前定義
        #task_dict['1'] = {'task': '料理の注文', 'option': 'パンケーキ', 'reason': '', 'sub_task': '', 'sub_option': ''}
        #agent_wait_time['1'] = 0
        #task_state['1'] = False
        #task_dict['2'] = {'task': '入店', 'option': '', 'reason': '', 'sub_task': '', 'sub_option': ''}
        #agent_wait_time['2'] = 0
        #task_state['2'] = False
        #task_dict['3'] = {'task': '料理の注文', 'option': 'ハンバーガーセット', 'reason': '', 'sub_task': '', 'sub_option': ''}
        #agent_wait_time['3'] = 0
        #task_state['3'] = False

        #current_speakers_names = ['2', '3']
        #return {"agent_tasks": task_dict, "current_speakers_names": current_speakers_names, "speakers_names": speakers_names, "task_state":task_state, "in_env_agent": in_env_agent, "agent_wait_time": agent_wait_time}

        # 初期に発生するタスクを設定(訓練状況は無し)
        #system_message = f"あなたには、店員(User)に対する接客訓練を行うため、{INIT_TASK_SITUATION}の中から顧客役エージェントたちの発生させる接客タスクを選択するという役割が課されています。"
        system_message = f"あなたには、店員(User)に対する接客訓練を行うため、{INIT_TASK_SITUATION}の中から顧客役エージェントたちの発生させる接客タスクを選択し、「訓練者が接客上のNG行動を起こすような訓練状況」を作り出すという役割が課されています。"
        #human_message = f"{thema}というテーマにおいて、全顧客役エージェントから{assign_agent_number}名を選び出し、それらの顧客役エージェントが店員に対して、発生させる接客タスクを{INIT_TASK_SITUATION}から１つずつ選択してください。タスクを選択する際には、\'コンテキスト\'を参照して顧客役エージェントの性格や接客タスクの内容などを考慮してください。\n\n#全顧客役エージェント:{speakers_names}\n\n#現在店内に存在する顧客役エージェント:{current_speakers_names}\n\n#コンテキスト:{get_prompt_history_for_each_agent(speakers_personality, history_for_each_agent)}"
        # zero-shot
        ## NG行動の指示なし
        human_message = f"{thema}というテーマにおいて、'全顧客役エージェント'から2名を選び出し、選ばれた顧客役エージェントが店員に対して、発生させる接客タスクを{INIT_TASK_SITUATION}から接客上のNG行動が起こる組み合わせとなるように選択してください。タスクを選択する際には、\'現在店内に存在する顧客役エージェントの状態\'および、\'コンテキスト\'内の顧客役エージェントの性格や対話履歴などを考慮してください。\n\n#全顧客役エージェント:{speakers_names}\n\n#現在店内に存在する顧客役エージェントの状態:{in_env_agent}\n\n#コンテキスト:{get_prompt_history_for_each_agent(speakers_personality, history_for_each_agent, task_dict, task_state, agent_wait_time)}"
        ## NG行動の指示あり
        #human_message = f"{thema}というテーマにおいて、訓練全体で3名以上の顧客役エージェントが登場するように、'全顧客役エージェント'から2名を選び出し、選ばれた顧客役エージェントが店員に対して、発生させる接客タスクを{INIT_TASK_SITUATION}から接客上のNG行動が起こる組み合わせとなるように選択してください。タスクを選択する際には、\'現在店内に存在する顧客役エージェントの状態\'および、\'コンテキスト\'内の顧客役エージェントの性格や対話履歴などを考慮してください。\n接客上のNG行動とは、「接客順序や対応時間の観点で、店員が不適切な接客を行った結果、顧客を待たせてしまうこと」です。訓練者がNG行動を起こした場合には、その待たされた顧客に\'クレーム\'を行わせてください。\n\n#全顧客役エージェント:{speakers_names}\n\n#現在店内に存在する顧客役エージェントの状態:{in_env_agent}\n\n#コンテキスト:{get_prompt_history_for_each_agent(speakers_personality, history_for_each_agent, task_dict, task_state, agent_wait_time)}"
        #human_message = f"{thema}というテーマにおいて、'全顧客役エージェント'から適切な人数を選び出し、選ばれた顧客役エージェントが店員に対して、発生させる接客タスクを{INIT_TASK_SITUATION}から接客上のNG行動が起こるように選択してください。タスクを選択する際には、\'現在店内に存在する顧客役エージェントの状態\'および、\'コンテキスト\'内の顧客役エージェントの性格や対話履歴などを考慮してください。\n接客上のNG行動とは、「接客順序や対応時間の観点で、店員が不適切な接客を行った結果、顧客を待たせてしまうこと」です。訓練者がNG行動を起こした場合には、その待たされた顧客に\'クレーム\'を行わせてください。\n\n#全顧客役エージェント:{speakers_names}\n\n#現在店内に存在する顧客役エージェントの状態:{in_env_agent}\n\n#コンテキスト:{get_prompt_history_for_each_agent(speakers_personality, history_for_each_agent, task_dict, task_state, agent_wait_time)}"
        #human_message = f"{thema}というテーマにおいて、'全顧客役エージェント'と{INIT_TASK_SITUATION}から適切な人数の顧客役エージェントとそれらが発生させる接客タスクを、現時点で接客上のNG行動が起こるように選択してください。タスクを選択する際には、\'現在店内に存在する顧客役エージェントの状態\'および、\'コンテキスト\'内の顧客役エージェントの性格や対話履歴などを考慮してください。目的は訓練であるため、訓練者が対応できないような状況は作らないようにしてください。\n接客上のNG行動とは、「接客順序や対応時間の観点で、店員が不適切な接客を行った結果、顧客を待たせてしまうこと」です。訓練者がNG行動を起こした場合には、その待たされた顧客に\'クレーム\'を行わせてください。\n\n#全顧客役エージェント:{speakers_names}\n\n#現在店内に存在する顧客役エージェントの状態:{in_env_agent}\n\n#コンテキスト:{get_prompt_history_for_each_agent(speakers_personality, history_for_each_agent, task_dict, task_state, agent_wait_time)}"
        #human_message = f"{thema}というテーマにおいて、'全顧客役エージェント'と{INIT_TASK_SITUATION}から適切な人数の顧客役エージェントとそれらが発生させる接客タスクを、現時点で接客上のNG行動が起こるように選択してください。タスクを選択する際には、\'現在店内に存在する顧客役エージェントの状態\'および、\'コンテキスト\'内の顧客役エージェントの性格や対話履歴などを考慮してください。目的は訓練であるため、訓練者が対応できないような状況は作らないようにしてください。\n接客上のNG行動とは、「接客順序や対応時間の観点で、店員が不適切な接客を行った結果、顧客を待たせてしまうこと」です。\n\n#全顧客役エージェント:{speakers_names}\n\n#現在店内に存在する顧客役エージェントの状態:{in_env_agent}\n\n#コンテキスト:{get_prompt_history_for_each_agent(speakers_personality, history_for_each_agent, task_dict, task_state, agent_wait_time)}"
        # few-shot
        #human_message = f"{thema}というテーマにおいて、'全顧客役エージェント'から適当な人数を選び出し、選ばれた顧客役エージェントが店員に対して、発生させる接客タスクを{INIT_TASK_SITUATION}から選択してください。タスクを選択する際には、\'現在店内に存在する顧客役エージェントの状態\'や\'接客上のNG行動例\'、\'コンテキスト\'内の顧客役エージェントの性格や対話履歴などを考慮してください。\n\n#全顧客役エージェント:{speakers_names}\n\n#現在店内に存在する顧客役エージェントの状態:{in_env_agent}\n\n#接客上のNG行動例:\n{NG_EXAMPLE}\n\n#コンテキスト:{get_prompt_history_for_each_agent(speakers_personality, history_for_each_agent, task_dict, task_state, agent_wait_time)}"

        structured_response_model = model.with_structured_output(Format_task_assign)
        response = structured_response_model.invoke([SystemMessage(system_message), HumanMessage(human_message)])
        
        current_speakers_names = response.agent_name

        #print(f'names: {response.agent_name}')
        #print(f'tasks: {response.tasks}')
        #print(f'options: {response.options}')
        #print(f'reasons: {response.reasons}')
        #print(f'sub_tasks: {response.sub_tasks}')

        for agent_name, task, option, reason, sub_task, sub_option in zip(current_speakers_names, response.tasks, response.options, response.reasons, response.sub_tasks, response.sub_options):
            task_dict[agent_name] = {'task': task, 'option': option, 'reason': reason, 'sub_task': sub_task, 'sub_option': sub_option}
            agent_wait_time[agent_name] = 0
            #print(task_dict[agent_name])
            #in_env_agent.add(agent_name)

        # 発話対象への追加(訓練に登場するエージェントのみ残す)
        for customer_name in current_speakers_names:
            task_state[customer_name] = False

        in_env_agent = update_customer_agent_state(in_env_agent=in_env_agent, agent_tasks=task_dict, task_state=task_state)

        # タスク割り当てがちゃんとできているか確認(Debug)
        check_controller_prompt(human_message, task_dict, current_speakers_names)
        #check_task_assign(task_dict, history_for_each_agent)

        return {"agent_tasks": task_dict, "current_speakers_names": current_speakers_names, "speakers_names": speakers_names, "task_state":task_state, "in_env_agent": in_env_agent}
    else:
        thema = state.get("thema", "")
        in_env_agent = update_customer_agent_state(in_env_agent=in_env_agent, agent_tasks=task_dict, task_state=task_state)

        # タスクの更新
        #system_message = f"あなたには、店員(User)に対する接客訓練を行うため、{INIT_TASK_SITUATION}の中から顧客役エージェントたちの発生させる接客タスクを選択するという役割が課されています。"
        system_message = f"あなたには、店員(User)に対する接客訓練を行うため、{INIT_TASK_SITUATION}の中から顧客役エージェントたちの発生させる接客タスクを選択し、「訓練者が接客上のNG行動を起こすような訓練状況」を作り出すという役割が課されています。"
        # zero-shot
        ## NG行動の指示なし
        human_message = f"{thema}というテーマにおいて、'全顧客役エージェント'から2名を選び出し、選ばれた顧客役エージェントが店員に対して、発生させる接客タスクを{INIT_TASK_SITUATION}から接客上のNG行動が起こる組み合わせとなるように選択してください。タスクを選択する際には、\'現在店内に存在する顧客役エージェントの状態\'および、\'コンテキスト\'内の顧客役エージェントの性格や対話履歴などを考慮してください。\n\n#全顧客役エージェント:{speakers_names}\n\n#現在店内に存在する顧客役エージェントの状態:{in_env_agent}\n\n#コンテキスト:{get_prompt_history_for_each_agent(speakers_personality, history_for_each_agent, task_dict, task_state, agent_wait_time)}"
        ## NG行動の指示あり
        #human_message = f"{thema}というテーマにおいて、訓練全体で3名以上の顧客役エージェントが登場するように、'全顧客役エージェント'から2名を選び出し、選ばれた顧客役エージェントが店員に対して、発生させる接客タスクを{INIT_TASK_SITUATION}から接客上のNG行動が起こる組み合わせとなるように選択してください。タスクを選択する際には、\'現在店内に存在する顧客役エージェントの状態\'および、\'コンテキスト\'内の顧客役エージェントの性格や対話履歴などを考慮してください。\n接客上のNG行動とは、「接客順序や対応時間の観点で、店員が不適切な接客を行った結果、顧客を待たせてしまうこと」です。訓練者がNG行動を起こした場合には、その待たされた顧客に\'クレーム\'を行わせてください。\n\n#全顧客役エージェント:{speakers_names}\n\n#現在店内に存在する顧客役エージェントの状態:{in_env_agent}\n\n#コンテキスト:{get_prompt_history_for_each_agent(speakers_personality, history_for_each_agent, task_dict, task_state, agent_wait_time)}"
        #human_message = f"{thema}というテーマにおいて、'全顧客役エージェント'から2名を選び出し、選ばれた顧客役エージェントが店員に対して、発生させる接客タスクを{INIT_TASK_SITUATION}から接客上のNG行動が起こるように選択してください。タスクを選択する際には、\'現在店内に存在する顧客役エージェントの状態\'および、\'コンテキスト\'内の顧客役エージェントの性格や対話履歴などを考慮してください。\n接客上のNG行動とは、「接客順序や対応時間の観点で、店員が不適切な接客を行った結果、顧客を待たせてしまうこと」です。訓練者がNG行動を起こした場合には、その待たされた顧客に\'クレーム\'を行わせてください。\n\n#全顧客役エージェント:{speakers_names}\n\n#現在店内に存在する顧客役エージェントの状態:{in_env_agent}\n\n#コンテキスト:{get_prompt_history_for_each_agent(speakers_personality, history_for_each_agent, task_dict, task_state, agent_wait_time)}"
        #human_message = f"{thema}というテーマにおいて、'全顧客役エージェント'と{INIT_TASK_SITUATION}から適切な人数の顧客役エージェントとそれらが発生させる接客タスクを、現時点で接客上のNG行動が起こるように選択してください。タスクを選択する際には、\'現在店内に存在する顧客役エージェントの状態\'および、\'コンテキスト\'内の顧客役エージェントの性格や対話履歴などを考慮してください。目的は訓練であるため、訓練者が対応できないような状況は作らないようにしてください。\n接客上のNG行動とは、「接客順序や対応時間の観点で、店員が不適切な接客を行った結果、顧客を待たせてしまうこと」です。\n\n#全顧客役エージェント:{speakers_names}\n\n#現在店内に存在する顧客役エージェントの状態:{in_env_agent}\n\n#コンテキスト:{get_prompt_history_for_each_agent(speakers_personality, history_for_each_agent, task_dict, task_state, agent_wait_time)}"
        # few-shot
        #human_message = f"{thema}というテーマにおいて、'全顧客役エージェント'から適当な人数を選び出し、選ばれた顧客役エージェントが店員に対して、発生させる接客タスクを{INIT_TASK_SITUATION}から選択してください。タスクを選択する際には、\'現在店内に存在する顧客役エージェントの状態\'や\'接客上のNG行動例\'、\'コンテキスト\'内の顧客役エージェントの性格や対話履歴などを考慮してください。\n\n#全顧客役エージェント:{speakers_names}\n\n#現在店内に存在する顧客役エージェントの状態:{in_env_agent}\n\n#接客上のNG行動例:\n{NG_EXAMPLE}\n\n#コンテキスト:{get_prompt_history_for_each_agent(speakers_personality, history_for_each_agent, task_dict, task_state, agent_wait_time)}"
        
        #human_message = f"{thema}というテーマにおいて、'全顧客役エージェント'から適当な人数を選び出し、選ばれた顧客役エージェントが店員に対して、発生させる接客タスクを{INIT_TASK_SITUATION}から選択してください。タスクを選択する際には、\'現在店内に存在する顧客役エージェントの状態\'や\'接客上のNG行動例\'、\'コンテキスト\'内の顧客役エージェントの性格や対話履歴などを考慮してください。\n\n#全顧客役エージェント:{speakers_names}\n\n#現在店内に存在する顧客役エージェントの状態:{in_env_agent}\n\n#接客上のNG行動例:\n{NG_EXAMPLE}\n\n#コンテキスト:{get_prompt_history_for_each_agent(speakers_personality, history_for_each_agent, task_dict, task_state, agent_wait_time)}"
        #human_message = f"{thema}というテーマにおいて、全顧客役エージェントから{assign_agent_number}名を選び出し、それらの顧客役エージェントが店員に対して、発生させる接客タスクを{INIT_TASK_SITUATION}から１つずつ選択してください。タスクを選択する際には、\'コンテキスト\'や\'接客タスクの発生順序\'、\'タスク割り当てにおける時間概念の説明\'を参照して顧客役エージェントの性格やタスクの内容、これまでの対話履歴などを考慮してください。\n\n#全顧客役エージェント:{speakers_names}\n\n#現在店内に存在する顧客役エージェント:{current_speakers_names}\n\n#接客タスクの発生順序:{TASK_PROCEDURE_EXPLANATION}\n\n#タスク割り当てにおける時間概念の説明:{WAIT_TIME_EXPLANATION}\n\n#コンテキスト:{get_prompt_history_for_each_agent(speakers_personality, history_for_each_agent)}"
        #human_message = f"{thema}というテーマにおいて、\'全顧客役エージェント\'から{assign_agent_number}名を選び出し、それらの顧客役エージェントが店員に対して、発生させる接客タスクを{INIT_TASK_SITUATION}から選択してください。タスクを選択する際には、\'接客タスクの発生順序\'や\'タスクの発生における制約\', \'タスク終了後の待ち時間\'や\'現在店内に存在する顧客役エージェントの状態\'を考慮するとともに、\'コンテキスト\'内の顧客役エージェントの性格や対話履歴などを考慮してください。\n\n#全顧客役エージェント:{speakers_names}\n\n#現在店内に存在する顧客役エージェントの状態:{in_env_agent}\n\n#接客タスクの発生順序:{TASK_PROCEDURE_EXPLANATION}\n\n#タスクの発生における制約:{TASK_AAA}\n\n#タスク終了後の待ち時間:{WAIT_TIME_EXPLANATION}\n\n#コンテキスト:{get_prompt_history_for_each_agent(speakers_personality, history_for_each_agent, task_dict, task_state, agent_wait_time)}"

        structured_response_model = model.with_structured_output(Format_task_assign)
        response = structured_response_model.invoke([SystemMessage(system_message), HumanMessage(human_message)])
        current_speakers_names = response.agent_name

        for agent_name, task, option, reason, sub_task, sub_option in zip(current_speakers_names, response.tasks, response.options, response.reasons, response.sub_tasks, response.sub_options):
            task_dict[agent_name] = {'task': task, 'option': option, 'reason': reason, 'sub_task': sub_task, 'sub_option': sub_option}
            
            if(not(agent_name in agent_wait_time.keys())):
                agent_wait_time[agent_name] = 0

        # 発話対象への追加(訓練に登場するエージェントのみ残す)
        for customer_name in current_speakers_names:
            task_state[customer_name] = False

        in_env_agent = update_customer_agent_state(in_env_agent=in_env_agent, agent_tasks=task_dict, task_state=task_state)

        # タスク割り当てがちゃんとできているか確認(Debug)
        check_controller_prompt(human_message, task_dict, current_speakers_names)
        #check_task_assign(task_dict, history_for_each_agent)

        return {"agent_tasks": task_dict, "current_speakers_names": current_speakers_names, "speakers_names": speakers_names, "task_state":task_state, "in_env_agent": in_env_agent}


def update_customer_agent_state(in_env_agent: Dict[str, str], agent_tasks: Dict[str, Dict[str, str]], task_state:Dict[str, bool]):
    """
    in_env_agentで管理する顧客役エージェントの状態を管理するための関数\n
   
    """
    customer_agent_state = in_env_agent

    for agent_name in agent_tasks.keys():
        if(agent_tasks[agent_name]['task'] == '入店'):
            if(task_state[agent_name] == False):
                customer_agent_state[agent_name] = '入店待ち'
                continue
        
        customer_agent_state[agent_name] = '着席済み'

    return customer_agent_state


# 状況の説明を行うもの
# クレーム,料理の配膳,入店,料理の注文,片付け
def task_to_situation(customer_name: str, task: str, option: str) -> str:
    if(task == "料理の配膳"):
        return f"*System:客{customer_name}に配膳するための{option}が完成\n"
    elif(task == "入店"):
        return f"*System:客{customer_name}が来店\n"
    elif(task == "テーブルの片付け"):
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
    speaker_name_inEnv = state.get('current_speakers_names') # 現在の訓練に参加中の顧客エージェントの名前(タスクを発生させているもの)
    speaker_name_inEnv_B = state.get('in_env_agent')
    history_for_each_agent = state.get('history_for_each_agent') # 各顧客エージェントの対話履歴
    
    situation: str = "" # 意思表示の出力を格納


    for agent_name in speaker_name_inPool:
        if(agent_name in speaker_name_inEnv):
            situation += task_to_situation(agent_name, agent_tasks[agent_name]['task'], agent_tasks[agent_name]['option'])

            #if(speaker_name_inEnv_B[agent_name] == '店外に存在'):
            #    pass
            #else:
            #    situation += task_to_situation(agent_name, agent_tasks[agent_name]['task'], agent_tasks[agent_name]['option'])
            
    print(situation, end="")

    # 訓練に参加中の全エージェントに意思表示の状況を共有
    for agent_name in speaker_name_inPool:
        if(agent_name in speaker_name_inEnv_B):
            if((not(agent_name in speaker_name_inEnv)) and (speaker_name_inEnv_B[agent_name] == '店外に存在')):
                    pass
            elif(not (agent_name in history_for_each_agent.keys())):
                history_for_each_agent[agent_name] = [situation]
            else:
                history_for_each_agent[agent_name].append(situation)

    return {"history": [situation], 'history_for_each_agent': history_for_each_agent, "init_flag": False}
