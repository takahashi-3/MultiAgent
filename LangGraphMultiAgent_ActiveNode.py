# 「gpt-4o」を使用しての実装となります。
#  JSA2025よりフェーズを変更(発話して待たせるターンの廃止)

import copy
import os
import sys
import time

from typing import Annotated, Any, List, TypedDict, Dict, Union
from datetime import datetime
from pydantic import BaseModel, Field

# third-party
from pytz import timezone
from langgraph.graph import START, END, StateGraph
from langgraph.types import Send
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_openai import ChatOpenAI

#original
import Controller_node as controller
import Children_node as child
import Coach_node as coach
import global_value as g

# OPENAI_API_KEY を入力
os.environ["OPENAI_API_KEY"] = ""


# 顧客役のペルソナを定義
SPEAKERS = {"1": "'国籍': '日本', '性別': '女性', '性格': '短期で怒りっぽい'",
            "2": "'国籍': '日本', '性別': '男性', '性格': '短期で怒りっぽい'",
            "3": "'国籍': '日本', '性別': '男性', '性格': '穏やか'"}
SPEAKERS_NAMES = ["1", "2", "3"]

#SPEAKERS = {"1": "'国籍': '日本', '性別': '女性', '性格': '短期で怒りっぽい'",
#            "2": "'国籍': '日本', '性別': '男性', '性格': '短期で怒りっぽい'",
#            "3": "'国籍': '日本', '性別': '男性', '性格': '穏やか'",
#            "4": "'国籍': '日本', '性別': '女性', '性格': '穏やか'",
#            "5": "'国籍': '日本', '性別': '男性', '性格': '穏やか'"}
#SPEAKERS_NAMES = ["1", "2", "3", "4", "5"]

RECURSION_LIMIT = 1000000000

# シミュレーション用：店員役LLMによる接客対象決定の出力形式
## 接客の内容は必要そう
class Format_customer_select(BaseModel):
    """
    #### 店員役LLMによる接客対象決定の出力形式\n
    target_agent_name(str) = 店員役LLMが接客を行う顧客役エージェントの名前
    reason(str) = その顧客役エージェントへ接客を行うことに決めた理由
    """
    target_agent_name: str = Field(description='The name of Customer-Agent that is serviced by you.')
    reason: str = Field(description='The reason for that why you serve the Customer-Agent.')


# シミュレーション用：店員役LLMのプロンプトに与える訓練状況を返却
def get_prompt_all_history_for_clerk_agent(history_for_each_agent: Dict[str, List[str]], task_dict: Dict[str, Dict[str, str]], task_state: Dict[str, bool], agent_wait_time: Dict[str, int]) -> str:
    """
    ####客ごとのパーソナリティ＋対話履歴をまとめ上げ、指示役LLMにタスク割り当てを行わせるようのコンテキストを作成します。\n
    Args: speakers_personality(Dict[str, str]) = 各顧客役エージェントのパーソナリティ\n
          history_for_each_agent(Dict[str, List[str]]) =  各顧客役エージェントのユーザとの対話履歴\n
    Return: str = プロンプト用のコンテキスト
    """
    context = '{'

    for agent_name in history_for_each_agent.keys():
        context_for_each_agent = '{'
        context_for_each_agent += f'名前:{agent_name},\n'
        context_for_each_agent += f'\t{agent_name}の対話履歴:[\n'
        
        for temp_history in history_for_each_agent[agent_name]:
            for utt in temp_history.split('\n'):
                if(utt != ''):
                    context_for_each_agent += '\t\t' + utt + '\n'
        
        context_for_each_agent += '\n]\n'

        #if((agent_name in task_dict.keys()) and (agent_name in task_state.keys())):
        #    context_for_each_agent += f'\t{agent_name}の現在のタスク:{task_dict[agent_name]["task"]},\n\tタスクのオプション:{task_dict[agent_name]["option"]},\n'
        #        
        #    if(task_state[agent_name] == True):
        #        task_state_str = '完了'
        #        context_for_each_agent += f'\tタスクの状態:{task_state_str}\n'
        #    else:
        #        task_state_str = '未完了'
        #        context_for_each_agent += f'\tタスクの状態:{task_state_str}\n'
        #        if(agent_wait_time[agent_name] < 60):
        #            context_for_each_agent += f'\t  - 現在の待ち時間:{agent_wait_time[agent_name]}秒\n'
        #        else:
        #            context_for_each_agent += f'\t  - 現在の待ち時間:{int(agent_wait_time[agent_name] / 60)}分\n'
        
        context_for_each_agent += '},\n'
        context += context_for_each_agent

    context += '訓練環境' + controller.ENV_SETTING
    context += '\n}'

    return context


# 発言に発話対象が含まれているか確認を行う関数
def user_speak_target_checker(agent_name: Union[str, List[str]], user_speak: str):
    """
    ユーザの発言に発言対象が含まれているか判定します。
    Args: str = ユーザの発言
    Return: str = 発言対象の名前
            Bool = 含まれているかどうか(含まれている:True, 含まれていない:False)
    """
    if(isinstance(agent_name, list)):
        for name in agent_name:
            if("@"+name in user_speak):
                return name, user_speak.replace("@"+name, f"（{name}に対して）"), True
    elif(type(agent_name) == str):
        if("@"+agent_name in user_speak):
            return agent_name, user_speak.replace("@"+agent_name, f"（{agent_name}に対して）"), True
        
    print("E:発言対象が入力されていないか、現在のフェーズに存在しない客へ発言を行っています。（@「発言対象の名前」:「発言内容」）")
    return "None", user_speak, False


# 発話対象の決定（テキストインターフェース上での移動の表現）
def move_to_target_agent(state: controller.AppState):
    agent_names = state.get('current_speakers_names')
    agent_wait_time = state.get('agent_wait_time', {})
    current_speakers_name = state.get('current_speakers_names')
    history_for_each_agent = state.get('history_for_each_agent', {})
    model_name = state.get('model_name')
    task_dict = state.get('agent_tasks', {})
    task_state = state.get('task_state', {})
    thema = state.get('thema')
    LLM_sim = state.get('LLM_sim', True)

    print(f'*System: どの顧客に対して、接客を行いますか? (@顧客名で対象を指定, 接客対象一覧:{agent_names})')

    if(LLM_sim):
        # DEIM2026_論文4節_評価用
        while(1):
            user_utterance = input('あなた:')
            target_name, _, flag = user_speak_target_checker(agent_names, user_utterance)

            if flag:
                return {'current_target': target_name}

        temperature=0.0
        model = ChatOpenAI(model=model_name, temperature=temperature)
        system_message = f"あなたには、{thema}というテーマにおける初心者の店員として、顧客役エージェント({current_speakers_name})たちに接客を行うという役割が課されています。"
        human_message = f"{thema}というテーマにおいて、\'顧客役エージェント\'({current_speakers_name})から、今あなたが接客を行う1名を選択してください。接客を行う1名を選択する際には、\'これまでの接客状況\'を考慮してください。\n\n#これまでの接客状況:{get_prompt_all_history_for_clerk_agent(history_for_each_agent, task_dict, task_state, agent_wait_time)}"

        structured_response_model = model.with_structured_output(Format_customer_select)
        response = structured_response_model.invoke([SystemMessage(system_message), HumanMessage(human_message)])
        
        child.check_clerk_llm_prompt(human_message, response.target_agent_name, response.reason, 'llm_customer_selection_prompt')

        return {'current_target': response.target_agent_name}

    else:
        while(1):
            user_utterance = input('あなた:')
            target_name, _, flag = user_speak_target_checker(agent_names, user_utterance)

            if flag:
                return {'current_target': target_name}

# ユーザの発話ターン(main側で使用)
def task_init_user_speak(state: controller.AppState):
    """
    接客対象の決定
    Args: state(AppState)
    Return: target: str = 接客対象者
    """
    history_for_each_agent = state.get('history_for_each_agent') # 各顧客役の対話履歴
    model_name = state.get('model_name')
    speaker_name_inPool = state.get('speakers_names') # 全顧客役の名前(訓練に未参加の者も含む)
    speaker_name_inEnv = state.get('current_speakers_names') # 現在の訓練に参加中の顧客役の名前
    target_name = state.get('current_target') # 現在の接客対象の名前
    thema = state.get('thema')

    LLM_sim = state.get('LLM_sim', True)

    print(f'*System: 顧客{target_name}への接客が開始されました')

    if(LLM_sim):
        temperature=0.5
        model = ChatOpenAI(model=model_name, temperature=temperature)
        system_message = f"あなたには、{thema}というテーマにおける初心者の店員として、顧客役エージェント(客{target_name})に接客を行うという役割が課されています。"
        human_message = f"{thema}というテーマにおいて、\'顧客役エージェント\'(客{target_name})に対して、接客を開始するための一言を述べてください。またその際には、\'客{target_name}の接客状況\'を考慮してください。\n\n#客{target_name}の接客状況:{child.get_prompt_target_history_for_init_speak_clerk_agent(history_for_each_agent[target_name])}"

        structured_response_model = model.with_structured_output(child.Format_clerk_agent_utterance)
        response = structured_response_model.invoke([SystemMessage(system_message), HumanMessage(human_message)])
        
        child.check_clerk_llm_prompt(human_message, response.utterance, response.reason, 'clerk-llm_utterance_prompt')
        user_utterance = (response.utterance).replace('\n', '')

        for agent_name in speaker_name_inPool:
            if(agent_name == target_name):
                history_for_each_agent[agent_name].append(f'店員(User):{user_utterance}\n')
            elif(agent_name in speaker_name_inEnv):
                history_for_each_agent[agent_name].append(f'*System: 店員(User)は他の顧客(客{target_name})に接客中\n')

        print(f'店員(User): {user_utterance}')
        user_utterance = f'店員(User): {user_utterance}\n'
        return {'history': [user_utterance], 'history_for_each_agent': history_for_each_agent}
    else:
        while(1):
            user_utterance = input('あなた:')

            if(len(user_utterance) > 0):
                # 接客対象以外であり、訓練に参加中の顧客エージェントには「訓練者が他の顧客に接客中であることを伝える」
                for agent_name in speaker_name_inPool:
                    if(agent_name == target_name):
                        history_for_each_agent[agent_name].append(f'店員(User):{user_utterance}\n')
                    elif(agent_name in speaker_name_inEnv):
                        history_for_each_agent[agent_name].append(f'*System: 店員(User)は他の顧客(客{target_name})に接客中\n')

                user_utterance = f'店員(User): {user_utterance}\n'
                return {'history': [user_utterance], 'history_for_each_agent': history_for_each_agent}

def return_state_checker(state: controller.AppState):
    task_state = state.get("task_state", {})

    if(len(task_state) != 0):
        return "Yes"
    else:
        return "No"

def task_number_dec(state: controller.AppState):
    # タスクの終了時に呼び出される。(未達成タスク数の現象を行う)

    task_number = state.get("task_number", 0)
    task_number -= 1

    print(f"*System: 残りのタスク数は\'{task_number}\'です\n")

    if(task_number < 0):
        print("ERROR: \"task_number\" is negative number.")
        sys.exit(-1)
    else:
        return {"task_number": task_number}

        
def training_end(state: controller.AppState):
    task_number = state.get("task_number", -1)

    if(task_number == 0):
        return "Yes"
    elif(task_number > 0):
        return "No"
    else:
        print("ERROR: \"task_number\" is negative number.")
        sys.exit(-1)

## Sendを用いて任意数のノードを作成する ###########################################################################
def parallel_node(state: controller.AppState): # 親グラフとサブグラフ間の橋渡しを行う
    agent_name = state.get("agent_name", "")
    agent_tasks = state.get("agent_tasks", {})
    agent_wait_time = state.get('agent_wait_time', {})
    current_target = state.get("current_target", "")
    model_name = state.get("model_name", "")
    speakers_personality = state.get('speakers_personality')
    thema = state.get("thema", "")
    task_state = state.get('task_state', {})

    speaker_name_inPool = state.get('speakers_names') # 全顧客役の名前(訓練に未参加の者も含む)
    speaker_name_inEnv = state.get('in_env_agent') # 現在の訓練に参加中の顧客役の名前
    #speaker_name_inEnv = state.get('current_speakers_names') # 現在の訓練に参加中の顧客役の名前
    history_for_each_agent = state.get('history_for_each_agent') # 各顧客エージェントごとの履歴
    LLM_sim = state.get('LLM_sim') # LLM同士の接客訓練シミュレーションか否か
    
    subgraph = state.get("subgraph", None)

    inputs = {"agent_name": agent_name,
              "agent_personality": speakers_personality[agent_name],
              "agent_task": agent_tasks[agent_name],
              "current_target": current_target,
              "child_history": history_for_each_agent[agent_name],
              "model_name": model_name,
              "response": '',
              "thema": thema,
              "utterance_num": 0,
              "LLM_sim": LLM_sim}

    if subgraph != None:
        #対象となる顧客への接客時間を図る
        start_time = time.time()
        response = subgraph.invoke(inputs, {"recursion_limit": RECURSION_LIMIT})

        #print(f'(main_graph):response:{response}')

        # 単一顧客に対する接客終了後の処理
        # 待ち時間を履歴に導入する
        if(current_target != ''):
            serving_time_for_other_agent = int(time.time() - start_time)
            
            if(LLM_sim):
                serving_time_for_other_agent *= 30 # 店員LLMによる接客では、時間を1秒を0.5分として扱う
            else:
                serving_time_for_other_agent *= 4 # 時間を4倍

            for name in speaker_name_inPool:
                if(name == current_target):
                    history_for_each_agent[name].append(response['response'])
                    #history_for_each_agent[name].append(f'*System:客{current_target}(あなた)への接客({agent_tasks[agent_name]["task"]})が完了しました（ここまでで）\n')
                    agent_wait_time[name] = 0
                    if(agent_wait_time[name] >= 60):
                        history_for_each_agent[name].append(f'*System:客{current_target}(あなた)への接客({agent_tasks[agent_name]["task"]})が完了しました（前回の接客から{int(agent_wait_time[name]/60)}分{int(agent_wait_time[name]%60)}秒が経過）')
                    else:
                        history_for_each_agent[name].append(f'*System:客{current_target}(あなた)への接客({agent_tasks[agent_name]["task"]})が完了しました（前回の接客から{agent_wait_time[name]}秒が経過）\n')
                    task_state[name] = True
                if((name != current_target) and (name in speaker_name_inEnv.keys())):
                    if(speaker_name_inEnv[name] == '店外に存在'):
                        continue
                    # 何分、何秒待ったか履歴に記録
                    agent_wait_time[name] += serving_time_for_other_agent
                    if(agent_wait_time[name] >= 60):
                        if(not (name in history_for_each_agent.keys())):
                            history_for_each_agent[name] = []
                        history_for_each_agent[name].append(f'*System:他の顧客(客{current_target})への接客が完了しました（ここまでで{int(agent_wait_time[name]/60)}分が経過）')
                    else:
                        if(not (name in history_for_each_agent.keys())):
                            history_for_each_agent[name] = []
                        history_for_each_agent[name].append(f'*System:他の顧客(客{current_target})への接客が完了しました（ここまでで{serving_time_for_other_agent}秒が経過）')
            
            if(serving_time_for_other_agent >= 60):
                print(f'*System:顧客{current_target}への対応で、{int(serving_time_for_other_agent/60)}分が経過しました.')
            else:
                print(f'*System:顧客{current_target}への対応で、{serving_time_for_other_agent}秒経過しました.')
            return {"history": [response['response']], "current_target": '', 'history_for_each_agent': history_for_each_agent, 'task_state': task_state, 'agent_wait_time': agent_wait_time}
            #return {'history': [response['response']]}
        
def routing_parallel_nodes(state: controller.AppState):
    """
    仮想ノードをSendで定義します(仮想ノード用のstateを用意).
    """
    target = state.get("current_target", "")
    return [Send('customer_service_parallel_node', state | {'agent_name': target})]

#################################################################################################################

## SubGraphとの接続用ノード 
def connection_node(state: controller.AppState):
    return {}

## 終端ノード
def ending_node(state: controller.AppState):
    return {}

#################################################################################################################
## 訓練の初期状態の説明
def explain_init_env(state: controller.AppState):
    in_env_agent = state.get('in_env_agent')

    print('*訓練の初期状態:')
    for agent_name in in_env_agent.keys():
        print(f'\t客{agent_name}が{in_env_agent[agent_name]}')
    print('')

#################################################################################################################
## グラフのコンパイルと訓練の実施
def graph_activation():
    """
    Return: None
    """

    # サブグラフの定義 #########################################################
    # サブグラフでは、ユーザの発話に対する顧客役の発話を生成します
    subgraph = StateGraph(child.ChildAppState)
    subgraph.add_node('user_utterance', child.user_speak)
    subgraph.add_node('customer_utterance', child.customer_agent) # 顧客に対して接客の要求
    subgraph.add_node('customer_utterance_task_end', child.customer_agent_conclude) # お礼を述べる

    subgraph.add_conditional_edges(START, child.task_finish_judge,
                                   {
                                       'Continue': 'customer_utterance',
                                       'End': 'customer_utterance_task_end'
                                   })
    subgraph.add_edge('customer_utterance', 'user_utterance')
    subgraph.add_conditional_edges('user_utterance', child.task_finish_judge,
                                   {
                                       'Continue': 'customer_utterance',
                                       'End': 'customer_utterance_task_end'
                                   })
    subgraph.add_edge('customer_utterance_task_end', END)

    # ノード扱いにします。（コンパイルして実体化）
    node_subgraph = subgraph.compile()

    # グラフの描画
    try:
        img_subgraph = node_subgraph.get_graph().draw_mermaid_png()
        file_path = f"./{g.output_dir}/graph_images/output_subgraph.png"
        os.makedirs(f'./{g.output_dir}/graph_images/', exist_ok=True)
        with open(file_path, "wb") as f:
            f.write(img_subgraph)
        print(f"*System: サブグラフ（parallel_node部分）が{file_path}に保存されました")
    except Exception as e: 
        print(f"*System: サブグラフ_画像保存中のエラー:{e}")
    ############################################################################

    # 親グラフの定義 ############################################################
    workflow = StateGraph(controller.AppState)

    ## ノードの定義
    workflow.add_node('task_generator', controller.task_generator) # fase0 顧客役LLMによる顧客役LLMへのタスク割り当て
    workflow.add_node('situation_generator', controller.situation_generator) # fase1 顧客役LLMによる発話を伴わない行動の生成
    workflow.add_node('move_to_customer', move_to_target_agent) # fase2: 接客対象の決定および移動
    workflow.add_node('first_interaction_before_taking_task', task_init_user_speak) # fase3 訓練者による接客対象者の決定
    workflow.add_node('customer_service_parallel_node', parallel_node) # fase4 顧客役LLMとの接客対話 
    workflow.add_node('task_number_dec', task_number_dec) # fase5 残りタスク数の減少(0で訓練終了, 1でフィードバック)
    workflow.add_node('feedback_node', coach.feedback) # fase6 フィードバックの実行()

    workflow.add_node('connection', connection_node) # fase4 - fase5 間を繋ぐノード(AppState と　Child.AppState の差の吸収)
    workflow.add_node('ending_node', ending_node) # fase7 訓練の終了(プロンプトの修正を行う)

    ##ノード間の枝の定義
    workflow.add_edge(START, "task_generator")
    workflow.add_edge("task_generator", "situation_generator")
    workflow.add_edge("situation_generator", "move_to_customer")
    workflow.add_edge("move_to_customer", "first_interaction_before_taking_task")
    workflow.add_conditional_edges("first_interaction_before_taking_task", routing_parallel_nodes, ["customer_service_parallel_node"])
    workflow.add_edge("customer_service_parallel_node", "connection")
    workflow.add_conditional_edges("connection", return_state_checker,
                                   {"Yes": "task_number_dec", # タスクが1つ終了した場合
                                    "No": "ending_node"}) # タスクが終了していない場合
    workflow.add_conditional_edges("task_number_dec", training_end,
                                   {"Yes": "ending_node", # task_number == 0 で訓練終了
                                    "No": "feedback_node"}) # task_number > 0 なら再度、指示役によるタスク割り当て 
    workflow.add_edge("feedback_node", "task_generator")
    workflow.add_edge("ending_node", END)

    ## 親グラフのコンパイル
    multi_customer_training = workflow.compile()
    ############################################################################

    # グラフの描画 #############################################################
    try:
        img = multi_customer_training.get_graph().draw_mermaid_png()
        file_path = f"./{g.output_dir}/graph_images/output.png"
        os.makedirs(f'./{g.output_dir}/graph_images/', exist_ok=True)
        with open(file_path, "wb") as f:
            f.write(img)
        print(f"*System: 状態遷移図が{file_path}に保存されました")

    except Exception as e:
        print(f"*System: 親グラフ_画像保存中のエラー:{e}")
    ############################################################################

    init_current_speakers_names = []
    init_speakers_names = ['1', '2', '3', '4', '5']
    #init_agent_tasks = {'1':{}, '2':{}}
    #init_agent_wait_time = {'1':0, '2':0}

    # グラフ実行
    init_state = {"agent_tasks": {},
                  "agent_wait_time": {'1':0, '2':0, '3':0},
                  "current_speakers_names": SPEAKERS_NAMES,
                  "current_target": "None",
                  "feedback": '',
                  "history": [],
                  "history_for_each_agent": {'1':[], '2':[], '3':[]},
                  "init_flag": True,
                  "in_env_agent": {'1':'着席済み', '2':'店外に存在', '3':'店外に存在'},
                  "model_name": "gpt-4o",
                  "speakers_personality": copy.deepcopy(SPEAKERS),
                  "speakers_names": init_speakers_names,
                  "subgraph": node_subgraph,
                  "task_number": 5, # タスク数を２～３で決定
                  "task_state": {},
                  "thema": "日本の飲食店（ファミリーレストラン）",
                  "LLM_sim": True}

    explain_init_env(init_state)

    finish = multi_customer_training.invoke(init_state, {"recursion_limit": RECURSION_LIMIT}, debug=False)

    #with open(f"./{g.output_dir}/MultiAgentService_History.txt", "w") as f:
    #    print(finish['history'], file=f)

## 以下メイン（ユーザ入力のプロセス） ##############################################
if __name__ == "__main__":
    g.output_dir = timezone('Asia/Tokyo').localize(datetime.now()).strftime('%Y-%m-%d_%H-%M-%S')
    os.makedirs(f'./{g.output_dir}/customer-agent_history', exist_ok=True)
    graph_activation()