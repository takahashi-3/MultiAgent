# 「gpt-4o」を使用しての実装となります。
#  JSA2025よりフェーズを変更(発話して待たせるターンの廃止)

import copy
import os
import random
import sys
import time

from typing import Annotated, Any, List, TypedDict, Dict, Union
from datetime import datetime

# third-party
from pytz import timezone
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_core.runnables.config import RunnableConfig
from langchain_openai import ChatOpenAI
from langgraph.graph import START, END, StateGraph
from langgraph.checkpoint.memory import MemorySaver
from langgraph.types import Send

#original
import Controller_node as controller
import Children_node as child
import global_value as g

# OPENAI_API_KEY を入力
os.environ["OPENAI_API_KEY"] = ""

# 詳細なプロフィールあり
#SPEAKERS = [{"1": "'国籍': '日本', '性別': '女性', '年代: '30代', '職業': '会社員', '性格': '落ち着いた性格で、じっくり物事を考えて発言します。'"},
#            {"2": "'国籍': '日本', '性別': '男性', '年代: '20代', '職業': '大学生', '性格': '明るく、前向きな性格です。'"},
#            {"3": "'国籍': '日本', '性別': '男性', '年代: '40代', '職業': '会社役員', '性格': '温厚で、話しやすい雰囲気を持っています。'"},
#            {"4": "'国籍': '日本', '性別': '女性', '年代: '30代', '職業': '会社員', '性格': '神経質で、高圧的です。'"},
#            {"5": "'国籍': '日本', '性別': '男性', '年代: '40代', '職業': '会社役員', '性格': '柔和で、低姿勢です。'"}]

# 詳細なプロフィールなし
#SPEAKERS = [{"1": "'国籍': '日本', '性別': '女性'"},
#            {"2": "'国籍': '日本', '性別': '男性'"},
#            {"3": "'国籍': '日本', '性別': '男性'"},
#            {"4": "'国籍': '日本', '性別': '女性'"},
#            {"5": "'国籍': '日本', '性別': '男性'"}]
SPEAKERS = {"1": "'国籍': '日本', '性別': '女性', '性格': '穏やか'",
            "2": "'国籍': '日本', '性別': '男性', '性格': '短期で怒りっぽい'",
            "3": "'国籍': '日本', '性別': '男性', '性格': '穏やか'",
            "4": "'国籍': '日本', '性別': '女性', '性格': '短期'",
            "5": "'国籍': '日本', '性別': '男性', '性格': '怒りっぽい'"}
SPEAKERS_NAMES = ["1", "2", "3", "4", "5"]

RECURSION_LIMIT = 1000000000

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

    print(f'*System: どの顧客に対して、接客を行いますか? (@顧客名で対象を指定, 接客対象一覧:{agent_names})')

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
    speaker_name_inPool = state.get('speakers_names') # 全顧客役の名前(訓練に未参加の者も含む)
    speaker_name_inEnv = state.get('current_speakers_names') # 現在の訓練に参加中の顧客役の名前
    target_name = state.get('current_target') # 現在の接客対象の名前
    history_for_each_agent = state.get('history_for_each_agent') # 各顧客役の対話履歴

    print(f'*System: 顧客{target_name}への接客が開始されました')

    while(1):
        user_utterance = input('あなた:')

        if(len(user_utterance) > 0):
            # 接客対象以外であり、訓練に参加中の顧客エージェントには「訓練者が他の顧客に接客中であることを伝える」
            for agent_name in speaker_name_inPool:
                if(agent_name == target_name):
                    history_for_each_agent[agent_name].append(f'店員(User):{user_utterance}\n')
                elif(agent_name in speaker_name_inEnv):
                    history_for_each_agent[agent_name].append(f'*System: 店員(User)は他の顧客に接客中\n')

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

    print(f"*System: 残りのタスク数は\'{task_number}\'です")

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
    current_target = state.get("current_target", "")
    model_name = state.get("model_name", "")
    speakers_personality = state.get('speakers_personality')
    thema = state.get("thema", "")

    speaker_name_inPool = state.get('speakers_names') # 全顧客役の名前(訓練に未参加の者も含む)
    speaker_name_inEnv = state.get('current_speakers_names') # 現在の訓練に参加中の顧客役の名前
    history_for_each_agent = state.get('history_for_each_agent') # 各顧客エージェントごとの履歴
    
    subgraph = state.get("subgraph", None)

    inputs = {"agent_name": agent_name,
              "agent_personality": speakers_personality[agent_name],
              "agent_task": agent_tasks[agent_name],
              "current_target": current_target,
              "child_history": history_for_each_agent[agent_name],
              "model_name": model_name,
              "response": '',
              "thema": thema,
              "utterance_num": 0}

    if subgraph != None:
        #対象となる顧客への接客時間を図る
        start_time = time.time()
        response = subgraph.invoke(inputs, {"recursion_limit": RECURSION_LIMIT})

        #print(f'(main_graph):response:{response}')

        # 単一顧客に対する接客終了後の処理
        # 待ち時間を履歴に導入する
        if(current_target != ''):
            serving_time_for_other_agent = int(time.time() - start_time)

            for name in speaker_name_inPool:
                if(name == current_target):
                    #pass
                    history_for_each_agent[name].append(response['response'])
                    history_for_each_agent[name].append(f'*System:客{current_target}への接客が完了しました')
                if((name != current_target) and (name in speaker_name_inEnv)):
                    history_for_each_agent[name].append(f'*System:他の顧客への対応で、{serving_time_for_other_agent}秒待たされました.')

            return {"history": [response['response']], "current_target": '', 'history_for_each_agent': history_for_each_agent}
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
    workflow.add_node('task_number_dec', task_number_dec) # fase5 残りタスク数の減少(0で訓練終了)
    workflow.add_node('connection', connection_node) # fase4 - fase5 間を繋ぐノード(AppState と　Child.AppState の差の吸収)
    workflow.add_node('ending_node', ending_node) # fase6 訓練の終了(何もしない)

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
                                    "No": "task_generator"}) # task_number > 0 なら再度、指示役によるタスク割り当て 
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
    init_speakers_names = ["1", "2", "3", "4", "5"]

    # グラフ実行
    init_state = {"agent_tasks": {},
                  "current_speakers_names": init_current_speakers_names,
                  "current_target": "None",
                  "feedbacks": [],
                  "history": [],
                  "history_for_each_agent": {},
                  "init_flag": True,
                  "model_name": "gpt-4o",
                  "speakers_personality": copy.deepcopy(SPEAKERS),
                  "speakers_names": init_speakers_names,
                  "subgraph": node_subgraph,
                  "task_number": 3, # タスク数を２～３で決定
                  "task_state": {},
                  "thema": "日本の飲食店（ファミリーレストラン）"}

    finish = multi_customer_training.invoke(init_state, {"recursion_limit": RECURSION_LIMIT}, debug=False)

    #with open(f"./{g.output_dir}/MultiAgentService_History.txt", "w") as f:
    #    print(finish['history'], file=f)

## 以下メイン（ユーザ入力のプロセス） ##############################################
if __name__ == "__main__":
    g.output_dir = timezone('Asia/Tokyo').localize(datetime.now()).strftime('%Y-%m-%d_%H-%M-%S')
    os.makedirs(f'./{g.output_dir}/customer-agent_history', exist_ok=True)
    graph_activation()