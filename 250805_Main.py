from multiprocessing import Process, Manager
import operator
import os
import random
import sys
import time
from typing import Annotated, List, TypedDict, Dict, Any

from langchain_core.messages import HumanMessage, SystemMessage
from langchain_core.runnables.config import RunnableConfig
from langchain_groq import ChatGroq
from langgraph.graph import START, END, StateGraph
from langgraph.checkpoint.memory import MemorySaver
from langgraph.types import Send

import Children_node as child

os.environ["GROQ_API_KEY"] = "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"

SPEAKERS = [{"1": "'国籍': '日本', '性別': '女性', '年代: '30代', '職業': '会社員', '性格': '落ち着いた性格で、じっくり物事を考えて発言します。'"},
            {"2": "'国籍': '日本', '性別': '男性', '年代: '20代', '職業': '大学生', '性格': '明るく、前向きな性格です。'"},
            {"3": "'国籍': '日本', '性別': '男性', '年代: '40代', '職業': '会社役員', '性格': '温厚で、話しやすい雰囲気を持っています。'"},
            {"4": "'国籍': '日本', '性別': '女性', '年代: '30代', '職業': '会社員', '性格': '神経質で、高圧的です。'"},
            {"5": "'国籍': '日本', '性別': '男性', '年代: '40代', '職業': '会社役員', '性格': '柔和で、低姿勢です。'"}]
SPEAKERS_NAMES = ["1", "2", "3", "4", "5"]

# タスクの順序を設定する必要がある
TASK_DICT = {"入店": "店員に声を掛け、テーブルもしくはカウンター席への案内を待つ。\n店員によって、席への案内が行われた場合にこの接客タスクは終了する。", # どのテーブルに誰が座っているのか決定する必要がある。（初期状態の設定）
             "商品の注文": "店員に声を掛け、その後具体的な商品の注文を行う。メニューとしては「パンケーキ」「ハンバーガーセット」「バゲットセット」の３つがあります。\n店員によって、注文の受領と確認が行われ、間違いがない場合にこの接客タスクは終了する。",  # 商品としてどれがあるということを指定する必要がある
             "料理の配膳を待つ": "店員に声を掛け、自分のオーダーした商品が後どのくらいで配膳されるのか尋ねる。\n店員からの回答が得られた場合にこの接客タスクは終了する。", # 自分がどの商品をオーダーしているのか記憶する必要がある
             "片付けの要求": "店員に声を掛け、テーブル上の食器について片付けを要求する。\n店員によって、片付けの了承が得られた場合にこの接客タスクは終了する。", # 自分がどの商品をオーダーしているのか記憶する必要がある
             "クレーム": "店員に対して接客の不手際を述べる。\nクレーム内容に関して、店員からの回答が得られた場合にこの接客タスクは終了する。"} 
PRIORITIES = "\n#接客タスクの優先順位\n1:クレーム\n2:商品の注文, 入店\n3:料理の配膳\n4:片付け\n"
RECURSION_LIMIT = 1000000000

# Graph全体のstate
class AppState(TypedDict):
    """
    agent_tasks(Dict[str, str]) = エージェントの名前をキー, タスクをコンテンツとする辞書
    current_speakers_names(List[str]) = 現在のフェーズにおいて行動を起こしているエージェントのリスト
    current_target(str) = 現在接客の対象となっているエージェント
    feedbacks(List[str]) = フィードバックの内容
    history(List[str]) = 会話の履歴
    init_flag(bool) = task_generatorで初期タスクの生成を行うか(True),タスクの更新を行うか(False)の判断をするためのflag
    model_name(str) = 推論を行わせるモデル名(利用するLLMのAPIに基づいた名前を設定してください. ここではGroq)
    speakers(List[Dict[str]]) = 客役エージェントの名前をキー, パーソナリティをコンテンツとして持つ辞書
    speakers_names(List[str]) = 訓練に参加しているエージェントの名前(客役のプールとして機能する)
    subgraph(Any) = サブグラフのインスタンス?
    task_number(int) = 訓練全体で処理すべきタスクの数（この数のタスクを完了したら訓練終了）
    task_state(Dict[str, bool]) = 現在の２つのタスクが終了しているかどうか(key:エージェント名, content:タスクの状態(True:完了 , False:未完了))
    thema(str) = 会話のテーマ
    fase_number(int) = ユーザとの接客フェーズ（パラレルノードの選択用）
    """
    agent_tasks:  Dict[str, str]# {"name": "task", "name": "task", ...}
    current_speakers_names: List[str]
    current_target: str
    feedbacks: Annotated[List[str], operator.add]
    history: Annotated[List[str], operator.add]
    init_flag: bool
    model_name: str
    speakers: List[Dict[str, str]]
    speakers_names: List[str]
    subgraph: Any # 型がわからないのでひとまず任意型です
    task_number: int
    task_state: Dict[str, bool]
    thema: str
    fase_number: int

def task_generator(state: AppState):
    """
    (初回)客ごとの接客タスクを生成します。
    Args: state(AppState)
    Return: Dict[str] = 客ごとのタスク
    """
    init_flag = state.get("init_flag", False)
    model_name = state.get("model_name")
    current_speakers_names = state.get("current_speakers_names", [])
    speakers_names = state.get("speakers_names", [])
    task_state = {}

    candidates_names = SPEAKERS_NAMES
    speakers: List[Dict[str, str]] = []

    model = ChatGroq(model=model_name)

    # 行動を起こすユーザの決定(現状二人同時に行動を起こす場合のみ想定)
    if(len(current_speakers_names) == 0): # 初期状態で誰が行動を起こすか決定されていない場合
        customer_number = 2
        current_speakers_names = random.sample(candidates_names, customer_number)
    elif((not init_flag) and (len(current_speakers_names) == 1) and (len(speakers_names) > 0)):
        current_speakers_names.append((random.sample(speakers_names, 1))[0])

    # speaker_namesからcurrent_speakers_names内の人物を除外する
    for name in current_speakers_names:
        if(name in speakers_names):
            speakers_names.remove(name)
        
    for customer_name in current_speakers_names:
        speakers.append({customer_name: SPEAKERS[SPEAKERS_NAMES.index(customer_name)]})
        task_state[customer_name] = False

    if(init_flag):
        thema = state.get("thema")
        task_dict = {}

        for i in current_speakers_names:
            system_message = f"あなたには日本語で、{TASK_DICT.keys()}の中から{current_speakers_names}たちの起こしえる接客タスクを優先度が異なるように選択するという役目が課されています。"
            human_message = f"{thema}というテーマにおいて、客である{i}が店員に対して、起こしえる接客タスクを{TASK_DICT.keys()}から１つ選択してください。また「接客タスクの優先順位」や「全客の接客タスク」を参考に、各客の選択した接客タスクの優先順位が異なるように選択してください。ローマ字を使ってはいけません。\n理由づけは行わずに選択した接客タスクのみを出力してください\n\n#全客の接客タスク:{[task_dict[name] for name in task_dict.keys()]}\n{PRIORITIES}\n{i}: "
            response = model.invoke([SystemMessage(system_message), HumanMessage(human_message)])
            task_dict[i] = response.content

        return {"agent_tasks": task_dict, "current_speakers_names": current_speakers_names, "speakers": speakers, "speakers_names": speakers_names, "task_state":task_state}
    else:
        history = state.get("history", [])
        task_dict = {}
        thema = state.get("thema", "")

        # タスクの更新
        for i in current_speakers_names:
            system_message = f"あなたには日本語で、{TASK_DICT.keys()}の中から{current_speakers_names}たちの起こしえる接客タスクを選択するという役目が課されています。"
            human_message = f"{thema}というテーマにおけるこれまでの履歴を確認したうえで、客である{i}が店員に対して、起こしえる接客タスクを{TASK_DICT.keys()}から１つ選択してください。また「接客タスクの優先順位」や「全客の接客タスク」を参考に、各客の選択した接客タスクの優先順位が異なるように選択してください。既に接客タスクが決定されていた場合、続けてそれを選択してください。ローマ字を使ってはいけません。\n理由づけは行わずに選択した接客タスクのみを出力してください。\n\n#全ての客のタスク:{[task_dict[name] for name in task_dict.keys()]}\n{PRIORITIES}\n#会話の履歴\n"
            human_message = human_message + "\n".join(history) + f"\n{i}: "
            response = model.invoke([SystemMessage(system_message), HumanMessage(human_message)])
            task_dict[i] = response.content

        return {"agent_tasks": task_dict, "current_speakers_names": current_speakers_names, "speakers": speakers, "speakers_names": speakers_names, "task_state":task_state}

def situation_generator(state: AppState):
    """
    二人の客役が同時に行動を起こす状況を生成します。
    Args: state(AppState)
    Return: Dict[str] = 履歴の作成
    """
    init_flag = state.get("init_flag", False)
    speakers = state.get("speakers", [])
    speakers_names = state.get("current_speakers_names", [])
    thema = state.get("thema")
    agent_tasks = state.get("agent_tasks")
    first_situation = ""
    situation_history = ""
    history = state.get("history", [])
    model_name = state.get("model_name")

    model = ChatGroq(model=model_name)

    if(init_flag):
        for i in speakers_names:
            system_message = f"あなたの名前は{i}で、{thema}というテーマにおける客としての役割を持っています。また、{speakers[speakers_names.index(i)]}というパーソナリティをもっています。"
            human_message = f"あなたが選択した接客タスク（{agent_tasks[i]}）の内容をもとに、「すみません」や「ちょっといいですか？」のような店員に対する呼びかけを一言で行ってください。また、この段階では具体的に何をして欲しいのか伝える必要はありません。ローマ字を使ってはいけません。\n#各接客タスクの詳細{TASK_DICT}\n{i}: "
            response = model.invoke([SystemMessage(system_message), HumanMessage(human_message)])
            first_situation += f"客{i}: {response.content}({agent_tasks[i]})\n"
            situation_history += f"{i}: {response.content}\n"
    else:
        for i in speakers_names:
            system_message = f"あなたの名前は{i}で、{thema}というテーマにおける客としての役割を持っています。また、{speakers[speakers_names.index(i)]}というパーソナリティをもっています。"
            human_message = f"あなたが選択した接客タスク（{agent_tasks[i]}）の内容とこれまでの履歴をもとに、「すみません」や「ちょっといいですか？」のような店員に対する呼びかけを一言で行ってください。また、この段階では具体的に何をして欲しいのか伝える必要はありません。ローマ字を使ってはいけません。\n#各接客タスクの詳細{TASK_DICT}\n#会話の履歴\n "
            human_message = human_message + "\n".join(history) + f"\n{i}: "
            response = model.invoke([SystemMessage(system_message), HumanMessage(human_message)])
            first_situation += f"客{i}: {response.content}({agent_tasks[i]})\n"
            situation_history += f"{i}: {response.content}\n"

    print(first_situation, end="")

    return {"history": [situation_history], "init_flag": False}

def utterance_target_checker(state: AppState, user_speak: str):
    """
    ユーザの発言に発言対象が含まれているか判定します。
    Args: str = ユーザの発言
    Return: str = 発言対象の名前
            Bool = 含まれているかどうか(含まれている:True, 含まれていない:False)
    """
    speakers_names = state.get("current_speakers_names", [])

    for i in speakers_names:
        if("@"+i+":" in user_speak):
            return i, user_speak.replace("@"+i+":", ""), True
    
    print("E:発言対象が入力されていないか、現在のフェーズに存在しない客へ発言を行っています。（@「発言対象の名前」:「発言内容」）")
    print(f"発言対象一覧{speakers_names}")
    return "None", user_speak, False

def user_speak_priority_check(state: AppState):
    """
    Args: state(AppState)
    Return: Dict[str] = 生成した発言
    """
    history = state.get("history", [])
    speakers = state.get("speakers", {})
    speakers_names = state.get("current_speakers_names", [])
    model_name = state.get("model_name", "")
    
    current_history = ""
    target_list = speakers_names.copy()
    model = ChatGroq(model=model_name)

    if(len(speakers_names) == 1):
        speakers_names.remove(target_list[0])
        return {"current_speakers_names": speakers_names, "current_target": target_list[0]}

    while(1):
        user_utterance = input("＊対応の優先順位が低いお客様にお待たせすることを伝えてください。\nあなた:")    
        target, s, flag = utterance_target_checker(state, user_utterance)

        if flag:
            current_history += f"店員: {s}\n"

            system_message = f"あなたの名前は{target}で、{speakers[speakers_names.index(target)]}というパーソナリティをもっています。"
            human_message = f"店員は他の客に対応するため、あなたにひとまず待ってもらうという選択をとりました。これまでの履歴を参考に、店員に対して「待たされることに納得した」ということを可能な限り短い言葉で伝えてください。また、ローマ字を使ってはいけません。\n#会話の履歴\n "
            human_message = human_message + "\n".join(history) + "\n" + current_history + f"\n{target}: "
            response = model.invoke([SystemMessage(system_message), HumanMessage(human_message)])

            current_history += f"客{target}: {response.content}\n"
            print(f"客{target}:{response.content}")
            target_list.remove(target)

            if(len(target_list) == 1):
                speakers_names.remove(target_list[0])
                print(f"＊客{target_list[0]}へ接客を行いましょう。")
                return {"history": [current_history], "current_speakers_names": speakers_names, "current_target": target_list[0]}
            else:
                print("他のお客様にもお待たせする事を伝えましょう。")

# subgraph側で用いられる
def user_speak_target_checker(agent_name: str, user_speak: str):
    """
    ユーザの発言に発言対象が含まれているか判定します。
    Args: str = ユーザの発言
    Return: str = 発言対象の名前
            Bool = 含まれているかどうか(含まれている:True, 含まれていない:False)
    """
    if("@"+agent_name+":" in user_speak):
        return agent_name, user_speak.replace("@"+agent_name+":", ""), True
    
    print("E:発言対象が入力されていないか、現在のフェーズに存在しない客へ発言を行っています。（@「発言対象の名前」:「発言内容」）")
    return "None", user_speak, False

def user_speak(state: child.ChildAppState):
    """
    Args: state(AppState)
    Return: Dict[str] = 生成した発言
    """
    target = state.get("agent_name", "")

    while(1):
        user_utterance = input("あなた:")
        _, s, flag = user_speak_target_checker(target, user_utterance)

        if flag:
            #print(f"あなた:{s}")
            return {"history": [f"店員: {s}"]}

def return_state_checker(state: AppState):
    task_state = state.get("task_state", {})

    if(len(task_state) != 0):
        for agent_name in task_state.keys():
            if(task_state[agent_name] == False):
                return "No"
            
        return "Yes"

def task_number_dec(state: AppState):
    # ２つのタスクの終了時に呼び出される。(未達成タスク数の現象を行う)
    task_number = state.get("task_number", 0)
    task_number -= 1

    print(f"残りのタスク数:{task_number}")

    if(task_number < 0):
        print("ERROR: \"task_number\" is negative number.")
        sys.exit(-1)
    else:
        return {"task_number": task_number}
        
def training_end(state: AppState):
    task_number = state.get("task_number", -1)

    if(task_number == 0):
        return "Yes"
    elif(task_number > 0):
        return "No"
    else:
        print("ERROR: \"task_number\" is negative number.")
        sys.exit(-1)

## Sendを用いて任意数のノードを作成する ###########################################################################
def parallel_node(state: AppState): # 親グラフとサブグラフ間の橋渡しを行う
    agent_name = state.get("agent_name", "")
    #agent_names_list = state.get("current_speakers_names", [])
    agent_tasks = state.get("agent_tasks", {})
    #agent_personalities = state.get("speakers", [])
    current_target = state.get("current_target", "")
    history = state.get("history", [])
    model_name = state.get("model_name", "")
    thema = state.get("thema", "")
    fase_number = state.get("fase_number", 1)
    
    subgraph = state.get("subgraph", None)

    inputs = {"agent_name": agent_name,
              "agent_personality": SPEAKERS[SPEAKERS_NAMES.index(agent_name)],
              "agent_task": agent_tasks[agent_name],
              "current_target": current_target,
              "history": history,
              "model_name": model_name,
              "thema": thema}

    if subgraph != None:
        #response = subgraph.invoke(inputs, {"recursion_limit": 1000})
        response = subgraph.invoke(inputs, {"recursion_limit": RECURSION_LIMIT})

        return {"history": [response['response']], "fase_number": fase_number+1} 

def routing_parallel_nodes(state: AppState):
    """
    仮想ノードをSendで定義します(仮想ノード用のstateを用意).
    """
    target = state.get("current_target", "")
    fase_number = state.get("fase_number", 1)

    return [Send('parallel_node_' + str(fase_number), state | {'agent_name': target})]
#################################################################################################################

def connection_node(state: AppState):
    return {}

## フィードバック用の関数 #########################################################################################
def feedback_node(state: AppState):
    feedbacks = state.get("feedbacks", [])

    print("訓練結果:")

    if(len(feedbacks) == 0):
        print("問題ありません。")
    else:
        print("問題があります。")

#################################################################################################################

## プロセスの軌道 ################################################################################################
def graph_activation():
    pass

## 以下メイン（ユーザ入力のプロセス） ##############################################
if __name__ == "__main__":
    graph_activation()