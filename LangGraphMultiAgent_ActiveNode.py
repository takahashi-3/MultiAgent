# 「gpt-4o-mini」を使用しての実装となります。
# 2/4: JSAI_figure.2. の通りに変更（８フェーズからなるもの）
# 配膳を実際に渡すまで行う
# 配膳＋片付けでは初期に別の行動 -> プロンプトの検証を行う

from multiprocessing import Process, Manager
import operator
import os
import random
import sys
import time
from typing import Annotated, List, TypedDict, Dict, Any

from langchain_core.messages import HumanMessage, SystemMessage
from langchain_core.runnables.config import RunnableConfig
from langchain_openai import ChatOpenAI
from langgraph.graph import START, END, StateGraph
from langgraph.checkpoint.memory import MemorySaver
from langgraph.types import Send

import Children_node as child

os.environ["OPENAI_API_KEY"] = ""

# 詳細なプロフィールあり
#SPEAKERS = [{"1": "'国籍': '日本', '性別': '女性', '年代: '30代', '職業': '会社員', '性格': '落ち着いた性格で、じっくり物事を考えて発言します。'"},
#            {"2": "'国籍': '日本', '性別': '男性', '年代: '20代', '職業': '大学生', '性格': '明るく、前向きな性格です。'"},
#            {"3": "'国籍': '日本', '性別': '男性', '年代: '40代', '職業': '会社役員', '性格': '温厚で、話しやすい雰囲気を持っています。'"},
#            {"4": "'国籍': '日本', '性別': '女性', '年代: '30代', '職業': '会社員', '性格': '神経質で、高圧的です。'"},
#            {"5": "'国籍': '日本', '性別': '男性', '年代: '40代', '職業': '会社役員', '性格': '柔和で、低姿勢です。'"}]

# 詳細なプロフィールなし
# , '性格': '神経質な性格', '性格': '自分本位な性格'
SPEAKERS = [{"1": "'国籍': '日本', '性別': '女性'"},
            {"2": "'国籍': '日本', '性別': '男性'"},
            {"3": "'国籍': '日本', '性別': '男性'"},
            {"4": "'国籍': '日本', '性別': '女性'"},
            {"5": "'国籍': '日本', '性別': '男性'"}]
SPEAKERS_NAMES = ["1", "2", "3", "4", "5"]

# タスクの順序を設定する必要がある
# タスクの内容を変更（配膳、片付け）
#TASK_EXPLAIN = "#接客タスクの発生順序\n基本的には「入店→商品の注文→料理の配膳を待つ→片付けの要求」の順番で進んでいきます。\n「クレーム」は順序中のどの段階でも発生する可能性があります。"
TASK_EXPLAIN = ""

#TASK_DICT = {"入店": "店の入口に立ち、店員から声がかかるのを待つ。その後入店の人数を回答し、テーブルもしくはカウンター席への案内を待つ。\n店員によって、席への案内が行われた場合にこの接客タスクは終了する。'", # どのテーブルに誰が座っているのか決定する必要がある。（初期状態の設定）
#             "料理の注文": "挙手を行い、店員から声がかかるのを待つ。その後具体的な料理の注文を行う。メニューとしては「パンケーキ」「ハンバーガーセット」「バゲットセット」「サンドウィッチセット」「チョコレートケーキ」「ピザ」の6つが存在している。\n店員によって、注文の受領が行われ、間違いがない場合にこの接客タスクは終了する。'",  # 商品としてどれがあるということを指定する必要がある
#             "料理の配膳を待つ": "挙手を行い、店員から声がかかるのを待つ。その後店員に自分の注文した料理があとどのくらいで配膳されるのか尋ねる。\n店員によって回答が得られた場合にこの接客タスクは終了する。", # 自分がどの商品をオーダーしているのか記憶する必要がある
#             "片付けの要求": "挙手を行い、店員から声がかかるのを待つ。その後テーブル上の皿について片付けを要求する。\n店員によって、片付けが了承された場合にこの接客タスクは終了する。", # 自分がどの商品をオーダーしているのか記憶する必要がある
#             "クレーム": "挙手を行い、店員から声がかかるのを待つ。その後店員に対して「今回のテーマにおいて発生する可能性のある不手際」を述べる。\nクレーム内容に関して、店員からの回答が得られた場合にこの接客タスクは終了する。"} 

TASK_DICT = {"入店": "'発言':'入店の人数を回答し、テーブルもしくはカウンター席への案内を待つ。店員によって、席への案内が行われた場合にこの接客タスクは終了する。'", # どのテーブルに誰が座っているのか決定する必要がある。（初期状態の設定）
             "料理の注文": "'発言':'具体的な料理の注文を行う。メニューとしては「パンケーキ」「ハンバーガーセット」「バゲットセット」「サンドウィッチセット」「チョコレートケーキ」「ピザ」の6つが存在している。店員によって、注文の受領が行われ、間違いがない場合にこの接客タスクは終了する。'",  # 商品としてどれがあるということを指定する必要がある
             "料理の配膳": "'発言':'まず店員からの料理の配膳が行われ、配膳された料理が違う場合には、そのことについて発言を行う。店員からの配膳が行われ、配膳された料理が注文したものと同じである場合にこの接客タスクは終了する。'", # 自分がどの商品をオーダーしているのか記憶する必要がある
             "片付け": "'発言':'店員から片付けが行われた場合にはそれを了承し、行われなかった場合にはテーブル上の皿について片付けを要求する。店員によって、片付けが了承された場合にこの接客タスクは終了する。'", # 自分がどの商品をオーダーしているのか記憶する必要がある
             "クレーム": "'発言':'店員に対して「今回のテーマにおいて発生する可能性のある不手際」を述べる。クレーム内容に関して、店員から自分の望む回答が得られた場合にこの接客タスクは終了する。'"} 


PRIORITIES = "#接客タスクの優先順位\n1番:クレーム\n2番:料理の配膳, 入店\n3番:料理の注文\n4番:片付け\n"
RECURSION_LIMIT = 1000000000
MENU = ['パンケーキ','ハンバーガーセット', 'バゲットセット', 'サンドウィッチセット', 'チョコレートケーキ', 'ピザ']

INIT_TASK_SITUATION = ['料理の注文', '片付け']

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
    
    *del: fase_number = parallel_node用、フェーズの記録
    fase_priority(bool) = parallel_node用（True: high, False: low）
    request_users(List[str]) = 入店以外を行った顧客役の名前を格納するリスト(request_checkで使用)
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
    fase_priority: bool
    request_users: List[str]

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
    request_users: List[str] = []

    candidates_names = SPEAKERS_NAMES
    speakers: List[Dict[str, str]] = []

    model = ChatOpenAI(model=model_name)

    # 行動を起こすユーザの決定(現状二人同時に行動を起こす場合のみ想定)
    if(len(current_speakers_names) == 0): # 初期状態で誰が行動を起こすか決定されていない場合
        current_speakers_names = speakers_names
        #customer_number = 2
        #current_speakers_names = random.sample(speakers_names, customer_number)
    elif((not init_flag) and (len(current_speakers_names) == 1) and (len(speakers_names) > 0)):
        current_speakers_names.append((random.sample(speakers_names, 1))[0])

    # speaker_namesからcurrent_speakers_names内の人物を除外する
    #for name in current_speakers_names:
    #    if(name in speakers_names):
    #        speakers_names.remove(name)
        
    for customer_name in current_speakers_names:
        speakers.append({customer_name: SPEAKERS[SPEAKERS_NAMES.index(customer_name)]})
        task_state[customer_name] = False

    if(init_flag):
        thema = state.get("thema")
        task_dict = {}

        # 初期に発生するタスクを「入店」と「注文」で固定した場合
        for i in current_speakers_names:
            system_message = f"あなたには日本語で、{INIT_TASK_SITUATION}の中から{current_speakers_names}たちの起こしえる接客タスクを優先順位が異なるように選択するという役目が課されています。"
            human_message = f"{thema}というテーマにおいて、客である{i}が店員に対して、起こしえる接客タスクを{INIT_TASK_SITUATION}から１つ選択してください。また「接客タスクの優先順位」や「全ての客の接客タスク」を参考に、接客タスクの優先順位は他の客と異なるように選択してください。\n理由づけは行わずに選択した接客タスクのみを出力してください\n\n#全ての客の接客タスク:{task_dict}\n{PRIORITIES}\n{i}: "
            response = model.invoke([SystemMessage(system_message), HumanMessage(human_message)])
            task_dict[i] = response.content

            # 料理の注文、クレームを行った客には, request_checkで個別に対応
            if(('料理の注文' in response.content) or ('クレーム' in response.content)):
                request_users.append(i)

        # 初期に発生するタスクを固定しない場合
        #for i in current_speakers_names:
        #    system_message = f"あなたには日本語で、{TASK_DICT.keys()}の中から{current_speakers_names}たちの起こしえる接客タスクを優先度が異なるように選択するという役目が課されています。"
        #    human_message = f"{thema}というテーマにおいて、客である{i}が店員に対して、起こしえる接客タスクを{TASK_DICT.keys()}から１つ選択してください。また「接客タスクの優先順位」や「全客の接客タスク」を参考に、各客の選択した接客タスクの優先順位が異なるように選択してください。ローマ字を使ってはいけません。\n理由づけは行わずに選択した接客タスクのみを出力してください\n\n#全ての客の接客タスク:{[task_dict[name] for name in task_dict.keys()]}\n{PRIORITIES}\n{i}: "
        #    response = model.invoke([SystemMessage(system_message), HumanMessage(human_message)])
        #    task_dict[i] = response.content

        return {"agent_tasks": task_dict, "current_speakers_names": current_speakers_names, "speakers": speakers, "speakers_names": speakers_names, "task_state":task_state, "request_users":request_users}
    else:
        history = state.get("history", [])
        previous_task_dict = state.get("agent_tasks", {})
        task_dict = {}
        thema = state.get("thema", "")

        # 前のフェーズで接客を行われなかった顧客のタスクを取得
        for i in current_speakers_names:
            if(i in previous_task_dict.keys()):
                task_dict[i] = previous_task_dict[i]

        # タスクの更新
        for i in current_speakers_names:
            system_message = f"あなたには日本語で、{TASK_DICT.keys()}の中から{current_speakers_names}たちの起こしえる接客タスクを選択するという役目が課されています。"
            human_message = f"{thema}というテーマにおけるこれまでの履歴を確認したうえで、客である{i}が店員に対して、起こしえる接客タスクを{TASK_DICT.keys()}から１つ選択してください。また「接客タスクの優先順位」や「全ての客の接客タスク」を参考に、接客タスクの優先順位は他の客と異なるように選択してください。既に客である{i}の接客タスクが決定されていた場合、基本的にはそれに続く接客タスクを選択してください。\n理由づけは行わずに選択した接客タスクのみを出力してください。\n\n#全ての客のタスク:{task_dict}\n{TASK_EXPLAIN}\n{PRIORITIES}\n#会話の履歴\n"
            human_message = human_message + "\n".join(history) + f"\n{i}: "
            response = model.invoke([SystemMessage(system_message), HumanMessage(human_message)])
            task_dict[i] = response.content

            if(('料理の注文' in response.content) or ('クレーム' in response.content)):
                request_users.append(i)

        return {"agent_tasks": task_dict, "current_speakers_names": current_speakers_names, "speakers": speakers, "speakers_names": speakers_names, "task_state":task_state, "request_users": request_users}

# 行動を起こすように変更&request_user
#def situation_generator(state: AppState):
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
    model_name = state.get("model_name")

    model = ChatOpenAI(model=model_name)

    if(init_flag):
        for i in speakers_names:
            system_message = f"あなたの名前は{i}で、{thema}というテーマにおける客としての役割を持っています。また、{speakers[speakers_names.index(i)]}というパーソナリティをもっています。"
            human_message = f"あなたが選択した接客タスク（{agent_tasks[i]}）と「各接客タスクの詳細」の'行動'の内容をもとに、発言を行わず行動のみをシステムメッセージのように出力してください。また、「」を付けずに答えてください。この段階では具体的に何をして欲しいのか伝える必要はありません。\n#各接客タスクの詳細{TASK_DICT}\n{i}: "
            response = model.invoke([SystemMessage(system_message), HumanMessage(human_message)])
            first_situation += f"客{i}: {response.content}\n"
            situation_history += f"{i}: {response.content}\n"
    else:
        for i in speakers_names:
            system_message = f"あなたの名前は{i}で、{thema}というテーマにおける客としての役割を持っています。また、{speakers[speakers_names.index(i)]}というパーソナリティをもっています。"
            human_message = f"あなたが選択した接客タスク（{agent_tasks[i]}）の内容と「各接客タスクの詳細」をもとに、発言を行わず「手を挙げています」や「入口に現れました」のような行動をとってください。また、「」を付けずに答えてください。この段階では具体的に何をして欲しいのか伝える必要はありません。\n#各接客タスクの詳細{TASK_DICT}\n{i}: "
            response = model.invoke([SystemMessage(system_message), HumanMessage(human_message)])
            first_situation += f"客{i}: {response.content}\n"
            situation_history += f"{i}: {response.content}\n"

    print(first_situation, end="")

    return {"history": [situation_history], "init_flag": False}

# 状況の説明を行うもの
# クレーム,料理の配膳,入店,料理の注文,片付け
def task_to_situation(customer_name: str, task: str, option: str):
    if(task == "料理の配膳"):
        return f"{customer_name}に{option}を配膳する準備ができました\n", f"客{customer_name}に{option}を配膳する準備ができました\n"
    elif(task == "入店"):
        return f"{customer_name}が来店しました\n", f"客{customer_name}が来店しました\n"
    elif(task == "片付け"):
        return f"{customer_name}の皿が空になっています\n", f"客{customer_name}の皿が空になっています\n"
    else:
        return f"{customer_name}が挙手しています\n", f"客{customer_name}が挙手しています\n"

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
    model_name = state.get("model_name")

    model = ChatOpenAI(model=model_name)

    if(init_flag):
        for i in speakers_names:
            sh, fs = task_to_situation(i, agent_tasks[i], (random.sample(MENU, 1))[0])
            first_situation += fs
            situation_history += sh

    print(first_situation, end="")

    return {"history": [situation_history], "init_flag": False}


def utterance_target_checker(state: AppState, user_speak: str, speakers_names: List[str]):
    """
    ユーザの発言に発言対象が含まれているか判定します。
    Args: str = ユーザの発言
    Return: str = 発言対象の名前
            Bool = 含まれているかどうか(含まれている:True, 含まれていない:False)
    """
    #speakers_names = state.get("current_speakers_names", [])

    for i in speakers_names:
        if("@"+i+":" in user_speak):
            return i, user_speak.replace("@"+i+":", f"（{i}に対して）:"), True
    
    print("E:発言対象が入力されていないか、現在のフェーズに存在しない客へ発言を行っています。（@「発言対象の名前」:「発言内容」）")
    print(f"発言対象一覧{speakers_names}")
    return "None", user_speak, False

def request_check(state: AppState):
    
    """
    Args: state(AppState)
    Return: Dict[str] = 生成した発言
    """
    agent_tasks = state.get("agent_tasks", {})
    history = state.get("history", [])
    speakers = state.get("speakers", {})
    speakers_names = state.get("current_speakers_names", [])
    model_name = state.get("model_name", "")
    request_users = state.get("request_users", [])

    if(len(request_users) == 0):
        return {}
    
    current_history = ""
    model = ChatOpenAI(model=model_name)

    while(1):
        user_utterance = input("＊挙手を行っているお客様にご用件を伺いましょう。\nあなた:")    
        target, s, flag = utterance_target_checker(state, user_utterance, request_users)

        if flag:
            current_history += f"店員: {s}\n"

            system_message = f"あなたの名前は{target}で、{speakers[speakers_names.index(target)]}というパーソナリティをもっています。"
            human_message = f"あなたが選択した接客タスク（{agent_tasks[target]}）の内容と「各接客タスクの詳細」の'発言'の内容をもとに、自分の要件を一言で伝えてください。この段階では、具体的な要求をしてはいけません。\nまた、「」をつけて話してください。\n#各接客タスクの詳細{TASK_DICT}\n{target}: "
            response = model.invoke([SystemMessage(system_message), HumanMessage(human_message)])

            current_history += f"{target}: {response.content}\n"
            print(f"客{target}:{response.content}")
            request_users.remove(target)

            if(len(request_users) == 0):
                return {"history": [current_history], "current_speakers_names": speakers_names, "request_users": request_users}
            else:
                print("他のお客様にもご用件を伺いましょう。")


def user_speak_priority_check(state: AppState):
    """
    Args: state(AppState)
    Return: Dict[str] = 生成した発言
    """
    history = state.get("history", [])
    speakers = state.get("speakers", {})
    speakers_names = state.get("current_speakers_names", [])
    task_dict = state.get("agent_tasks", {})
    model_name = state.get("model_name", "")
    
    current_history = ""
    target_list = speakers_names.copy()
    model = ChatOpenAI(model=model_name)

    if(len(speakers_names) == 1):
        speakers_names.remove(target_list[0])
        return {"current_speakers_names": speakers_names, "current_target": target_list[0]}

    while(1):
        user_utterance = input("＊対応の優先順位が低いお客様にお待たせすることを伝えてください。\nあなた:")    
        target, s, flag = utterance_target_checker(state, user_utterance, speakers_names)

        if flag:
            current_history += f"店員: {s}\n"

            system_message = f"あなたの名前は{target}で、{SPEAKERS[SPEAKERS_NAMES.index(target)]}というパーソナリティをもっています。"
            human_message = f"店員は他の客に対応するため、あなたに待ってもらうという選択をとりました。これまでの履歴と「全ての客の接客タスク」、「接客タスクの優先順位」を参考にして、店員の接客の優先順位が間違っていると判断した場合には、渋々納得していることを自然な短い言葉で伝えてください。\n優先順位が正しい場合には、店員に対して「待たされることに納得した」ということを自然な短い言葉で伝えてください。また、発言は「」で囲ってください。\n\n#全ての客のタスク:{task_dict}\n{PRIORITIES}\n#会話の履歴\n "
            human_message = human_message + "\n".join(history) + "\n" + current_history + f"\n{target}: "
            response = model.invoke([SystemMessage(system_message), HumanMessage(human_message)])

            current_history += f"{target}: {response.content}\n"
            print(f"客{target}:{response.content}")
            target_list.remove(target)

            if(len(target_list) == 1):
                speakers_names.remove(target_list[0])
                print(f"＊客{target_list[0]}へ接客を行いましょう。")
                # speakers_namesには優先度の低い顧客が残る
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
        return agent_name, user_speak.replace("@"+agent_name+":", f"（{agent_name}に対して）"), True
    
    print("E:発言対象が入力されていないか、現在のフェーズに存在しない客へ発言を行っています。（@「発言対象の名前」:「発言内容」）")
    return "None", user_speak, False

def user_speak(state: child.ChildAppState):
    """
    Args: state(AppState)
    Return: Dict[str] = 生成した発言
    """
    target = state.get("agent_name", "")
    response = state.get("response", "")

    while(1):
        user_utterance = input("あなた:")
        _, s, flag = user_speak_target_checker(target, user_utterance)

        if flag:
            #print(f"あなた:{s}")
            return {"response": response + f"店員: {s}\n"}

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
    agent_names_list = state.get("current_speakers_names", [])
    agent_tasks = state.get("agent_tasks", {})
    #agent_personalities = state.get("speakers", [])
    current_target = state.get("current_target", "")
    history = state.get("history", [])
    model_name = state.get("model_name", "")
    thema = state.get("thema", "")
    fase_number = state.get("fase_number", 1)
    fase_priority = state.get("fase_priority", True)
    
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

        if fase_priority:
            if(len(agent_names_list) == 1):
                print(f"＊客{agent_names_list[0]}へ接客を行いましょう。")
                return {"history": [response['response']], "current_target": agent_names_list[0], "fase_priority": False}
        else:
            return {"history": [response['response']], "current_speakers_names": [], "fase_number": fase_number+1, "fase_priority": True}

def routing_parallel_nodes(state: AppState):
    """
    仮想ノードをSendで定義します(仮想ノード用のstateを用意).
    """
    target = state.get("current_target", "")
    fase_number = state.get("fase_number", 1)
    fase_priority = state.get("fase_priority", True)

    if fase_priority:
        return [Send('high_priority_parallel_node_' + str(fase_number), state | {'agent_name': target})]
    else:
        return [Send('low_priority_parallel_node_' + str(fase_number), state | {'agent_name': target})]
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
    """
    Return: None
    """

    # サブグラフの定義 #########################################################
    # サブグラフでは、ユーザの発話に対する顧客役の発話を生成します
    subgraph = StateGraph(child.ChildAppState)
    subgraph.add_node("user_utterance_1", user_speak)
    subgraph.add_node("customer_utterance_1", child.customer_agent)
    subgraph.add_node("user_utterance_2", user_speak)
    subgraph.add_node("customer_utterance_2", child.customer_agent_conclude) # お礼を述べる
    subgraph.add_node("customer_utterance_claim", child.customer_agent_claim) # 接客の足りない点を要求
    #subgraph.add_node("action_making_normal", child.action_making_normal)

    subgraph.add_edge(START, "user_utterance_1")
    subgraph.add_edge("user_utterance_1", "customer_utterance_1")
    subgraph.add_edge("customer_utterance_1", "user_utterance_2")
    subgraph.add_conditional_edges("user_utterance_2", child.task_finish_judge,
                                   {
                                       "Continue": "customer_utterance_claim",
                                       "End": "customer_utterance_2"
                                   })
    subgraph.add_edge("customer_utterance_claim", "user_utterance_2")
    subgraph.add_edge("customer_utterance_2", END)

    # ノード扱いにします。（コンパイルして実体化）
    node_subgraph = subgraph.compile()
    ############################################################################

    # 親グラフの定義 ############################################################
    workflow = StateGraph(AppState)

    ## ノードの定義
    # 1回目
    workflow.add_node("task_generator_1", task_generator) # fase0 顧客役LLMのタスク生成
    workflow.add_node("situation_generator_1", situation_generator) # fase1 顧客役LLMによる発話を伴わない行動の生成
    workflow.add_node("request_check_1", request_check) # fase2 挙手を行った顧客役に対する用件の確認
    workflow.add_node("user_speak_priority_check_1", user_speak_priority_check) # fase3 訓練者による声掛けのフェーズ
    workflow.add_node("high_priority_parallel_node_1", parallel_node) # fase4 優先度の高い顧客役に対する接客
    workflow.add_node("low_priority_parallel_node_1", parallel_node) # fase5 優先度の低い顧客役に対する接客
    
    # 2回目
    workflow.add_node("task_generator_2", task_generator) # fase0
    workflow.add_node("situation_generator_2", situation_generator) # fase1
    workflow.add_node("request_check_2", request_check) # fase2 挙手を行った顧客役に対する用件の確認 # fase2
    workflow.add_node("user_speak_priority_check_2", user_speak_priority_check) # fase3
    workflow.add_node("high_priority_parallel_node_2", parallel_node) # fase4
    workflow.add_node("low_priority_parallel_node_2", parallel_node) # fase5

    #workflow.add_node("task_number_dec", task_number_dec)
    #workflow.add_node("feedback_node", feedback_node)
    
    ##ノード間の枝の定義
    workflow.add_edge(START, "task_generator_1")
    workflow.add_edge("task_generator_1", "situation_generator_1")
    workflow.add_edge("situation_generator_1", "request_check_1")
    workflow.add_edge("request_check_1", "user_speak_priority_check_1")
    workflow.add_conditional_edges("user_speak_priority_check_1", routing_parallel_nodes, ["high_priority_parallel_node_1"])
    workflow.add_conditional_edges("high_priority_parallel_node_1", routing_parallel_nodes, ["low_priority_parallel_node_1"])
    workflow.add_edge("low_priority_parallel_node_1", END)

    #workflow.add_edge("low_priority_parallel_node_1", "task_generator_2")
    #workflow.add_edge("task_generator_2", "situation_generator_2")
    #workflow.add_edge("situation_generator_2", "request_check_2")
    #workflow.add_edge("request_check_2", "user_speak_priority_check_2")
    #workflow.add_conditional_edges("user_speak_priority_check_2", routing_parallel_nodes, ["high_priority_parallel_node_2"])
    #workflow.add_conditional_edges("high_priority_parallel_node_2", routing_parallel_nodes, ["low_priority_parallel_node_2"])
    #workflow.add_edge("low_priority_parallel_node_2", END)

    ## 親グラフのコンパイル
    group_discussion = workflow.compile()
    ############################################################################

    # グラフの描画 #############################################################
    try:
        img = group_discussion.get_graph().draw_mermaid_png()
        file_path = "../graph_images/output.png"
        with open(file_path, "wb") as f:
            f.write(img)
        print(f"状態遷移図が{file_path}に保存されました")

        img_subgraph = node_subgraph.get_graph().draw_mermaid_png()
        file_path = "../graph_images/output_subgraph.png"
        with open(file_path, "wb") as f:
            f.write(img_subgraph)
        print(f"サブグラフ（parallel_node部分）が{file_path}に保存されました")

    except Exception as e:
        print(f"画像保存中のエラー:{e}")
    ############################################################################

    init_current_speakers_names = ["1", "2"]
    init_speakers_names = ["1", "2"]

    # グラフ実行
    init_state = {"agent_tasks": {},
                  "current_speakers_names": init_current_speakers_names,
                  "current_target": "None",
                  "feedbacks": [],
                  "history": [],
                  "init_flag": True,
                  "model_name": "gpt-4o",
                  "speakers": {},
                  "speakers_names": init_speakers_names,
                  "subgraph": node_subgraph,
                  "task_number": random.randint(2,3), # タスクグループ数を２～３で決定
                  "task_state": {},
                  "thema": "日本の飲食店（ファミリーレストラン）",
                  "fase_number": 1,
                  "fase_priority": True,
                  "request_users": []}

    #print(f"{TASK_DICT.keys()}")

    #group_discussion.invoke(init_state, {"recursion_limit": 1000}, debug=False)
    finish = group_discussion.invoke(init_state, {"recursion_limit": RECURSION_LIMIT}, debug=False)

    with open("../MultiAgentService_History.txt", "w") as f:
        print(finish['history'], file=f)

## 以下メイン（ユーザ入力のプロセス） ##############################################
if __name__ == "__main__":
    graph_activation()