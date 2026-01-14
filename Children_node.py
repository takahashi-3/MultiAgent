import os

from typing import List, TypedDict, Dict
from pydantic import BaseModel, Field

from langchain_core.messages import HumanMessage, SystemMessage
from langchain_openai import ChatOpenAI

import global_value as g

TASK_DICT = {"入店": "'発言':'入店の人数を回答し、テーブルもしくはカウンター席への案内を待つ。店員によって、席への案内が行われた場合にこの接客タスクは終了する。'", # どのテーブルに誰が座っているのか決定する必要がある。（初期状態の設定）
             "料理の注文": "'発言':'具体的な料理の注文を行う。メニューとしては「パンケーキ」「ハンバーガーセット」「バゲットセット」「サンドウィッチセット」「チョコレートケーキ」「ピザ」の6つが存在している。店員によって、注文の受領が行われ、間違いがない場合にこの接客タスクは終了する。'",  # 商品としてどれがあるということを指定する必要がある
             "料理の配膳": "'発言':'まず店員からの料理の配膳が行われ、配膳された料理が違う場合には、そのことについて発言を行う。店員からの配膳が行われ、配膳された料理が注文したものと同じである場合にこの接客タスクは終了する。'", # 自分がどの商品をオーダーしているのか記憶する必要がある
             "テーブルの片付け": "'発言':'店員から片付けが行われた場合にはそれを了承し、行われなかった場合にはテーブル上の皿について片付けを要求する。店員によって、片付けが了承された場合にこの接客タスクは終了する。'", # 自分がどの商品をオーダーしているのか記憶する必要がある
             "クレーム": "'発言':'店員に対して「今回のテーマにおいて発生する可能性のある不手際」を述べる。クレーム内容に関して、店員からそのクレーム内容に対して、上手に対応するような妥当な回答が得られた場合にこの接客タスクは終了する。'"} 


# シミュレーション用：店員役LLMによる発話生成の出力形式
class Format_clerk_agent_utterance(BaseModel):
    """
    #### 店員役LLMによる発話生成の出力形式\n
    utterance(str) = 店員役LLMによる発話内容
    reason(str) = その発話を行った理由
    """
    utterance: str = Field(description='The Utterance made by you.')
    reason: str = Field(description='The reason for that why you made the utterance.')

# シミュレーション用：店員役LLMによる初期発話生成のプロンプトにおける対話履歴の整形
def get_prompt_target_history_for_init_speak_clerk_agent(target_agent_history: List[str]) -> str:
    context = ''
    for temp_history in reversed(target_agent_history):
        if((len(context) > 0)):
            break
        for utt in temp_history.split('\n'):
            if('*System:' in utt):
                context += '\t\t' + utt + '\n'

    return '\n' + context

# シミュレーション用：店員役LLMによる発話生成のプロンプトにおける対話履歴の整形
def get_prompt_target_history_for_clerk_agent(target_agent_history: List[str]) -> str:
    context = '\n'
    for temp_history in target_agent_history:
        for utt in temp_history.split('\n'):
            if(utt != ''):
                context += '\t\t' + utt + '\n'

    return context

# 店員役LLMのプロンプト確認
def check_clerk_llm_prompt(prompt: str, selected_customer: str, reason: str, out_f_name: str):
    # コンテキストの確認
    #print(task_dict)
    os.makedirs(f'./{g.output_dir}/prompt', exist_ok=True)
    with open(f'./{g.output_dir}/prompt/{out_f_name}.txt', 'a') as fp:
        fp.write(prompt + f'\n\n\t出力: {selected_customer}\n\t理由: {reason}\n\n')

## state ######################################################################################
class ChildAppState(TypedDict):
    """
    agent_name(str) = エージェントの名前\n
    agent_personality(str) = エージェントのパーソナリティ\n
    agent_task(Dict[str, str]) = エージェントのタスク(task: 具体的なタスク, option:タスクの詳細)\n
    current_target(str) = 現在ユーザが対応を行っているエージェント\n
    child_history(List[str]) = 会話の履歴\n
    model_name(str) = 推論を行わせるモデル名\n
    response: この時点におけるエージェントの発言(出力のみ)\n
    return_state: 現在接客を受けているエージェントがクレームを行う場合、または設定されたタスクが完了していないと判定した場合""（ユーザの発話へ）, タスクが完了していると判断した場合"エージェント名"(タスク生成へ)\n
    thema(str) = 会話のテーマ\n
    utterance_num(int) = 接客中の発話回数\n
    LLM_sim(bool) = LLM同士の接客訓練シミュレーションであるか否か
    """
    agent_name: str
    agent_personality: str
    agent_task: Dict[str, str]
    current_target: str
    child_history: List[str]
    model_name: str
    response: str
    return_state: str
    thema: str
    utterance_num: int
    LLM_sim: bool

###################################################################################################

## サブグラフに使用されるノード(ChildAppState) ######################################################
def customer_agent(state: ChildAppState) -> bool:
    agent_name = state.get("agent_name", "")
    agent_personality = state.get("agent_personality")
    agent_task = state.get("agent_task", {})
    #current_target = state.get("current_target", "")
    history = state.get("child_history", [])
    model_name = state.get("model_name", "")
    prev_response = state.get('response')
    thema = state.get("thema", "")

    model = ChatOpenAI(model=model_name)

    system_message = f"あなたの名前は{agent_name}で、{thema}というテーマにおける客としての役割を持っています。また、{agent_personality}というパーソナリティをもっています。"
    #system_message = f"あなたは次のようなパーソナリティをもった人物です。\n---------------\n{agent_personality}\n---------------\nこの人物\"{agent_name}\"として店員であるuserと{thema}について会話をしてください。"
    #human_message_prefix = f"あなたは今、店員に{agent_task['task']}を行うというタスクをもっています。\nこれまでの会話の履歴を見て、あなたの次の発言を短い自然な話し言葉で行ってください。また、発言は「」で囲い、発言が「各接客タスクの詳細」の'発言'の内容から逸脱し過ぎないように気をつけてください。\n#各接客タスクの詳細{TASK_DICT}\n#会話の履歴\n"
    human_message_prefix = f"あなたは今、店員に{agent_task['task']}を行うというタスクをもっています。\nこれまでの会話の履歴を見て、あなたの次の発言を短い自然な話し言葉で行ってください。また、発言は「」で囲い、もし\'Option\'や\'Sub_Task\', \'Sub_Option\'の指定があれば、それに沿って対話を行ってください。\n\n#各接客タスクの詳細:{TASK_DICT}\n\n#Option:{agent_task['option']}\n\n#Sub_Task:{agent_task['sub_task']}\n#Sub_Option:{agent_task['sub_option']}\n\n#会話の履歴:\n"
    human_message = human_message_prefix + "\n".join(history) + f"\n{agent_name}: "
    
    response = model.invoke([SystemMessage(content=system_message), HumanMessage(content=human_message)])

    utterance = f"{agent_name}: {response.content}"
    utterance = utterance.replace(" ", "")
    utterance = utterance.replace("\n", "")
    left_limit = utterance.find("「")
    right_limit = utterance.find("」")

    print(f"客{agent_name}:{utterance[left_limit:right_limit+1]}")

    agent_output_dir = f'./{g.output_dir}/customer-agent_history/agent-{agent_name}/'
    os.makedirs(agent_output_dir, exist_ok=True)
    with open(agent_output_dir + 'history.txt', 'a') as fp:
        fp.write(f'utterance: {utterance[left_limit:right_limit+1]}\nhistory:\n\t{history}\n\n\n')
    
    with open(agent_output_dir + 'prompt.txt', 'a') as fp:
        fp.write(f'customer-utt_prompt:\n{human_message}\n\n')

    history.append(agent_name + ":" + utterance[left_limit:right_limit+1])

    return {"response": prev_response + '\n' + agent_name + ":" + utterance[left_limit:right_limit+1], "child_history": history}

    #return {"response": "", "return_state": ""} # 対象でない場合に対応できるよう、初期値の設定を行います。

# ユーザの発話ターン
def user_speak(state: ChildAppState):
    """
    接客対象との対話
    Args: state(AppState)
    Return: Dict[str] = 生成した発言
    """
    agent_name = state.get('agent_name')
    history = state.get('child_history')
    model_name = state.get("model_name", "")
    response = state.get('response', '')
    thema = state.get('thema')
    LLM_sim = state.get('LLM_sim')

    if(LLM_sim):
        model = ChatOpenAI(model=model_name)
        system_message = f"あなたには、{thema}というテーマにおける店員として、顧客役エージェント(客{agent_name})に接客を行うという役割が課されています。"
        human_message = f"{thema}というテーマにおいて、\'顧客役エージェント\'(客{agent_name})に対して、接客を行ってください。またその際には、\'客{agent_name}の接客状況\'を考慮してください。また、顧客からのクレームには、上手に対応するような妥当な回答を行うようにしてください。\n\n#客{agent_name}の接客状況:{get_prompt_target_history_for_clerk_agent(history)}"

        structured_response_model = model.with_structured_output(Format_clerk_agent_utterance)
        clerk_response = structured_response_model.invoke([SystemMessage(system_message), HumanMessage(human_message)])
        
        check_clerk_llm_prompt(human_message, clerk_response.utterance, clerk_response.reason, 'clerk-llm_utterance_prompt')

        clerk_llm_utt = (clerk_response.utterance).replace('\n', '')
        print(f'店員(User): {clerk_llm_utt}')
        user_utterance = f'店員(User): {clerk_llm_utt}\n'
        history.append(user_utterance)
        return {'response': response + '\n' + user_utterance, 'child_history': history}
    else:
        while(1):
            user_utterance = input('あなた:')

            if(len(user_utterance) > 0):
                history.append(f'店員(User): {user_utterance}\n')
                return {'response': response + '\n' + f'店員(User): {user_utterance}\n', 'child_history': history}


# 接客の締めくくりにおける顧客エージェントの発話
def customer_agent_conclude(state: ChildAppState):
    agent_name = state.get("agent_name", "")
    agent_personality = state.get("agent_personality")
    agent_task = state.get("agent_task", "")
    #current_target = state.get("current_target", "")
    history = state.get("child_history", [])
    model_name = state.get("model_name", "")
    prev_response = state.get("response", "")
    thema = state.get("thema", "")

    model = ChatOpenAI(model=model_name)
    system_message = f"あなたの名前は{agent_name}で、{thema}というテーマにおける客としての役割を持っています。また、{agent_personality}というパーソナリティをもっています。"
    #system_message = f"あなたは次のようなパーソナリティをもった人物です。\n---------------\n{agent_personality}\n---------------\nこの人物\"{agent_name}\"として店員であるuserと{thema}について会話をしてください。"
    #human_message_prefix = f"あなたは今、店員に{agent_task['task']}を行うというタスクをもっています。\nこれまでの会話であなたに対しての接客は完了しました。なのでこれまでの履歴をもとに、店員に対して、納得したこともしくは感謝していることを自然な短い文体で伝えてください。発言は丁寧になりすぎないように注意してください。また、発言は「」で囲ってください。\n\n#会話の履歴:\n"
    human_message_prefix = f"あなたは今、店員に{agent_task['task']}を行うというタスクをもっています。\nこれまでの会話であなたに対しての接客は完了しました。なので、これまでの履歴をもとに、あなたの次の発言を短い自然な話し言葉で行ってください。発言は丁寧になりすぎないように注意してください。また、発言は「」で囲ってください。\n\n#会話の履歴:\n"
    human_message = human_message_prefix + "\n".join(history) + f"\n{agent_name}: " #"\n" + prev_response 
        
    response = model.invoke([SystemMessage(content=system_message), HumanMessage(content=human_message)])

    utterance = f"{agent_name}: {response.content}"
    utterance = utterance.replace(" ", "")
    utterance = utterance.replace("\n", "")
    left_limit = utterance.find("「")
    right_limit = utterance.find("」")
    print(f"客{agent_name}:{utterance[left_limit:right_limit+1]}")
    
    agent_output_dir = f'./{g.output_dir}/customer-agent_history/agent-{agent_name}/'
    os.makedirs(agent_output_dir, exist_ok=True)
    with open(agent_output_dir + 'history.txt', 'a') as fp:
        fp.write(f'utterance: {utterance[left_limit:right_limit+1]}\nhistory:\n\t{history}\n\n\n')
    
    with open(agent_output_dir + 'prompt.txt', 'a') as fp:
        fp.write(f'customer-conclude-utt_prompt:\n{human_message}\n\n')

    #return {"response": prev_response + '\n' + agent_name + ':' + utterance[left_limit:right_limit+1] + '\n'}
    return {"response": agent_name + ':' + utterance[left_limit:right_limit+1] + '\n'}

###############################################################################################

## 分岐判定などに使用する関数（参照は"LangGraphMultiAgent...py"側です） ###################################################################
def target_judge(state: ChildAppState):
    """
    エージェントが現在の接客対象であるか判定を行い、それに応じてグラフの遷移先を決定します（Mainのsubgraphを見てください）。
    """
    agent_name = state.get("agent_name", "")
    current_target = state.get("current_target", "")
    
    if(agent_name == current_target):
        return True
    else:
        return False

#def user_proper_judge(state: ChildAppState):
    """
    ユーザの対応が顧客側から見て適切かどうか判定します。
    Args: state=AppState
    Return: 
    """
    agent_name = state.get("agent_name", "")
    agent_personality = state.get("agent_personality", {})
    current_target = state.get("current_target", "")
    history = state.get("history", [])
    model_name = state.get("model_name", "")
    thema = state.get("thema", "")

    if(current_target == "None"):
        return "Yes"

    model = ChatOpenAI(model=model_name)

    system_message = f"あなたは{agent_name}です。{agent_personality[agent_name]}というパーソナリティをもっています。あなたには飲食店における顧客として、先ほどの店員(user)の行動が適切かどうか判断する役目が課せられています。"
    human_message = f"これまでの{thema}についての会話の履歴を見てください。\nそのあと、顧客として直前の店員(user)の行動が適切であると判断した場合[Yes]、そうでない場合[No]、自分が現在接客を受けておらず接客の順番に不満が無い場合[Except]を出力してください。\nまた現在、店員(user)は{current_target}に接客を行っています。\n\n# 履歴\n{history}\n\n"

    while(1):
        response = model.invoke([SystemMessage(content=system_message),
                                 HumanMessage(content=human_message)])

        if("Yes" in response.content):
            if(agent_name == current_target):
                return "Yes"
            else:
                #print(f"客{agent_name}:待機を選択しました。")
                return "Except"
        elif("No" in response.content):
            return "No"
        elif("Except" in response.content):
            #print(f"客{agent_name}:待機を選択しました。")
            return "Except"
        else:
            continue

def task_finish_judge(state: ChildAppState):
    """
    ユーザの接客対象となっているエージェントのタスクが完了しているか判定します。
    Args: state(AppState)
    Reture: str = タスクの終了判定(Yes or No)
    """
    agent_name = state.get("agent_name", "")
    agent_task = state.get("agent_task", "")
    history = state.get("child_history", [])
    model_name = state.get("model_name", "")
    prev_response = state.get("response")
    thema = state.get("thema", "")

    model = ChatOpenAI(model=model_name)

    system_message = f"あなたの名前は{agent_name}で、{thema}というテーマにおける客としての役割を持っています。また、これまでの会話の履歴から現在あなたに対して行われている接客が不足していないか判断する役目が課せられています。"
    human_message = f"これまでの履歴と「各接客タスクの詳細」の'発言'の内容を見てください。\nそのあと、現在のあなたの接客タスク({agent_task['task']})に対する店員の接客を終了してもいいと判断した場合[Yes]、そうでない場合[No]を出力してください。またもし\'Option\'や\'Sub_Task\'、\'Sub_Option\'の指定があれば、それも判断材料に入れて判断を行ってください。\n\n#各接客タスクの詳細:{TASK_DICT}\n\n#Option:{agent_task['option']}\n\n#Sub_Task:{agent_task['sub_task']}\n#Sub_Option:{agent_task['sub_option']}\n\n#履歴:\n{history}\n{prev_response}\n"

    while(1):
        response = model.invoke([SystemMessage(content=system_message),
                                 HumanMessage(content=human_message)])

        if("Yes" in response.content):
            return "End"
        elif("No" in response.content):
            return "Continue"
        else:
            continue