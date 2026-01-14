# 顧客への接客終了後、訓練者がNG行動を起こした場合に、フィードバックを行うエージェント

import os

from typing import List, TypedDict, Dict
from pydantic import BaseModel, Field

from langchain_core.messages import HumanMessage, SystemMessage
from langchain_openai import ChatOpenAI

import global_value as g
import Controller_node as controller

# 接客上のNG行動に関する説明
NG_EXAMPLE = '\n - 前の時間から連続して注文などを行おうとしている顧客への接客を後回しにする\n - 訓練者が、以前の接客から特定の顧客が怒りっぽい性格であることを知っているにも関わらず、その顧客への接客を後回しにする\n - 怒りっぽい顧客2名が同時にタスクを発生させる場合に、一方の顧客への対応でもう一方を長時間待たせる\n - 前の時間からタスクを発生させている顧客に加えて怒りっぽい顧客が新規にタスクを発生させる場合に、一方の顧客への対応でもう一方を長時間待たせる'

class Format_feedback(BaseModel):
    flag: bool = Field(description='The flag that represents whether the \'feedback\' is required.')
    feedback: str = Field(description='Specific feedback content.')
    reason: str = Field(description='the reason for outputting the \'feedback\' content.')

def feedback(state: controller.AppState):
    agent_wait_time = state.get('agent_wait_time')
    history_for_each_agent = state.get('history_for_each_agent')
    model_name = state.get('model_name')
    speakers_personality = state.get('speakers_personality')
    task_dict = state.get('agent_tasks')
    task_state = state.get('task_state')
    thema = state.get('thema')

    model = ChatOpenAI(model=model_name, temperature=0.0)
    system_message = f"あなたには、{thema}というテーマにおける接客訓練の指導役として、訓練者である店員(User)の直近の接客に関するフィードバックを行うという役割が課されています。"
    human_message = f"\'これまでの接客状況\'を確認した上で、店員(User)の接客に関するフィードバックを行なってください。またフィードバックを行う際には、\'接客におけるNG行動\'を参照してください。また、フィードバックの内容に\'接客におけるNG行動\'を直接示してはいけません。\n\n#接客におけるNG行動:{NG_EXAMPLE}\n\n#これまでの接客状況:{controller.get_prompt_history_for_each_agent(speakers_personality, history_for_each_agent, task_dict, task_state, agent_wait_time)}"

    structured_model = model.with_structured_output(Format_feedback)
    response = structured_model.invoke([SystemMessage(system_message), HumanMessage(human_message)])
    
    feedback_required = response.flag
    os.makedirs(f'./{g.output_dir}/prompt', exist_ok=True)

    if(feedback_required):
        feedback = response.feedback
        reason = response.reason
        
        with open(f'./{g.output_dir}/prompt/coach_agent_prompt.txt', 'a') as fp:
            fp.write(human_message + f'\n\n\tフィードバック:{feedback}\n\t理由: {reason}\n\n\n\n')
    else:
        with open(f'./{g.output_dir}/prompt/coach_agent_prompt.txt', 'a') as fp:
            fp.write(human_message + f'\n\n\tフィードバック: 無し\n\t理由: 無し\n\n\n\n')


    return {'feedback': feedback}
