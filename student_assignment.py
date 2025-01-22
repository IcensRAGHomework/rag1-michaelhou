import json
import traceback

from model_configurations import get_model_configuration

from langchain_openai import AzureChatOpenAI
from langchain_core.messages import HumanMessage
from langchain_core.prompts import ChatPromptTemplate, FewShotChatMessagePromptTemplate


gpt_chat_version = 'gpt-4o'
gpt_config = get_model_configuration(gpt_chat_version)

def generate_hw01(question):
    model_name = gpt_chat_version #'gpt-4o'
    model = get_model(model_name)
    examples = [
        {
            'input':'2024年台灣10月紀念日有哪些?',
            'output':'''
{
    "Result": [
        {
            "date": "2024-10-10",
            "name": "國慶日"
        }
   ]
}'''
        },
    ]
    example_prompt = ChatPromptTemplate.from_messages(
        [
            ('human', '{input}'),
            ('ai', '{output}'),
        ]
    )
    few_shot_prompt = FewShotChatMessagePromptTemplate(
        example_prompt = example_prompt,
        examples = examples,
    )
    #print(few_shot_prompt.invoke({}).to_messages())
    #print('#'*20)
    final_prompt = ChatPromptTemplate.from_messages(
        [
            ('system', '請回答台灣特定月份的紀念日有哪些(請用JSON格式呈現)?'),
            few_shot_prompt,
            ('human', '{input}'),
        ]
    )
    chain = final_prompt | model
    response = chain.invoke({'input': question})
    #print('content=\n', response.content)
    #with open('content.json', 'w', encoding='utf-8') as f:
    #    f.write(response.content)
    return response.content
    
def generate_hw02(question):
    pass
    
def generate_hw03(question2, question3):
    pass
    
def generate_hw04(question):
    pass
    
def demo(question):
    llm = AzureChatOpenAI(
            model=gpt_config['model_name'],
            deployment_name=gpt_config['deployment_name'],
            openai_api_key=gpt_config['api_key'],
            openai_api_version=gpt_config['api_version'],
            azure_endpoint=gpt_config['api_base'],
            temperature=gpt_config['temperature']
    )
    message = HumanMessage(
            content=[
                {"type": "text", "text": question},
            ]
    )
    response = llm.invoke([message])
    
    return response
def get_model(model_name):
    # we use deepseek model for fast verify...
    if model_name == 'deepseek-chat':
        model = ChatOpenAI(
            model=model_name,
            openai_api_key=os.environ.get('DEEP_SEEK_API_KEY'),
            openai_api_base='https://api.deepseek.com',
            #max_tokens=1024
            max_tokens=4096
        )
    elif 'gpt-4o' in model_name:
        model = AzureChatOpenAI(
                # model=gpt_config['model_name'],
                #model = 'gpt-4o-mini',
                #model = 'gpt-4o',
                model = model_name,
                deployment_name=gpt_config['deployment_name'],
                openai_api_key=gpt_config['api_key'],
                openai_api_version=gpt_config['api_version'],
                azure_endpoint=gpt_config['api_base'],
                temperature=gpt_config['temperature']
        )
    return model

#generate_hw01('3月')
