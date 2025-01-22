import json
import traceback
import requests

from model_configurations import get_model_configuration

from langchain_openai import AzureChatOpenAI
from langchain_core.messages import HumanMessage
from langchain_core.prompts import ChatPromptTemplate, FewShotChatMessagePromptTemplate

from langchain.output_parsers import StructuredOutputParser, ResponseSchema
from langchain.schema.runnable import RunnableSequence
from langchain.prompts import PromptTemplate

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



def get_calendarific_key():
    # TODO security issue on hardcode
    return 'mr7UegsykS1XglaQqOmVfSELeDerWtvB'

def query_holiday_by_api_raw_dict(year, month, country='TW', language='zh'):
    api_key = get_calendarific_key()
    #url = f'https://calendarific.com/api/v2/holidays?&api_key={api_key}&country={country}&year={year}&language={language}'
    # language NOT support
    # Our API supports premium parameters which are available to all users of our professional plan and above
    url = f'https://calendarific.com/api/v2/holidays?&api_key={api_key}&country={country}&year={year}&month={month}'
    dummy_load_flag = False
    if dummy_load_flag:
        try:
            # load local cach to prevet too much API usage
            with open(f'calendary/holidy_{year}_{month}.json', 'r', encoding='utf-8') as f:
                my_dict = json.load(f)
                return my_dict
        except:
            traceback.print_exc()
            pass
    try:
        #print('before call', url)
        response = requests.get(url)
        if response.status_code == 200:
            if not dummy_load_flag:
                return response.json()
            #os.makedirs('calendary', exist_ok=True)
            #with open(f'calendary/holidy_{year}_{month}.json', 'w', encoding='utf-8') as f:
            #    #f.write(response.text)
            #    my_dict = json.loads(response.text)
            #    json.dump(my_dict, f, indent=4)
            return response.json()
    except:
        traceback.print_exc()

def query_holiday(year, month):
    holidays = []
    try:
        my_dict = query_holiday_by_api_raw_dict(year=year, month=month)
        # we do a normalize here
        for holiday in my_dict['response']['holidays']:
            holidays.append({
                'date': holiday.get('date').get('iso'),
                'name': holiday.get('name'),
            })
    except Exception as e:
        traceback.print_exc()
        raise e
    return holidays

def generate_hw02(question):
    model_name = gpt_chat_version
    model = get_model(model_name)
    examples = [
        {
            'input':'2024年台灣10月紀念日有哪些?',
            'output':'''
{
    "year": "2024",
    "month": "10"
}'''
        },
        {
            'input':'2025年三月做了啥?',
            'output':'''
{
    "year": "2025",
    "month": "3"
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
            ('system', '從問題中提取年份和月份信息並輸出(請用JSON格式呈現)'),
            few_shot_prompt,
            ('human', '{input}'),
        ]
    )
    chain = final_prompt | model
    response = chain.invoke({'input': question})
    #with open(f'{output_dir}/content.md', 'w', encoding='utf-8') as f:
    #    f.write(response.content)
    parsered_info = json.loads(response.content)
    #print('extracted_info=', response)
    year = parsered_info['year']
    month = parsered_info['month']
    #test1 = query_holiday_by_api_raw_dict()
    result = query_holiday(year, month)
    return json.dumps({
        'Result': result
    })

    
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
# question = '2024年台灣12月紀念日有哪些?'
# print(generate_hw02(question))