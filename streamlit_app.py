### References: data_professor - https://tinyurl.com/bdhsrm84
# Code refactored from https://docs.streamlit.io/knowledge-base/tutorials/build-conversational-apps

import streamlit as st
import json
import openai
import os
from pathlib import Path
from dotenv import load_dotenv
import numpy as np
import pandas as pd
import glob
import re
import gdown
import requests
from io import StringIO

dir_path = Path.cwd()
data_path = (dir_path / "data").resolve()
images_path = (dir_path / "images").resolve()
temp_path = (dir_path / "temp").resolve()
env_path = (dir_path / ".env").resolve()

try:
    OPENAI_KEY = os.environ.get("OPENAI_API_KEY")
except:
    OPENAI_KEY = st.secrets['OPENAI_API_KEY']
    
@st.cache_resource(show_spinner=False)
def init_openai():
    # initialize connection to OpenAI
    openai.api_key = OPENAI_KEY

@st.cache_resource(show_spinner=False)
def load_data(file_name="plan_df.csv"):
    
    try:
        data = pd.read_csv(file_name)
    except:
        url="https://drive.google.com/uc?id=1Jwdb-rr8JAdEmNu0uf_uDuup03rBgv1m"
        output_file_name = "gdrive_plan_df.csv"
        gdown.download(url, output_file_name, quiet=False)
        data = pd.read_csv(output_file_name)
    else:
        # !pip install py7zr #for streamlit
        from py7zr import unpack_7zarchive
        import shutil
        
        shutil.register_unpack_format('7zip', ['.7z'], unpack_7zarchive)
        shutil.unpack_archive('plan_df.7z', '.')
        data = pd.read_csv(file_name)
    return data

plan_df = load_data()

@st.cache_resource(show_spinner=False)
def load_md_files(file_name="plan_df.csv"):
    try:
        directory_path = str(data_path)
        md_files = glob.glob((directory_path + '/*.md'))
        md_files = [file.split("\\data\\")[1].split(".md")[0] for file in md_files]
    except:
        # !pip install py7zr #for streamlit
        from py7zr import unpack_7zarchive
        import shutil
        
        shutil.register_unpack_format('7zip', ['.7z'], unpack_7zarchive)
        shutil.unpack_archive('pearl_ai_broker_mds.7z', '.')
        
        directory_path = str(data_path)
        md_files = glob.glob((directory_path + '/*.md'))
        md_files = [file.split("\\data\\")[1].split(".md")[0] for file in md_files]
    return md_files

md_files = load_md_files()

def clean_md(markdown):
    # Split the Markdown content by lines
    lines = markdown.strip().split('\n')

    # Regular expression to match table separators
    separator_re = re.compile(r'\|[^|]*\|')

    # Process each line and remove extra spaces in the table
    for i, line in enumerate(lines):
        if separator_re.match(line):
            # Split the table cells by the '|' character
            cells = line.split('|')
            # Remove leading and trailing spaces from each cell
            cleaned_cells = [cell.strip() for cell in cells]
            # Reconstruct the line with cleaned cells
            lines[i] = '|'.join(cleaned_cells)

    # Reconstruct the updated Markdown content
    updated_markdown_content = '\n'.join(lines)
    return updated_markdown_content

def create_contexts(df = plan_df, k=3, **kwargs):
    # state_code, county_name, metal_level, plan_type, child_only_offering, 
    # Function will take a dictionary and use it to filter a dataframe ('plan_df').
    # It returns up to k contexts, joined together.
    
    # Need to rename arguments to their column names
    rename_filter_keys = {
        "state_code": "state code",
        "county_name": "county name",
        "price_insensitive": "metal level", # a little confusing because 'high' actually means customer is price insensitive. 
        "plan_type": "plan type",
        "child_only_offering": "child only offering"
    }
    # ChatGPT: Define a dictionary to hold the combined arguments
    filter_dict = {}
    contexts = ''
    additional_cols = None
    
    print(kwargs)
    # Add all keyword arguments (**kwargs) to the dictionary
    for key, value in kwargs.items():
        # print(key, value)
        if key in rename_filter_keys.keys():
            filter_dict[rename_filter_keys[key]] = value.lower()
            if key.lower() == "price_insensitive" and (value.lower() != "low" and value.lower() != "high"): #in case we get anything other than low or high
                filter_dict[rename_filter_keys[key]] = "high" # metal level
        else:
            if value !=None: # expecting a list of columns from the string the LLM will pass in but the string could be NoneType
                if isinstance(value, list):
                    additional_cols = ','.join(list).split(',') #being extra cautious here that our list really ends up being a list of separate columns not a comma separated string of cols
                elif isinstance(value, str):
                    additional_cols = value.lower().split(',') 

    # Initialize some variables
    contexts = ""
    cols = [
        'issuer name',
        'plan marketing name',
        'child only offering',
        'customer service phone number local',
        'plan brochure url'
        # Premium
    ] 
    
    if 'additional_cols' in kwargs.keys() and additional_cols !=None and len(additional_cols) > 0:
        cols.extend(additional_cols)

    # ChatGPT: Create a mask based on the filters
    mask = pd.Series([True] * len(df))

    for col, value in filter_dict.items():
        mask = mask & (df[col] == value)

    # Apply the mask to filter the DataFrame
    filtered_df = df[mask]

    # Initialize an empty list to store formatted text
    formatted_text = []
        
    file_names = [file+'.md' for file in filtered_df['unique_name']]
    # processed_plan_unique_name = [item for item in file_names if item in md_files]

    final_filtered_df = filtered_df[cols].head(k)
    additional_plan_cols = [
        'routine dental services - adult (coverage)',
        'basic dental care - adult (coverage)',
        'major dental care - adult (coverage)',
        'orthodontia - adult (coverage)',
        'dental check-up for children (coverage)',
        'basic dental care - child (coverage)',
        'major dental care - child (coverage)',
        'dental deductible - individual - standard',
        'dental deductible - family - standard',
        'dental deductible - family (per person) - standard',
        'dental maximum out of pocket - individual - standard',
        'dental maximum out of pocket - family - standard',
        'dental maximum out of pocket - family (per person) - standard',
        'routine dental services - adult', 'basic dental care - adult',
        'major dental care - adult', 'orthodontia - adult',
        'dental check-up for children', 'basic dental care - child',
        'major dental care - child', 'orthodontia - child',
    ]
    
    new_cols = cols + additional_plan_cols
    final_filtered_df = filtered_df[new_cols].head(k)

    # Get dictionary of rows
    # Convert DataFrame to a list of dictionaries
    dict_list = final_filtered_df.to_dict(orient='records')

    
    if len(final_filtered_df) == 0: 
        contexts +="### There are no plans. Instruction: Ask for other options like different price sensitivity. Use another county in that state if necessary. Then call this function again."
        return contexts

    # Loop through the list of dictionaries and format them
    for index, row_dict in enumerate(dict_list):
        
        try:
            if file_names[index].replace(".md", "") in md_files:
                
                # row_dict = {k:v for k,v in row_dict.items() if k in cols} # stripping out additional info if we have the plan brochure
                row_dict = {k:v for k,v in row_dict.items() if k in cols} # stripping out additional info if we have the plan brochure
                
                formatted_row = ';'.join([f'{key} is {value}' for key, value in row_dict.items()])
                    
                # ChatGPT: Get md and clean it
                with open(Path(data_path / f'{file_names[index]}'), 'r', encoding='utf-8') as md_file:
                    markdown_content = md_file.read()
                    markdown_content = '\n'.join(line.strip() for line in markdown_content.splitlines())
                    cleaned_markdown_content = clean_md(markdown_content)
                formatted_text.append(f"{index+1}: {formatted_row}"+'\n'+f'plan details:\n{cleaned_markdown_content}')
            else: #if we don't have the plan brochure, rock and roll with original row_dict with additional cols / plan details
                formatted_row = ';'.join([f'{key} is {value}' for key, value in row_dict.items()])
                formatted_text.append(f"{index+1}: {formatted_row}"+ formatted_row+'\n')
                
        except Exception as e:
            pass

    # Combine the formatted text with a separator
    separator = '\n#\n'
    if len(final_filtered_df) > 0:
        contexts += f"### Here are the list of plans: \n#\n# " 
        contexts += separator.join(formatted_text)
    else:
        contexts +="### There are no plans. Instruction: Ask for other options like different price sensitivity. Use another county in that state if necessary. Then call this function again."
    # print(tiktoken_len(contexts))

    return contexts

# Inspired by this code: https://github.com/openai/openai-cookbook/blob/main/examples/How_to_call_functions_with_chat_models.ipynb & https://platform.openai.com/docs/guides/gpt/function-calling
def create_system_prompt():
    # ChatGPT
    chat_prompt = [
        {
            "role": "system", #prompt created by chat.openai.com based on prompting.
            "content": 
                """Your name is `Pearl`. You are a truthful and friendly dental insurance broker assistant. Your role is to assist customers in finding the most suitable dental insurance policy based on their location, dental history, and preferences. You can also provide information about oral health practices and answer related questions.
                Please follow this conversation flow:
                1. Begin by introducing yourself and role. Prompt the user for only MAXIMUM of two questions at a time not all. They need to provide their state and county, how price conscious they are, plan type, adult / child only offering, and other details. Ask for clarification if a user request is ambiguous. Simplify insurance jargons like metal level, epo, ppo, indemnity as necessary.
                2. Once you have the key information, call the 'create_contexts' function to filter data for matching policies. Don't make assumptions about what values to plug into functions.
                3. Based on the available policies, compare them and recommend the policy that best suits the customer's needs for a reasonable price.
                4. Provide information about the recommended policy, including coverage details, costs, url to pdf, customer service number, and any other relevant information.
                5. Engage in a natural conversation with the user, addressing any questions or concerns they may have about dental insurance.
                Remember to maintain a helpful and informative tone throughout the conversation.
                6. Don't ask again for information they've given you in the chat history.
                7. If they ask you to recommend a plan based on existing {plan details} ask them about the dental care history and oral hygiene practices. Use this information and any other details they've given you to recommend the best plan.
                8. Lastly but important, be concise and if the question can't be answered based on the context, acknowledge that you don't know and respond with \"I can connect you with a customer service representative who can better assist you if you'd like.\""
                """
        }
    ]

    return chat_prompt

chat_model = "gpt-4-0613" # "gpt-3.5-turbo-0613" #
def run_conversation(messages):

    functions = [
        {
            "name": "create_contexts",
            "description": "Gets dental plan information from various plans",
            "parameters": {
                "type": "object",
                "properties": {
                    "state_code": {
                        "type": "string",
                        "description": "The member's 2-letter state code, e.g., al."
                    },
                    "county_name": {
                        "type": "string",
                        "description": "The member's county name, e.g., autauga."
                    },
                    "price_insensitive": {
                        "type": "string",
                        "enum": ["low", "high"],
                        "description": "Is affordability a key consideration when it comes to your dental insurance preferences? Can only be low or high not medium"
                        
                    },
                    "plan_type": {
                        "type": "string",
                        "enum": ["ppo", "epo", "hmo", "indemnity"],
                        "description": """
                                            What type of plan the member needs e.g. ppo, hmo, exclusive provider organization (epo), indemnity (fee-for-service)
                                            A ppo offers flexibility with a network of preferred providers, allowing you to choose any dentist, but offering cost savings when you visit in-network dentists.
                                            An hmo requires you to select a primary dentist within its network, and all dental care is coordinated through them.
                                            An epo plan gives you the freedom to choose any dentist or specialist for your dental care. There is no requirement to select a primary dentist, and you can visit any dental provider you prefer, in-network and out-of-network.
                                            Indemnity plans offer the freedom to see any dentist and provide reimbursement for a portion of the covered expenses. These plans often have higher premiums. Can only be ppo, epo, hmo, indemnity.
                                        """
                    },
                    "child_only_offering": {
                        "type": "string",
                        "enum": ["allows adult and child-only", "allows child-only"],
                        "description": "If the member is looking for child only or plans that cover adults as well"
                    },
                    "additional_cols": {
                        "type": "string",
                        "enum": [
                            'routine dental services - adult (coverage)',
                            'basic dental care - adult (coverage)',
                            'major dental care - adult (coverage)',
                            'orthodontia - adult (coverage)',
                            'dental check-up for children (coverage)',
                            'basic dental care - child (coverage)',
                            'major dental care - child (coverage)',
                            'orthodontia - child (coverage)', 
                            'premium child age 0-14',
                            'premium child age 18', 'premium adult individual age 21',
                            'premium adult individual age 27', 'premium adult individual age 30',
                            'premium adult individual age 40', 'premium adult individual age 50',
                            'premium adult individual age 60', 'premium couple 21',
                            'premium couple 30', 'premium couple 40', 'premium couple 50',
                            'premium couple 60', 'premium couple+1 child age 21',
                            'premium couple+1 child age 30', 'premium couple+1 child age 40',
                            'premium couple+1 child age 50', 'premium couple+2 children age 21',
                            'premium couple+2 children age 30',
                            'premium couple+2 children age 40',
                            'premium couple+2 children age 50',
                            'premium couple+3 children age 21',
                            'premium couple+3 children age 30',
                            'premium couple+3 children age 40',
                            'premium couple+3 children age 50',
                            'premium individual+1 child age 21',
                            'premium individual+1 child age 30',
                            'premium individual+1 child age 40',
                            'premium individual+1 child age 50',
                            'premium individual+2 children age 21',
                            'premium individual+2 children age 30',
                            'premium individual+2 children age 40',
                            'premium individual+2 children age 50',
                            'premium individual+3 children age 21',
                            'premium individual+3 children age 30',
                            'premium individual+3 children age 40',
                            'premium individual+3 children age 50'
                        ],
                        "description": "The type of (coverage) they need and the premium based on the makeup of the beneficiaries of the plan",
                    }
                },
                "required": ["state_code", "county_name", "price_insensitive", "child_only_offering"],
            },
        }
    ]
    response = openai.ChatCompletion.create(
        model=chat_model,
        messages=messages,
        functions=functions,
        max_tokens = 500,
        temperature = 0.3,
        presence_penalty = 0.2,
        frequency_penalty = 1,
        function_call="auto",  # auto is default, but we'll be explicit
    )
    non_function_call_response = response.copy()
    response_message = response["choices"][0]["message"]

    # Step 2: check if GPT wanted to call a function
    if response_message.get("function_call"):
        # Step 3: call the function
        # Note: the JSON response may not always be valid; be sure to handle errors
        available_functions = {
            "create_contexts": create_contexts,
        }  # only one function in this example, but you can have multiple
        function_name = response_message["function_call"]["name"]
        function_to_call = available_functions[function_name]
        function_args = json.loads(response_message["function_call"]["arguments"])
        # print(function_args)
        function_response = function_to_call(
            state_code=function_args.get("state_code"),
            county_name=function_args.get("county_name"),
            price_insensitive=function_args.get("price_insensitive"),
            child_only_offering=function_args.get("child_only_offering"),
            additional_cols=function_args.get("additional_cols")
        )
        
        # print(function_response)

        # Step 4: send the info on the function call and function response to GPT
        messages.append(response_message)  # extend conversation with assistant's reply
        messages.append(
            {
                "role": "function",
                "name": function_name,
                "content": function_response,
            }
        )  # extend conversation with function response
        second_response = openai.ChatCompletion.create(
            model="gpt-3.5-turbo-16k-0613", # "gpt-3.5-turbo-0613",
            messages=messages,
            # max_tokens = 500,
            temperature = 0.2,
            
        )  # get a new response from GPT where it can see the function response
        
        # print(f'{num_tokens_from_messages(second_response)} prompt tokens counted by num_tokens_from_messages()')
        # num_tokens = second_response["usage"]["prompt_tokens"]
        return second_response
    
    else:
        # print(f'{num_tokens_from_messages(response_message)} prompt tokens counted by num_tokens_from_messages()')
        # num_tokens = response_message["usage"]["prompt_tokens"]
        return non_function_call_response

def clean_messages(messages, clean_role = "function"):
    clean_messages = []
    for  m in messages:
        for k,v in m.items():
            if k=='role' and v!="function":
                clean_messages.append(m)
    return clean_messages

def summarize_message(messages, k = 4): #k=how many last conversations to keep #This is so much better using LangChain's ConversationSummaryBufferMemory  etc.
    messages_no_function = clean_messages(messages=messages)
    
    combined_messages = "\n".join([f"{m['role']}: {m['content']}" for m in messages_no_function[1:]]) # removes system message before summarizing
    summarizer_system_prompt = [{
        "role": "system",
        "content": """You are a helpful summarizer."""
    }]
    summarizer_user_prompt={"role": "user", 
            "content": """Summarize the following conversation. Make sure to remember the customer's dental plan preferences, family makeup, and dental history in your summary. Take note of each plan and summarize them too. Here is an example of a conversation history between the AI and a broker:\n
            ### conversation history:\n
                user: Hello
                assistant: Hi there! How can I assist you today?
                user: I'm looking for some dental insurance plans for me and my three kids. 
                assistant: Great! I can definitely help you with that. To get started, could you please provide me with the following information:

                1. Which state and county do you reside in?
                2. Are you price conscious when it comes to dental insurance? Do you prefer low-cost plans or are you willing to pay more for additional benefits?
                3. What type of plan are you looking for? Do you prefer a PPO, HMO, EPO, or indemnity plan?
                4. Are you looking for plans that cover both adults and children, or are you specifically interested in child-only plans?
                5. Is there any specific coverage or premium information you would like to know about?

                Once I have these details, I'll be able to find the most suitable dental insurance plans for you and your kids.
                user: Sure. I live in charlotte florida. I am ok with a little more expensive plan. What's a PPO, hmo etc? 
                assistant: Thank you for providing your location and your preference for a slightly more expensive plan. Let me explain the different types of dental insurance plans for you:

                1. PPO (Preferred Provider Organization): A PPO plan offers flexibility with a network of preferred providers. You have the freedom to choose any dentist, but you can enjoy cost savings when you visit in-network dentists. PPO plans usually have higher premiums but provide a wider range of options.

                2. HMO (Health Maintenance Organization): An HMO plan requires you to select a primary dentist within its network. All dental care is coordinated through this primary dentist. HMO plans generally have lower premiums but offer a more limited choice of dentists.

                3. EPO (Exclusive Provider Organization): An EPO plan gives you the freedom to choose any dentist or specialist for your dental care. There is no requirement to select a primary dentist, and you can visit any dental provider you prefer, both in-network and out-of-network.

                4. Indemnity (Fee-for-Service): Indemnity plans offer the freedom to see any dentist and provide reimbursement for a portion of the covered expenses. These plans often have higher premiums but offer more flexibility in choosing dentists.

                Now that you have an understanding of these plan types, please let me know which one you prefer, or if you have any more questions about them.
                user: Ok, thanks. I guess I am ok with a PPO or EPO plan then.
                assistant: Great! Now that we have that cleared up, let's move on to the next question.

                Are you looking for dental insurance plans that cover both adults and children, or are you specifically interested in child-only plans? This will help me narrow down the options for you and your three kids.
                user: For myself, my wife, and 3 kids.
                assistant: Got it! So you're looking for dental insurance plans that cover both adults and children. Thank you for clarifying.

                Now, before we proceed, I have a couple more questions to better understand your preferences:

                1. Are you primarily concerned about routine dental services, basic dental care, major dental care, or orthodontia coverage?
                2. How price sensitive are you? Are you willing to pay higher premiums for more comprehensive coverage, or are you looking for a more budget-friendly option?

                Once I have this information, I'll be able to find the most suitable dental insurance plans for you and your family.
                user: I am ok with paying more as long as the plan is comprehensive enough.
                assistant: None
                assistant: Based on your preferences, I have found a couple of dental insurance plans that may be suitable for you and your family:

                1. Humana Dental Smart Choice - High (PPO):
                - Issuer Name: Humana
                - Child Only Offering: Allows adult and child-only
                - Customer Service Phone Number: 1-877-720-4854
                - Plan Brochure URL: [Humana Dental Smart Choice - High](http://apps.humana.com/marketing/documents.asp?file=4718311)
                - Coverage Details:
                    - Routine Dental Services - Adult: Yes
                    - Basic Dental Care - Adult: Yes
                    - Major Dental Care - Adult: Yes
                    - Orthodontia - Adult: Not covered
                    - Dental Check-up for Children: Yes
                    - Basic Dental Care - Child: Yes
                    - Major Dental Care - Child: Yes
                    - Orthodontia - Child: Yes

                2. Best Dental Advantage - Gold (PPO):
                - Issuer Name: Best Life
                - Child Only Offering: Allows adult and child-only
                - Customer Service Phone Number: 1-949-253-4080
                - Plan Brochure URL: [Best Dental Advantage - Gold](https://www.bestlife.com/fl/current/fl_bestone_dental_advantage-gold_plan.pdf)
                - Coverage Details:
                    - Routine Dental Services - Adult: Yes
                    - Basic Dental Care - Adult: Yes
                    - Major Dental Care - Adult: Yes
                    - Orthodontia - Adult: Not covered
                    - Dental Check-up for Children: Yes
                    - Basic Dental Care - Child: Yes
                    - Major Dental Care - Child: Yes
                    - Orthodontia - Child: Yes

                3. Best Dental Plus - Gold (PPO):
                - Issuer Name: Best Life
                - Child Only Offering: Allows adult and child-only
                - Customer Service Phone Number: 1-949-253-4080
                - Plan Brochure URL: [Best Dental Plus - Gold](https://www.bestlife.com/fl/current/fl_bestone_dental_plus-gold_plan.pdf)
                - Coverage Details:
                    - Routine Dental Services - Adult: Yes
                    - Basic Dental Care - Adult: Yes
                    - Major Dental Care - Adult: Yes
                    - Orthodontia - Adult: Not covered
                    - Dental Check-up for Children: Yes
                    - Basic Dental Care - Child: Yes
                    - Major Dental Care - Child: Yes
                    - Orthodontia - Child: Yes

                These plans offer comprehensive coverage for routine dental services, basic dental care, major dental care, and orthodontia for children. The Humana Dental Smart Choice - High plan also covers orthodontia for adults.

                Please review the plan details and brochures to get more information about the coverage, costs, and any limitations or exclusions. If you have any specific questions about these plans or need further assistance, feel free to ask! \n
            # Here's the summary:\n\n
            ### Summary:
            In the conversation between the customer and the dental insurance AI broker, the customer expressed the need for dental insurance plans for themselves and their three children. They disclosed their location as Charlotte, Florida, and their willingness to consider slightly more expensive plans. The AI broker provided explanations of different plan types, including PPO, HMO, EPO, and indemnity, and received a preference for PPO or EPO plans.

            The conversation continued with the customer's preference for plans covering both adults and children. The AI broker then inquired about the customer's specific coverage needs and their price sensitivity. The customer expressed a willingness to pay more for comprehensive coverage.

            Based on this information, the AI broker recommended three dental insurance plans: Humana Dental Smart Choice - High (PPO), Best Dental Advantage - Gold (PPO), and Best Dental Plus - Gold (PPO). These plans offer comprehensive coverage for routine dental services, basic dental care, major dental care, and orthodontia for children.

            The AI broker encouraged the customer to review plan details and brochures for further information and assured assistance with recommending the best plan. The summary captures the customer's dental plan preferences, family makeup, dental history, and the available plan options. \n\n

            # Here's is the current conversation history:\n\n
            # ### conversation history:\n{}"\n\n
            # Here's the summary:\n\n""".format(combined_messages)
            
    }
    # print(summarizer_user_prompt)
    summarizer_system_prompt.append(summarizer_user_prompt)
    # print(summarizer_system_prompt)
    summarized_response = openai.ChatCompletion.create(
        model="gpt-4", # allows for longer window (e.g. 'chat history')
        messages=summarizer_system_prompt,
        # max_tokens = 300,
        temperature = 0.3,
        
    ) 
    # Extract the summary from the response
    summary = summarized_response["choices"][0]["message"]["content"]
    
    # So our new message takes the current list of chats, keeps the last k chats, adds system message the beginning and adds summary at the end: see here from user cesidarocha - https://tinyurl.com/5b6pbrxc
    new_message = create_system_prompt()
    for d in messages_no_function[1:][-k:]:
        new_message.append(d) # keep the last k conversations including the assistant's response. 
    
    summary_dict = {"role": "assistant", "content": f"conversation_summary: {summary}"}
    new_message.append(summary_dict)
    
    return new_message


if "messages" not in st.session_state:
    st.session_state.messages = create_system_prompt()
    

if "num_tokens" not in st.session_state:
    st.session_state.num_tokens = 0

for message in st.session_state.messages:
    if message["role"] != "system" and message["role"] != "function":
        with st.chat_message(message["role"]):
            st.markdown(message["content"])
        
# Display assistant response in chat message container
def generate_reply(query, messages, num_tokens = 0):
        if chat_model == 'gpt-4-0613':
                model_num_tokens = 8192
                k = 5 # how many previous conversations to keep
        elif chat_model == 'gpt-3.5-turbo-0613':
                k = 3
                model_num_tokens = 4097
                
        if messages == None and len(messages) == 0:
                messages = []
                messages.append(create_system_prompt()[0])
        else:
                messages.append({"role": "user", "content": query})
                response_message = run_conversation(messages)
                assistant_message = response_message["choices"][0]["message"]
                messages.append(assistant_message)
                # message_content = response_message["choices"][0]["message"]['content'].replace("\n", " ")
                num_tokens = response_message["usage"]["total_tokens"]
                
        summarized_messages = None
        
        summarize = True # idea from alijakhar to keep stay in the conversation and not run out of tokens: https://tinyurl.com/2r9ncykb
        while summarize:
                if num_tokens >= int(0.7 * model_num_tokens) and len(messages) > 0:
                        messages = summarize_message(messages, k=k)
                        summarize = False
                        break
                break
        # pretty_print_conversation(messages)
        
        return num_tokens, messages, assistant_message
    
import time

def typing_indicator():
    st.write("Pearl is thinking...")
    # st.timeout(2000)
    st.empty()  # Clears the typing indicator after a few seconds


if prompt := st.chat_input("Let's chat about affordable dental plans!"):
    # st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)
            
    with st.chat_message("assistant"):
        typing_indicator()
        print(st.session_state.messages)
        
        st.session_state.num_tokens, st.session_state.messages, assistant_response = generate_reply(prompt, st.session_state.messages, st.session_state.num_tokens)
        message_placeholder = st.empty()
        full_response = ""
        # Simulate stream of response with milliseconds delay

        for chunk in assistant_response['content'].split():
            if "conversation_summary" not in assistant_response and assistant_response != None and assistant_response !="None" and assistant_response !="none":
                full_response += chunk + " "
                time.sleep(0.1)
                # Add a blinking cursor to simulate typing
                message_placeholder.markdown(full_response + "▌")
        message_placeholder.markdown(full_response)
    # Add assistant response to chat history
    # st.session_state.messages.append({"role": "assistant", "content": full_response})
# print(st.session_state)

st.sidebar.write(f"""
    ### Description
    This application is a Generative AI proof-of-concept. It features an AI chatbot assistant that can intelligently hold long form conversations with dental insurance seekers. It is a more intelligent alternative (or add-on) to the traditional 'plan compare'-type experiences. **Note** that this is ***not production-grade*** but a simple deployment for a demonstration. So feel free to break it 😉
    
    ---
    
    ### App info

    **App name**: *Pearl - Your Dental Plan Assistant*

    **App details**: The application uses stand-alone dental plan (SADP) data from the federal exchange accessible through [Healthcare.gov](https://www.healthcare.gov/dental-plan-information-2023/). The AI chatbot is named '**Pearl**' (get it?). The bot is given access any of the 200+ unique plans in this dataset. It can surface matching plans, recommend the best plans based on customer needs, simplify insurance terms, etc.
    
    **Dataset details**:  
        - Dataset has over 56k records representing 56k counties in the U.S. There are 33 unique states represented in this dataset.  
        - Connecticut, Idaho, Kentucky, Maine, Maryland, Massachusetts, Minnesota, Nevada, New Jersey, New Mexico, New York, Pennsylvania, Rhode Island, Vermont, Washington, and D.C. (16 states) do not have plans in this dataset. These states have state-based marketplaces or insurance exchanges.  
        - There are 283 unique plans in this dataset. 
    
    **Dataset source**: [Healthcare.gov](https://www.healthcare.gov/dental-plan-information-2023/)

    ---

    **How do I make something like this?**

    Contact the Delta Dental of Washington (DDWA) IT/BI Data Science team via [Elijah Adeoye](mailto:eadeoye@deltadentalwa.com)!
    """)
