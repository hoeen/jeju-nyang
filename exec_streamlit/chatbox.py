from langchain_community.llms import LlamaCpp
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain
from langchain.callbacks.manager import CallbackManager
from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler # 출력을 스트리밍하는 데 사용

import streamlit as st
from streamlit_chatbox import *
import time
import simplejson as json
from item_base_rec import get_recommendations_for_item
import os
#######

#classify

template = """
You're an LLM that discerns question intentions based on user queries. Below, I've defined an 'intent' list along with brief explanations:

- Intent: Recommend travel places
  Explanation: This corresponds to questions where users seek recommendations for travel destinations.

Now, classify which intent the user's question corresponds to from the list above. If it doesn't fit any, output 'etc'.


Question: {question}
Response: 
"""



prompt = PromptTemplate(template=template, input_variables=["question"])

callback_manager = CallbackManager([StreamingStdOutCallbackHandler()])

llm = LlamaCpp(
                model_path="models/llama-2-7b-chat.gguf.q3_K_S.bin",
                input={"temperature": 0.1,
                       "max_length": 2000,
                       "top_p": 1},
                callback_manager=callback_manager,
                verbose=True,
                )
#ver1
llm_chain_classify = LLMChain(prompt=prompt, llm=llm)



##############
#etc
template = """
you are an answer giver. Let's talk in step by step.

question: {question}
response: 

"""


prompt = PromptTemplate(template=template, input_variables=["question"])

callback_manager = CallbackManager([StreamingStdOutCallbackHandler()])

llm = LlamaCpp(
                model_path="models/llama-2-7b-chat.gguf.q3_K_S.bin",
                input={"temperature": 0.5,
                       "max_length": 2000,
                       "top_p": 1},
                callback_manager=callback_manager,
                verbose=True,
                )
#ver1
llm_chain_etc = LLMChain(prompt=prompt, llm=llm)


#############
chat_box = ChatBox()

with st.sidebar:
    st.subheader('start to chat using streamlit')
    streaming = st.checkbox('streaming', True)
    in_expander = st.checkbox('show messages in expander', True)
    show_history = st.checkbox('show history', False)

    st.divider()

    btns = st.container()

    file = st.file_uploader(
      "chat_history.json",
      type=["json"]
    )

    # Clear history button
    if st.button("clear_history"):
        chat_box.init_session(clear=True)
        st.experimental_rerun()

    # Load history from file (optional)
    if file is not None:
        try:
            data = json.load(file)
            chat_box.from_dict(data)
        except json.JSONDecodeError:
            st.error("Error loading chat history. Please ensure it's valid JSON.")



chat_box.init_session()
chat_box.output_messages()

if query := st.chat_input('input your question here'):
    chat_box.user_say(query)
    if streaming:
#         response = llm_chain_classify.run(query) #query -> llm_chain_1
        
#         if 'recommend' in response:
            #generator = llm_chain_rec.run(query) #query ->function2
#         generator=get_recommendations_for_item('배고파','sim_mat_240403.pkl','filtered_data_240403.csv' )
            #product_id, sim_mat_path, data_path
#         else:
        generator = llm_chain_etc.run(query)
        
        elements = chat_box.ai_say(
            [
                # you can use string for Markdown output if no other parameters provided
                Markdown("thinking", in_expander=in_expander,
                         expanded=True, title="answer"),
                Markdown("", in_expander=in_expander, title="references"),
            ]
        )
        time.sleep(1)
        text = ""
        for x, docs in enumerate(list(generator)):
            text += docs
            chat_box.update_msg(text, element_index=0, streaming=True)
        # update the element without focus
        chat_box.update_msg(text, element_index=0, streaming=False, state="complete")
        chat_box.update_msg("\n\n".join(docs), element_index=1, streaming=False, state="complete")
    else:
        text, docs = llm_chain.run(query)
        chat_box.ai_say(
            [
                Markdown(text, in_expander=in_expander,
                         expanded=True, title="answer"),
                Markdown("\n\n".join(docs), in_expander=in_expander,
                         title="references"),
            ]
        )

# cols = st.columns(2)
# if cols[0].button('show me the multimedia'):
#     chat_box.ai_say(Image(
#         'https://tse4-mm.cn.bing.net/th/id/OIP-C.cy76ifbr2oQPMEs2H82D-QHaEv?w=284&h=181&c=7&r=0&o=5&dpr=1.5&pid=1.7'))
#     time.sleep(0.5)
#     chat_box.ai_say(
#         Video('https://sample-videos.com/video123/mp4/720/big_buck_bunny_720p_1mb.mp4'))
#     time.sleep(0.5)
#     chat_box.ai_say(
#         Audio('https://sample-videos.com/video123/mp4/720/big_buck_bunny_720p_1mb.mp4'))

# if cols[1].button('run agent'):
#     chat_box.user_say('run agent')
#     agent = FakeAgent()
#     text = ""

#     # streaming:
#     chat_box.ai_say() # generate a blank placeholder to render messages
#     for d in agent.run_stream():
#         if d["type"] == "complete":
#             chat_box.update_msg(expanded=False, state="complete")
#             chat_box.insert_msg(d["llm_output"])
#             break

#         if d["status"] == 1:
#             chat_box.update_msg(expanded=False, state="complete")
#             text = ""
#             chat_box.insert_msg(Markdown(text, title=d["text"], in_expander=True, expanded=True))
#         elif d["status"] == 2:
#             text += d["llm_output"]
#             chat_box.update_msg(text, streaming=True)
#         else:
#             chat_box.update_msg(text, streaming=False)

btns.download_button(
    "Export Markdown",
    "".join(chat_box.export2md()),
    file_name=f"chat_history.md",
    mime="text/markdown",
)

btns.download_button(
    "Export Json",
    chat_box.to_json(),
    file_name="chat_history.json",
    mime="text/json",
)

if btns.button("clear history"):
    chat_box.init_session(clear=True)
    st.experimental_rerun()


if show_history:
    st.write(chat_box.history)
