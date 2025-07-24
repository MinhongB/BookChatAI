####### 책봇 #######

from dotenv import load_dotenv
import os
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_chroma import Chroma
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
from langchain.schema.runnable import RunnablePassthrough, RunnableLambda
import gradio as gr
import re
import ast
import shutil

load_dotenv()

GEMINI_API_KEY = os.getenv('GEMINI_API_KEY')

### Gemini 모델 설정
llm_gemini = ChatGoogleGenerativeAI(model="gemini-2.0-flash", api_key=GEMINI_API_KEY)

####### Simple RAG #######

### Prompt
fewshot_template = """
책에 대해서 쉽고 친절하게 답변하는 인공지능입니다.
모든 답변은 한글로 하세요.
책과 관련된 정보 이외의 내용은 지어내지 마세요.
아래 예시의 말투를 참고하되, 정보는 주어진 context에서만 사용하세요.
답변의 최대 길이는 한글 기준 대략 180자 이내 정도로 제한하되 완성도 있는 문장으로 마무리해주고, 굳이 길게 대답할 필요가 없는 질문은 짧게 대답해도 됩니다.
참고하는 Vector DB의 자료가 영어이므로 한글로 주어지는 질문을 해당 자료의 영어 단어와 매칭해서 답변시 한글로 변환 후 알려주세요


예시:
Q: 플라톤이 말한 사랑은?
A: 플라톤이 말하는 사랑은 남성과 여성 모두를 포함하는 일반적인 개념이며, 감각적인 요소도 포함합니다. Symposium에서는 Aristophanes, Eryximachus, Agathon, Pausanias의 관점에서 사랑을 설명합니다.

Q: 플라톤이 말한 정의는?
A: 플라톤은 정의(justice)를 'dikaiou sunesis', 즉 '정의에 대한 이해'라고 표현했습니다. 또한, 정의는 음식에 비유되기도 하며, 소크라테스는 정의가 무엇을 누구에게 주는 것인지 묻습니다.

Q: 플라톤이 말한 정의는?
A: 플라톤은 정의(justice)를 이성이 감정에 우선하는 상태, 즉 이성이 감정을 통제하는 것으로 보았습니다. 또한, 소크라테스는 정의가 무엇을 누구에게 주는 것인지 질문합니다.

Q: 플라톤이 말한 사랑은?
A: 플라톤이 말하는 사랑(love)은 남성과 여성 모두를 포함하는 일반적인 개념이며, 감각적인 요소도 포함합니다. 또한, 플라톤은 사랑의 수수께끼가 인간과 자연 모두에 존재한다고 보았습니다.

Q: 플라토니 말하는 국가란?
A: 플라톤은 한 국가(nation)를 묘사할 때, 두 민족(two nations)에 대해 이야기합니다. 또한, 언어는 개인(individuals)이 아닌 국가(States)에 속한다고 언급합니다.

Q: 플라톤이 말한 사랑이란?
A: 플라톤이 묘사하는 사랑(love)은 남성과 여성 모두를 포함하는 일반적인 개념이며, 감각적인 요소도 포함합니다. 또한, 플라톤은 사랑의 본질에 대한 철학 전체를 다루고 있으며, 사랑에는 신비로움이 있다고 보았습니다.

Q: 플라토니 말한 두려움이란?
A: 플라톤은 그의 견해에 따르면, 소셜 라이프의 이상(ideals of social life)에 대해 지나치게 걱정했습니다. 또한, 플라톤은 영혼(soul)이 육체를 떠날 때 연기처럼 사라질까 봐 두려워했습니다.

Q: 플라톤이 말한 두려움이란?
A: 플라톤은 두 종류의 두려움(fear)을 구별합니다. 하나는 악(evil)에 대한 두려움이고, 다른 하나는 나쁜 평판(evil reputation)에 대한 두려움입니다. 또한, 플라톤은 두려움(fear) 또는 공포(terror)를 악의 예상(expectation of evil)으로 보았습니다.

Q: 플라톤이 말한 용기란?
A: 플라톤에게 용기(courage)는 도덕적 용기, 고통을 견디는 용기, 지적 능력을 극복하는 용기를 포함합니다. 또한, 소크라테스는 용기란 무엇인지 묻고, 쾌락과 고통 속에서 시험된 모든 종류의 용기에 대한 정의를 원했습니다.

Q: 플라톤에게 용기란?
A: 플라톤에게 용기(courage)는 도덕적 용기, 고통을 견디는 용기, 지적 능력을 극복하는 용기를 포함합니다. 또한, 소크라테스는 용기를 '지능'이라고 정의하기도 했습니다.

Q: 잘 사는 것이란 무엇인가?
A: 플라톤에 따르면 즐겁게 사는 것(live pleasantly)은 좋은 것이고, 불쾌하게 사는 것(live unpleasantly)은 나쁜 것입니다. 잘 사는 것(live well)은 행복하고 복된 삶을 의미하며, 정의롭게 사는 것(live justly)은 훌륭하고 명예롭게 사는 것과 같습니다.

Q: 힘들때는 어떻게 해야 하는가?
A: 힘든 시기(things get tough)에는 친구의 목소리를 듣거나 손을 잡는 것이 도움이 될 수 있습니다. 특히 난파, 패배, 실패 등의 상황에서 더욱 그렇습니다.

Q: 잘사는 삶이란?
A: 플라톤에 따르면 즐겁게 사는 것(live pleasantly)은 좋은 것이고, 불쾌하게 사는 것(live unpleasantly)은 나쁜 것입니다. 좋은 삶(good life)은 좋은 사람의 삶과는 다를 수 있습니다.

Q: 산다는게 뭘까?
A: 플라톤에 따르면 삶(live)은 행복(happiness)이나 불행(misery) 속에서 사는 것을 의미합니다. 또한, 정의와 모든 덕을 실천하는 것이 삶이라고 말합니다.

---------------------------------
아래의 context를 참고해 질문에 답변하세요.

Context:
{context}

Q: {question}
A:
"""


### Document loaders

from langchain_community.document_loaders import PyPDFDirectoryLoader

# './pdfs' 폴더 내의 모든 PDF 파일을 불러옵니다.
loader1 = PyPDFDirectoryLoader('/BookBot/data/plato')
data1 = loader1.load()

### Text Split
text_splitter = RecursiveCharacterTextSplitter(chunk_size=200, chunk_overlap=20)
all_splits1 = text_splitter.split_documents(data1)


### Embedding
embeddings_model = GoogleGenerativeAIEmbeddings(
    model="models/embedding-001", 
    google_api_key=GEMINI_API_KEY
)

### Vector stores - 벡터DB 재생성 방지 로직
persist_directory = '../vector_db/books'

def get_or_create_chroma(collection_name, splits):
    # Chroma 컬렉션이 이미 존재하면 from_existing, 아니면 from_documents
    try:
        db = Chroma(
            persist_directory=persist_directory,
            collection_name=collection_name,
            embedding_function=embeddings_model
        )
        # 실제로 컬렉션이 존재하는지 확인
        if len(db.get()["ids"]) > 0:
            return db
        else:
            # 컬렉션이 있지만 비어있으면 새로 생성
            raise Exception("컬렉션이 비어있음")
    except Exception:
        # 컬렉션이 없으면 새로 생성
        db = Chroma.from_documents(
            splits,
            embeddings_model,
            collection_name=collection_name,
            persist_directory=persist_directory,
            collection_metadata={'hnsw:space': 'cosine'},
        )
        return db

db_1 = get_or_create_chroma("plato", all_splits1)
# db_2 = get_or_create_chroma("plato", all_splits2)

### Retriever 생성
retriever_1 = db_1.as_retriever()
# retriever_2 = db_1.as_retriever()

def merged_retriever(query):
    results_1 = retriever_1.invoke(query)
    # results_2 = retriever_2.invoke(query)
    merged = []
    merged.extend(results_1)
    # merged.extend(results_2)
    return merged

merged_retriever_runnable = RunnableLambda(lambda question: {"context": merged_retriever(question), "question": question})

fewshot_prompt = ChatPromptTemplate.from_template(fewshot_template)

fewshot_chain = (
    merged_retriever_runnable
    | fewshot_prompt
    | llm_gemini
    | StrOutputParser()
)

import gradio as gr
import whisper
import openai

# Whisper 모델 로드
whisper_model = whisper.load_model("medium")

# OpenAI API KEY 설정 (환경변수나 직접 입력)
openai.api_key = "your-api-key"  # 본인 키로 교체

def translate_to_english(text):
    # OpenAI GPT를 이용한 번역 예시 (gpt-3.5-turbo)
    system_prompt = "Translate the following Korean question to natural English. Only return the translated sentence."
    response = openai.ChatCompletion.create(
        model="gpt-3.5-turbo",
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": text}
        ]
    )
    return response['choices'][0]['message']['content'].strip()

def speech_to_text(audio):
    if audio is None:
        return ""
    result = whisper_model.transcribe(audio)
    return result["text"]

def chatbot(text_input, audio_input):
    """
    질문을 영어로 번역해 chain에 전달, 답변은 한글로 출력
    """
    if text_input and text_input.strip() != "":
        original_query = text_input.strip()
    elif audio_input is not None:
        original_query = speech_to_text(audio_input)
    else:
        return "입력 없음", "텍스트 또는 음성 입력을 해주세요."

    # 1. 입력 질문을 영어로 번역
    try:
        english_query = translate_to_english(original_query)
    except Exception as e:
        return original_query, f"질문 번역 실패: {str(e)}"

    # 2. 영어 질문을 chain에 전달
    try:
        answer = fewshot_chain.invoke(english_query)
    except Exception as e:
        answer = f"에러 발생: {str(e)}"
    return original_query, answer

def create_demo():
    with gr.Blocks() as demo:
        gr.Markdown("# 책봇 : 플라톤 철학에 대해서 물어보세요")
        with gr.Row():
            with gr.Column(scale=1):
                text_input = gr.Textbox(label="텍스트 입력", placeholder="질문을 입력하세요")
            with gr.Column(scale=1):
                audio_input = gr.Audio(type="filepath", label="질문(음성 입력, 선택)")
        submit_btn = gr.Button("질문하기")
        question_out = gr.Textbox(label="입력된 질문(텍스트)", interactive=False)
        answer_out = gr.Textbox(label="챗봇 답변", interactive=False)

        submit_btn.click(
            chatbot,
            inputs=[text_input, audio_input],
            outputs=[question_out, answer_out]
        )
    return demo

if __name__ == "__main__":
    demo = create_demo()
    demo.launch(share=True, inbrowser=True)