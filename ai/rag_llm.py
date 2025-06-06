import os
import getpass
import warnings
import logging
import json
import csv
from langchain.schema import Document
from langchain.document_loaders import WebBaseLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma
from langchain_upstage import UpstageEmbeddings, ChatUpstage
from langchain_core.prompts import ChatPromptTemplate
from langchain.chains import ConversationChain
from langchain.memory import ConversationBufferMemory
from langchain_upstage import UpstageDocumentParseLoader
from langchain_core.output_parsers import StrOutputParser
from langchain_pinecone import PineconeVectorStore
from pinecone import Pinecone, ServerlessSpec
from langchain.docstore.document import Document
from langchain_community.vectorstores import FAISS
from langchain_milvus import Milvus
from datasets import Dataset
from ragas.metrics import context_precision, context_recall
from ragas import evaluate
import matplotlib.pyplot as plt
import numpy as np

# Retriever 평가용
from datasets import Dataset
from langchain_community.retrievers import BM25Retriever
from langchain.retrievers import EnsembleRetriever

llm = ChatUpstage(model='solar-pro')

questions = [
    "우유 알레르기로부터 안전한 KFC 햄버거 메뉴를 추천해줘.",
    "나는 꽃가루 알레르기가 있는데, 어떤 햄버거를 먹는 것이 안전할까?",
    "난류와 땅콩이 들어가지 않은 맥도날드 햄버거는 어떤 것이 있는지 알려줘.",
    "새우 알러지 환자를 위한 메뉴를 추천해줘.",
    "방금 추천한 불고기버거랑 어울리는 사이드메뉴도 추천해줘.",
    "밀가루 알레르기가 있는데, 내가 먹을 수 있는 메뉴는 어떤 것이 있는지 알려줘.",
    "조개 알레르기를 유발하지 않는 햄버거를 추천해줘. 롯데리아 버거는 제외하고.",
    "롯데리아의 핫크리스피버거가 어떤 알레르기를 유발하는지 말해줘.",
    "채식주의자와 함께 먹을 수 있는 햄버거는 어떤 것들이 있지?",
    "맘스터치에서 토마토가 들어가지 않은 햄버거와 사이드메뉴, 소스를 같이 추천해줘."
]

def fill_data(data, question, retr):
    results = retr.invoke(question)
    context = [doc.page_content for doc in results]

    # chain = prompt | llm | StrOutputParser()
    # answer = chain.invoke({"history": [], "context": context, "input": question})

    data["question"].append(question)
    data["answer"].append("")
    data["contexts"].append(context)
    data["ground_truth"].append("")

def ragas_evalate(dataset):
    result = evaluate(
        dataset,
        metrics=[
            context_precision,
            context_recall
        ],
        llm=llm,
        embeddings=UpstageEmbeddings(model="embedding-query"),
    )
    return result

# 글로벌 변수 선언
dense_dataset = None
sparse_dataset = None
ensemble_dataset5_5 = None
ensemble_dataset7_3 = None
dense_score = None
sparse_score = None
ensemble_score5_5 = None
ensemble_score7_3 = None
ensemble_score3_7 = None


class BurgerChatbot:
    def __init__(self):
        self.data = 'burger.csv'
        self.key_file = "upstage_api_key.txt"
        self.question = ""
        self.prev_chat = ""
        # Update memory to include return messages
        self.memory = ConversationBufferMemory(
            memory_key="chat_history",
            input_key="input",  # Changed from 'question' to 'input'
            return_messages=True
        )
        self.llm = None
        self.chain = None
        self.retriever = None
        self.pinecone_api_key = None

    def _load_api_key(self):
        """API 키 로드"""
        warnings.filterwarnings("ignore")
        try:
            if "UPSTAGE_API_KEY" not in os.environ or not os.environ["UPSTAGE_API_KEY"]:
              os.environ["UPSTAGE_API_KEY"] = getpass.getpass("Enter your Upstage API key: ")

            if not os.getenv("PINECONE_API_KEY"):
                os.environ["PINECONE_API_KEY"] = getpass.getpass("Enter your Pinecone API key: ")
                pinecone_api_key = os.environ.get("PINECONE_API_KEY")
            print("API key has been set successfully.")
            self.retriever = self.init()
        except:
            print("Something wrong with your API KEY. Check your API Console again.")


    # PDF CSV -> 청크로
    def chunkify(self):
        """PDF 파일 읽어와 파싱 및 스플릿"""
        splits = []

        # CSV 파일 처리
        csv_docs = self.load_csv(self.data)
        text_splitter500 = RecursiveCharacterTextSplitter(
            chunk_size=500,
            chunk_overlap=50
        )
        # CSV 문서들을 Document 객체로 변환 후 split
        CSV_document_objects = [Document(page_content=doc) for doc in csv_docs]
        splits.extend(text_splitter500.split_documents(CSV_document_objects))

        print("Total Splits:", len(splits), ", Splits type:", type(splits))
        return splits

    def load_csv(self, csv_file):
        """CSV 파일을 문서 형식으로 로드"""
        documents = []
        with open(csv_file, mode='r', encoding='cp949') as file:
            reader = csv.reader(file)
            for row in reader:
                # 각 행을 문서로 처리 (예: CSV 행의 각 항목을 하나의 문서로 취급)
                document = " ".join(row)  # CSV 행을 하나의 문자열로 결합
                documents.append(document)
        return documents

    # init_pdf_datas -> 청크를 벡터디비에 저장하고 리트리버 생성
    def init(self):
        global pinecone_score,dense_score, sparse_score, ensemble_score5_5, ensemble_score7_3
        global dense_dataset, sparse_dataset, ensemble_dataset5_5, ensemble_dataset7_3


        chunk = self.chunkify()

        #"""
        # 4. 청크로 벡터스토어 생성(파인콘)
        index_name = "retriever-demo"
        pc = Pinecone(api_key=self.pinecone_api_key)

        # create new index
        # pinecone setup
        if index_name not in pc.list_indexes().names():
            pc.create_index(
                name=index_name,
                dimension=4096,
                metric="cosine",
                spec=ServerlessSpec(cloud="aws", region="us-east-1"),
            )

        pinecone_vectorstore = PineconeVectorStore.from_documents(
            chunk, UpstageEmbeddings(model="embedding-query"), index_name="retriever-demo"
        )

        retriever = pinecone_vectorstore.as_retriever(
            search_type= 'mmr', # default : similarity(유사도) / mmr 알고리즘
            search_kwargs={"k": 3} # 쿼리와 관련된 chunk를 3개 검색하기 (default : 4)
        )

        # Dense Retriever
        dense_data = {
            "question": [],
            "answer": [],
            "contexts": [],
            "ground_truth": [],
        }

        for question in questions:
            fill_data(dense_data, question, retriever)
        dense_dataset = Dataset.from_dict(dense_data)
        dense_score = ragas_evalate(dense_dataset)

        # Sparse Retriever
        bm25_retriever = BM25Retriever.from_documents(
            documents=chunk
        )

        sparse_data = {
            "question": [],
            "answer": [],
            "contexts": [],
            "ground_truth": [],
        }

        for question in questions:
            fill_data(sparse_data, question, bm25_retriever)
        sparse_dataset = Dataset.from_dict(sparse_data)
        sparse_score = ragas_evalate(sparse_dataset)

        # Ensemble Retriever 5:5
        ensemble_retriever = EnsembleRetriever(
            retrievers=[retriever, bm25_retriever],
            weights=[0.5, 0.5]  # 각 Retriever 별 가중치를 조절 가능

        )

        ensemble_data = {
            "question": [],
            "answer": [],
            "contexts": [],
            "ground_truth": [],
        }

        for question in questions:
            fill_data(ensemble_data, question, ensemble_retriever)

        ensemble_dataset5_5 = Dataset.from_dict(ensemble_data)
        ensemble_score5_5 = ragas_evalate(ensemble_dataset5_5)

        # Ensemble Retriever 7:3
        ensemble_retriever = EnsembleRetriever(
            retrievers=[retriever, bm25_retriever],
            weights=[0.7, 0.3]  # 각 Retriever 별 가중치를 조절 가능
        )

        ensemble_data = {
            "question": [],
            "answer": [],
            "contexts": [],
            "ground_truth": [],
        }

        for question in questions:
            fill_data(ensemble_data, question, ensemble_retriever)

        ensemble_dataset7_3 = Dataset.from_dict(ensemble_data)
        ensemble_score7_3 = ragas_evalate(ensemble_dataset7_3)

        # Ensemble Retriever 3:7
        ensemble_retriever = EnsembleRetriever(
            retrievers=[retriever, bm25_retriever],
            weights=[0.3, 0.7]  # 각 Retriever 별 가중치를 조절 가능
        )

        ensemble_data = {
            "question": [],
            "answer": [],
            "contexts": [],
            "ground_truth": [],
        }

        for question in questions:
            fill_data(ensemble_data, question, ensemble_retriever)

        ensemble_dataset3_7 = Dataset.from_dict(ensemble_data)
        ensemble_score3_7 = ragas_evalate(ensemble_dataset3_7)

        # 파인콘 Dense Retriever 리턴
        return retriever


    def create_answer(self, question):
        # retriever에서 question에 적합한 청크 추출해 context에 저장
        context = self.retriever.invoke(question)

        """사용자의 질문과 이전 대화 기록을 포함하여 답변을 생성"""
        if self.chain is None:
            prompt = ChatPromptTemplate.from_messages([
                (
                    "system",
                    """너는 프랜차이즈 햄버거의 알레르기 성분 정보를 기반으로 사용자 요청에 따라
                        적합하고 안전한 메뉴를 추천하는 인공지능 챗봇이야.너의 주요 임무는 사용자의 질문에
                        예의 바르게 답변하며, 사용자 맞춤형 정보를 정확히 제공하는 거야.
                        너가 추천할 수 있는 햄버거 프랜차이즈의 브랜드는
                        맥도날드. 노브랜드버거, 프랭크버거, KFC 오직 5개의 브랜드야.
                        만약 사용자가 브랜드를 특정하지 않았다면 5개의 브랜드의 데이터를 적절하게 사용해서 구체적으로 답변해야 해.
                        CONTEXT 데이터와 사용자의 질문, 이전 대화 내역을 종합하여 다음 규칙에 따라 답변을 생성해줘:
                        1. 주어진 CONTEXT의 정보와 사용자 정보만을 기반으로 답변해. 문서에 없는 정보, 알레리그 정보가 없거나 식별할 수 없는 질문에 대해선는
                            '죄송합니다. 해당 정보가 부족하여 답변이 드리기 어렵습니다."라고 답변해
                            또한 CONTEXT에서 각 메뉴의 알레르기 성분을 확인하여 사용자가 피해야 할 음식을 필터링해.
                        2. 사용자가 제공한 알레르기 정보와 선호도를 바탕으로 가장 적합한 메뉴를 추천해.
                        3. 메뉴 성분이나 추가 요청사항에 따라 필요한 경우 사용자에게 질문을 던져 정보를 보완해.
                        4. 사용자가 이해하기 쉽게 작성해
                        5. 사용자의 질문의 의도를 정확히 파악하고, 불명확한 경우 추가 질문으로 의도를 명확히 해.
                    ---
                    CONTEXT:
                    {context}
                    ---
                    Chat History: {chat_history}
                    Human: {input}
                    Assistant:"""
                )
            ])

            self.chain = prompt | self.llm | StrOutputParser()

        # Get chat history from memory
        chat_history = self.memory.load_memory_variables({})["chat_history"]

        # Generate response
        response = self.chain.invoke({
            "context": context,
            "chat_history": chat_history,
            "input": question
        })

        # Save the interaction to memory
        self.memory.save_context(
            {"input": question},
            {"output": response}
        )

        return response

def main():
    chatbot = BurgerChatbot()
    chatbot._load_api_key()

    methods = ['Dense', 'Sparse', 'Ensemble 3:7', 'Ensemble 5:5', 'Ensemble 7:3']
    precision = [np.nanmean(dense_score['context_precision']), np.nanmean(sparse_score['context_precision']), np.nanmean(ensemble_score3_7['context_precision']) , np.nanmean(ensemble_score5_5['context_precision']), np.nanmean(ensemble_score7_3['context_precision'])]
    recall = [np.nanmean(dense_score['context_recall']), np.nanmean(sparse_score['context_recall']), np.nanmean(ensemble_score3_7['context_recall']), np.nanmean(ensemble_score5_5['context_recall']), np.nanmean(ensemble_score7_3['context_recall'])]

    # 시각화
    fig, ax = plt.subplots()

    # Precision과 Recall을 나란히 보여주기 위해 bar width 설정
    bar_width = 0.35
    index = range(len(methods))

    # Precision과 Recall Bar 생성
    bar1 = ax.bar(index, precision, bar_width, label='Precision')
    bar2 = ax.bar([i + bar_width for i in index], recall, bar_width, label='Recall')

    # Label 및 제목 설정
    ax.set_xlabel('Retrieval Methods')
    ax.set_ylabel('Scores')
    ax.set_title('Precision and Recall for Different Retrieval Methods')
    ax.set_xticks([i + bar_width / 2 for i in index])
    ax.set_xticklabels(methods)
    ax.legend()

    # 그래프 출력
    plt.tight_layout()
    plt.show()



if __name__ == "__main__":
    main()