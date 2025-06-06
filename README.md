# 🍔 BurgerMate
## 알레르기 기반 햄버거 추천 챗봇 & 리트리버 성능 평가

<p align="center">
  <img src="/images/리트리버성능평가.png" alt="리트리버 성능 평가" width="70%">
</p>

**BurgerMate**는 햄버거 프랜차이즈의 알레르기 성분 정보를 기반으로 사용자의 알레르기 유형에 맞는 안전한 메뉴를 추천하는 **RAG 기반 AI 챗봇**입니다.  
또한 다양한 **리트리버 조합(Dense, Sparse, Ensemble)** 및 **벡터 DB(Pinecone, FAISS, Milvus)**를 비교 평가하고 시각화합니다.

---

## 🎥 데모 영상

<p align="center">
  <video controls width="80%">
    <source src="images/버거메이트시연영상.mov" type="video/mp4">
    Your browser does not support the video tag.
  </video>
</p>

---

## 📌 프로젝트 특징

- 🍔 햄버거 성분 CSV 기반 알레르기 필터링
- 🤖 Upstage `solar-pro` 기반 Chat LLM
- 🧠 Upstage `embedding-query` 임베딩 모델 활용
- 🔍 Retriever 비교:
  - Dense (Pinecone)
  - Sparse (BM25)
  - Ensemble (가중치 조절 가능)
- 📈 RAGAS 기반 context precision/recall 정량 평가
- 📊 Matplotlib 기반 시각화

---

## 🧠 임베딩 모델 성능 평가

<p align="center">
  <img src="/images/임베딩모델성능평가.png" alt="임베딩 모델 성능 평가" width="60%">
</p>

---

## 📊 벡터 DB 기반 성능 비교

<p align="center">
  <img src="/images/벡터DB성평가.jpg" alt="벡터 DB 성능 비교" width="60%">
</p>

---

## ❓ 대표 질문 예시

```text
- 우유 알레르기로부터 안전한 KFC 햄버거 메뉴를 추천해줘.
- 꽃가루 알레르기가 있는데, 어떤 햄버거를 먹는 것이 안전할까?
- 난류와 땅콩이 들어가지 않은 맥도날드 햄버거는?
- 조개 알레르기를 유발하지 않는 햄버거를 추천해줘. 롯데리아는 제외.
