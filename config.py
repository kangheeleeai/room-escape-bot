import streamlit as st

# 프로젝트 설정
PROJECT_ID = "room-escape-chatbot" 
EMBEDDING_MODEL_NAME = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
LOCAL_CACHE_DIR = "./model_cache"

# API Keys (Streamlit Secrets에서 로드, 없으면 기본값/None)
# 실제 배포 시에는 .streamlit/secrets.toml 에 저장해야 함
GROQ_API_KEY = st.secrets.get("GROQ_API_KEY", "")
TAVILY_API_KEY = st.secrets.get("TAVILY_API_KEY", "")