import os
import streamlit as st
from sentence_transformers import SentenceTransformer
from config import EMBEDDING_MODEL_NAME, LOCAL_CACHE_DIR

try:
    from sentence_transformers import SentenceTransformer
    EMBEDDING_AVAILABLE = True
except ImportError:
    EMBEDDING_AVAILABLE = False

@st.cache_resource
def load_embed_model():
    if not EMBEDDING_AVAILABLE: 
        st.error("sentence-transformers 라이브러리가 필요합니다.")
        return None
    try:
        # 배포 환경 캐시 문제 방지
        os.makedirs(LOCAL_CACHE_DIR, exist_ok=True)
        model = SentenceTransformer(
            EMBEDDING_MODEL_NAME, 
            cache_folder=LOCAL_CACHE_DIR,
            model_kwargs={"use_safetensors": True}
        )
        return model
    except Exception as e:
        st.error(f"임베딩 모델 로드 실패: {e}")
        return None