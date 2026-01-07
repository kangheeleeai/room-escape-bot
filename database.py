import os
import streamlit as st
import firebase_admin
from firebase_admin import credentials, firestore

# ==============================================================================
# [라이브러리 안전 로딩 (Safe Import)]
# ==============================================================================
Vector = None
DistanceMeasure = None
FieldFilter = None

# 1. FieldFilter
try:
    from google.cloud.firestore import FieldFilter
except ImportError:
    pass

# 2. Vector
try:
    from google.cloud.firestore import Vector
except ImportError:
    try:
        from google.cloud.firestore_v1.vector import Vector
    except ImportError:
        pass

# 3. DistanceMeasure
try:
    from google.cloud.firestore_v1.base_vector_query import DistanceMeasure
except ImportError:
    pass

# 4. Fallback Classes
if Vector is None:
    class Vector:
        def __init__(self, val): self.value = val

if DistanceMeasure is None:
    class DistanceMeasure:
        COSINE = "COSINE"

# ==============================================================================
# [Firebase 초기화]
# ==============================================================================
@st.cache_resource
def init_firebase():
    """
    Firebase 초기화: Streamlit Secrets를 우선 사용하고, 없으면 로컬 파일을 찾습니다.
    """
    try:
        if not firebase_admin._apps:
            # 1. Streamlit Secrets (배포 환경)
            if "firebase" in st.secrets:
                cred_info = dict(st.secrets["firebase"])
                cred = credentials.Certificate(cred_info)
                firebase_admin.initialize_app(cred)
            # 2. 로컬 파일 (개발 환경)
            elif os.path.exists("serviceAccountKey.json"):
                cred = credentials.Certificate("serviceAccountKey.json")
                firebase_admin.initialize_app(cred)
            else:
                return None
        return firestore.client()
    except Exception as e:
        st.error(f"Firebase 초기화 실패: {e}")
        return None