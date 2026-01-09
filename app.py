import streamlit as st
import json
import os
import time
import copy
import logging # [NEW] 로깅 모듈 추가
from groq import Groq 
from tavily import TavilyClient
import numpy as np

# [Firebase]
import firebase_admin
from firebase_admin import credentials, firestore

# ==============================================================================
# [로깅 설정] (콘솔 + UI)
# ==============================================================================
# 1. 로거 설정 (Streamlit Cloud 콘솔용)
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(message)s')
logger = logging.getLogger(__name__)

# 2. UI 로그 함수 (화면 출력용)
def log_msg(message):
    """
    1. Streamlit Cloud 콘솔에 즉시 출력 (flush=True 효과)
    2. 세션 상태에 저장하여 UI 사이드바에 표시
    """
    # 콘솔 출력
    logger.info(message) 
    
    # UI 저장 (세션 초기화 확인)
    if "app_logs" not in st.session_state:
        st.session_state.app_logs = []
    
    # 로그 추가 (최신 로그가 위로 오도록 or 아래로 쌓이도록)
    timestamp = time.strftime("%H:%M:%S")
    st.session_state.app_logs.append(f"[{timestamp}] {message}")

# ==============================================================================
# [라이브러리 안전 로딩 (Safe Import)]
# ==============================================================================
Vector = None
DistanceMeasure = None
FieldFilter = None

try:
    from google.cloud.firestore import FieldFilter
except ImportError:
    pass

try:
    from google.cloud.firestore import Vector
except ImportError:
    try:
        from google.cloud.firestore_v1.vector import Vector
    except ImportError:
        pass

try:
    from google.cloud.firestore_v1.base_vector_query import DistanceMeasure
except ImportError:
    pass

if Vector is None:
    class Vector:
        def __init__(self, val): self.value = val

if DistanceMeasure is None:
    class DistanceMeasure:
        COSINE = "COSINE"

if FieldFilter is None:
    st.error("🚨 [Critical] Firestore 라이브러리 로드 실패.")
    st.stop()

# [Embedding]
try:
    from sentence_transformers import SentenceTransformer
    EMBEDDING_AVAILABLE = True
except ImportError:
    st.error("sentence-transformers 라이브러리가 필요합니다.")
    EMBEDDING_AVAILABLE = False

# ==============================================================================
# [설정]
# ==============================================================================
# API Keys (Secrets 사용 권장)
GROQ_API_KEY = st.secrets.get("GROQ_API_KEY", "")
TAVILY_API_KEY = st.secrets.get("TAVILY_API_KEY", "")

PROJECT_ID = "room-escape-chatbot" 
EMBEDDING_MODEL_NAME = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
LOCAL_CACHE_DIR = "./model_cache"

st.set_page_config(page_title="방탈출 AI 코난 (Hybrid)", page_icon="🕵️", layout="wide")

# ==============================================================================
# [초기화] Firebase & Model
# ==============================================================================
@st.cache_resource
def init_firebase():
    try:
        if not firebase_admin._apps:
            if "firebase" in st.secrets:
                cred_info = dict(st.secrets["firebase"])
                cred = credentials.Certificate(cred_info)
                firebase_admin.initialize_app(cred)
            elif os.path.exists("serviceAccountKey.json"):
                cred = credentials.Certificate("serviceAccountKey.json")
                firebase_admin.initialize_app(cred)
            else:
                return None
        return firestore.client()
    except Exception as e:
        st.error(f"Firebase 초기화 실패: {e}")
        return None

@st.cache_resource
def load_embed_model():
    if not EMBEDDING_AVAILABLE: return None
    try:
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

# ==============================================================================
# [Helper] 공통 정렬 로직
# ==============================================================================
def sort_candidates_by_query(candidates, user_query):
    if not candidates: return []
    query_text = user_query if user_query else ""
    
    if "안무서운" in query_text or "무섭지 않은" in query_text:
        candidates.sort(key=lambda x: (x['rating'], -x['fear']), reverse=True)
    elif "공포" in query_text or "무서운" in query_text or "호러" in query_text:
        candidates.sort(key=lambda x: (x['fear'], x['rating']), reverse=True)
    elif "쉬운" in query_text or "안어려운" in query_text:
        candidates.sort(key=lambda x: (x['rating'], -x['difficulty']), reverse=True)
    elif "문제방" in query_text or "어려운" in query_text or "문제" in query_text:
        candidates.sort(key=lambda x: (x['problem'], x['difficulty'], x['rating']), reverse=True)
    elif "활동적이지 않은" in query_text or "치마" in query_text:
        candidates.sort(key=lambda x: (x['rating'], -x['activity']), reverse=True)
    elif "활동" in query_text or "동적인" in query_text or "바지" in query_text:
        candidates.sort(key=lambda x: (x['activity'], x['rating']), reverse=True)
    elif "스토리" in query_text or "드라마" in query_text or "감성" in query_text:
        candidates.sort(key=lambda x: (x['story'], x['rating']), reverse=True)
    elif "인테리어" in query_text or "리얼리티" in query_text or "실제같은" in query_text:
        candidates.sort(key=lambda x: (x['interior'], x['rating']), reverse=True)
    elif "연출" in query_text or "장치" in query_text or "화려" in query_text or "스케일" in query_text:
        candidates.sort(key=lambda x: (x['act'], x['rating']), reverse=True)
    else:
        candidates.sort(key=lambda x: x['rating'], reverse=True)
    
    return candidates

# ==============================================================================
# [Recommender] RuleBased
# ==============================================================================
class RuleBasedRecommender:
    def __init__(self, db):
        self.db = db

    def search_themes(self, criteria, user_query="", limit=30, nicknames=None, exclude_ids=None):
        log_msg(f"🔎 [RuleBased] 검색 시작 | 조건: {criteria}")
        
        played_theme_ids = set()
        target_users = []
        if isinstance(nicknames, str):
            target_users = [n.strip() for n in nicknames.split(',')]
        elif isinstance(nicknames, list):
            target_users = nicknames

        if target_users:
            log_msg(f"   👤 플레이 이력 조회: {target_users}")
            try:
                users_ref = self.db.collection('users')
                if len(target_users) > 10: target_users = target_users[:10]
                user_q = users_ref.where(filter=FieldFilter("nickname", "in", target_users))
                user_docs = list(user_q.stream())
                
                for u_doc in user_docs:
                    u_data = u_doc.to_dict()
                    played = u_data.get('played', [])
                    for pid in played:
                        played_theme_ids.add(int(pid))
                log_msg(f"   -> 제외할 플레이 테마: {len(played_theme_ids)}개")
            except Exception as e:
                log_msg(f"   ⚠️ 그룹 기록 조회 오류: {e}")

        total_exclude_ids = set(exclude_ids) if exclude_ids else set()
        total_exclude_ids.update(played_theme_ids)

        themes_ref = self.db.collection('themes')
        query = themes_ref

        if criteria.get('location'):
            query = query.where(filter=FieldFilter('location', '==', criteria['location']))
            log_msg(f"   📌 지역 필터: {criteria['location']}")
        else:
            query = query.order_by('satisfyTotalRating', direction=firestore.Query.DESCENDING).limit(100)
            log_msg("   📌 지역 필터 없음: 전체 만족도순 검색")

        docs = list(query.stream())
        log_msg(f"   📦 DB 문서 수신: {len(docs)}개")

        raw_candidates = []
        count_excluded = 0

        for doc in docs:
            data = doc.to_dict()
            try:
                ref_id = data.get('ref_id')
                tid = int(ref_id) if ref_id is not None else int(doc.id)
                if tid in total_exclude_ids or str(tid) in total_exclude_ids or doc.id in total_exclude_ids:
                    count_excluded += 1
                    continue
            except:
                if doc.id in total_exclude_ids:
                    count_excluded += 1
                    continue

            is_match = True
            if criteria.get('location'):
                loc_input = criteria['location']
                db_loc = f"{data.get('location', '')} {data.get('store_name', '')}"
                if loc_input not in db_loc:
                    is_match = False

            if is_match:
                vec_obj = data.get('embedding_field')
                vector = None
                try:
                    if vec_obj:
                        if hasattr(vec_obj, 'to_map'): 
                            vector = vec_obj.to_map()['value']
                        else:
                            vector = list(vec_obj)
                except: pass

                raw_candidates.append({
                    'id': doc.id,
                    'title': data.get('title'),
                    'store': data.get('store_name'),
                    'location': data.get('location'),
                    'genre': data.get('genre'),
                    'desc': data.get('description', '')[:150],
                    'rating': float(data.get('satisfyTotalRating') or 0),
                    'fear': float(data.get('fearTotalRating') or 0),
                    'difficulty': float(data.get('difficultyTotalRating') or 0),
                    'activity': float(data.get('activityTotalRating') or 0),
                    'problem': float(data.get('problemTotalRating') or 0),
                    'story': float(data.get('storyTotalRating') or 0),
                    'interior': float(data.get('interiorTotalRating') or 0),
                    'act': float(data.get('actTotalRating') or 0),
                    'vector': vector
                })

        log_msg(f"   ✂️ 필터링 후 후보: {len(raw_candidates)}개 (제외됨: {count_excluded})")
        sorted_candidates = sort_candidates_by_query(raw_candidates, user_query)
        return sorted_candidates[:limit]

# ==============================================================================
# [Recommender] Vector
# ==============================================================================
class VectorRecommender:
    def __init__(self, db, model):
        self.db = db
        self.model = model

    def get_group_vector(self, nicknames):
        target_users = []
        if isinstance(nicknames, str):
            target_users = [n.strip() for n in nicknames.split(',')]
        elif isinstance(nicknames, list):
            target_users = nicknames
            
        if not target_users: return None

        log_msg(f"👥 [Vector] 그룹 벡터 계산: {target_users}")

        try:
            users_ref = self.db.collection('users')
            if len(target_users) > 10: target_users = target_users[:10]
            query = users_ref.where(filter=FieldFilter("nickname", "in", target_users))
            docs = list(query.stream())
            vectors = []
            for doc in docs:
                user_data = doc.to_dict()
                vec_obj = user_data.get('embedding_field')
                if vec_obj:
                    try:
                        if hasattr(vec_obj, 'to_map'):
                            v = vec_obj.to_map()['value']
                        else:
                            v = list(vec_obj)
                        vectors.append(v)
                    except: pass
            
            log_msg(f"   -> {len(vectors)}/{len(target_users)} 명의 벡터 확보")
            
            if not vectors: return None
            matrix = np.array(vectors)
            mean_vector = np.mean(matrix, axis=0)
            norm = np.linalg.norm(mean_vector)
            if norm > 0:
                mean_vector = mean_vector / norm
            return mean_vector.tolist()

        except Exception as e:
            log_msg(f"   ⚠️ 그룹 벡터 오류: {e}")
            return None

    def rerank_candidates(self, candidates, user_context):
        log_msg(f"🔄 [Vector] 재정렬 시작 (대상: {user_context})")
        
        if isinstance(user_context, list) or (isinstance(user_context, str) and ',' in user_context):
            target_vec = self.get_group_vector(user_context)
        else:
            target_vec = self.get_group_vector([user_context])

        if not target_vec:
            log_msg("   ⚠️ 타겟 벡터 없음. 재정렬 건너뜀.")
            return candidates

        try:
            u_v = np.array(target_vec)
            u_norm = np.linalg.norm(u_v)
            if u_norm == 0: return candidates

            for c in candidates:
                if c.get('vector'):
                    c_v = np.array(c['vector'])
                    c_norm = np.linalg.norm(c_v)
                    if c_norm > 0:
                        sim = np.dot(u_v, c_v) / (u_norm * c_norm)
                    else:
                        sim = 0
                else:
                    sim = -1
                c['score_vec'] = sim
            
            candidates.sort(key=lambda x: x.get('score_vec', -1), reverse=True)
            log_msg("   ✅ 재정렬 완료")
            return candidates
            
        except Exception as e:
            log_msg(f"   ⚠️ 재정렬 중 오류: {e}")
            return candidates

    def _execute_vector_search(self, vector, limit=20, filters=None, exclude_ids=None):
        log_msg("🚀 [Vector] DB 유사도 검색 실행")
        themes_ref = self.db.collection('themes')
        query = themes_ref
        
        if filters and filters.get('location'):
            query = query.where(filter=FieldFilter("location", "==", filters['location']))

        try:
            fetch_limit = limit + len(exclude_ids) if exclude_ids else limit
            
            vector_query = query.find_nearest(
                vector_field="embedding_field",
                query_vector=Vector(vector),
                distance_measure=DistanceMeasure.COSINE,
                limit=fetch_limit
            )
            results = []
            docs = vector_query.get()
            log_msg(f"   📦 벡터 검색 결과: {len(docs)}개")
            
            for doc in docs:
                if exclude_ids and doc.id in exclude_ids:
                    continue
                    
                data = doc.to_dict()
                results.append({
                    'id': doc.id,
                    'title': data.get('title'),
                    'store': data.get('store_name'),
                    'location': data.get('location'),
                    'genre': data.get('genre'),
                    'desc': data.get('search_text', '')[:150],
                    'rating': float(data.get('satisfyTotalRating') or 0),
                    'fear': float(data.get('fearTotalRating') or 0),
                    'activity': float(data.get('activityTotalRating') or 0),
                    'difficulty': float(data.get('difficultyTotalRating') or 0),
                    'interior': float(data.get('interiorTotalRating') or 0),
                    'problem': float(data.get('problemTotalRating') or 0),
                    'story': float(data.get('storyTotalRating') or 0),
                    'act': float(data.get('actTotalRating') or 0)
                })
                
                if len(results) >= limit:
                    break
            return results
        except Exception as e:
            log_msg(f"   ❌ Vector Search 실패: {e}")
            return []

    def recommend_by_text(self, query_text, filters=None, exclude_ids=None):
        if not self.model: return []
        log_msg(f"🔤 [TextVector] 텍스트 임베딩: '{query_text}'")
        query_vector = self.model.encode(query_text).tolist()
        return self._execute_vector_search(query_vector, filters=filters, exclude_ids=exclude_ids)

    def recommend_by_user_search(self, user_context, limit=3, filters=None, exclude_ids=None):
        if isinstance(user_context, list) or (isinstance(user_context, str) and ',' in user_context):
            target_vec = self.get_group_vector(user_context)
        else:
            target_vec = self.get_group_vector([user_context])
            
        if not target_vec: return []
        return self._execute_vector_search(target_vec, limit=limit, filters=filters, exclude_ids=exclude_ids)

# ==============================================================================
# [Bot Engine]
# ==============================================================================
class EscapeBotEngine:
    def __init__(self, vector_recommender, rule_recommender, groq_key, tavily_key):
        self.vector_recommender = vector_recommender
        self.rule_recommender = rule_recommender 
        self.db = rule_recommender.db
        
        self.tavily_client = TavilyClient(api_key=tavily_key) if tavily_key else None
        
        if groq_key:
            self.groq_client = Groq(api_key=groq_key)
            self.model_name = "llama-3.3-70b-versatile"
        else:
            self.groq_client = None

    def _call_llm(self, prompt, json_mode=False):
        if not self.groq_client: return None
        try:
            chat_completion = self.groq_client.chat.completions.create(
                messages=[
                    {"role": "system", "content": "You are a helpful assistant. Always respond in Korean." + (" Output JSON only." if json_mode else "")},
                    {"role": "user", "content": prompt}
                ],
                model=self.model_name,
                temperature=0.3,
                response_format={"type": "json_object"} if json_mode else None,
            )
            return chat_completion.choices[0].message.content
        except Exception as e:
            log_msg(f"❌ [Error] Groq API 호출 실패: {e}")
            return None

    def find_theme_id(self, location, theme_name):
        log_msg(f"🔎 [DB] 테마 ID 검색: {theme_name} ({location})")
        try:
            themes_ref = self.db.collection('themes')
            query = themes_ref
            if location:
                query = query.where(filter=FieldFilter("location", "==", location))
            docs = list(query.limit(500).stream())
            
            target_name = theme_name.replace(" ", "")
            for doc in docs:
                data = doc.to_dict()
                title = data.get('title', '')
                if target_name in title.replace(" ", ""):
                    tid = int(data.get('ref_id') or doc.id)
                    log_msg(f"   ✅ 찾음: {title} (ID: {tid})")
                    return tid
            log_msg("   ❌ 찾지 못함")
            return None
        except Exception:
            return None

    def update_play_history(self, nickname, theme_id, action):
        log_msg(f"✏️ [DB] 플레이 기록 업데이트: {nickname} -> {theme_id} ({action})")
        try:
            users_ref = self.db.collection('users')
            q = users_ref.where(filter=FieldFilter("nickname", "==", nickname)).limit(1)
            docs = list(q.stream())
            if not docs: return "❌ 유저를 찾을 수 없습니다."
            user_doc = docs[0]
            if action == "played_check":
                user_doc.reference.update({"played": firestore.ArrayUnion([theme_id])})
                return "✅ 플레이 목록에 추가했습니다!"
            elif action == "not_played_check":
                user_doc.reference.update({"played": firestore.ArrayRemove([theme_id])})
                return "✅ 플레이 목록에서 제외했습니다."
            return "❓ 알 수 없는 요청입니다."
        except Exception as e:
            return f"❌ 업데이트 중 오류: {e}"

    def analyze_user_intent(self, user_query):
        log_msg(f"🧠 [LLM] 의도 분석 요청: '{user_query}'")
        if not self.groq_client: return {}
        
        prompt = f"""
        사용자의 질문을 분석하여 JSON으로 반환하세요.
        질문: "{user_query}"
        
        Action 규칙:
        1. "played_check_inquiry": 기록 방법 문의.
        2. "played_check": "했다", "플레이했어".
        3. "not_played_check": "안했어", "취소해줘".
        4. "recommend": 추천 요청.
        5. "another_recommend": "다른거", "이거 말고".

        Fields:
        - action, location, theme, keywords, mentioned_users, items(지역/테마 리스트)
        """
        try:
            result_str = self._call_llm(prompt, json_mode=True)
            result = json.loads(result_str) if result_str else {"action": "recommend"}
            log_msg(f"   -> 분석 결과: {result.get('action')}")
            return result
        except Exception as e:
            log_msg(f"   ❌ 의도 분석 실패: {e}")
            return {"action": "recommend"}

    def generate_reply(self, user_query, user_context=None, session_context=None):
        if not self.groq_client:
            return "⚠️ Groq API Key가 설정되지 않았습니다.", {}, {}, "error"

        log_msg("\n🏁 [Generate Reply] 처리 시작")
        
        intent_data = self.analyze_user_intent(user_query)
        action = intent_data.get('action', 'recommend')

        # 플레이 기록 관리
        if action in ['played_check', 'not_played_check', 'played_check_inquiry']:
            # (기존 로직 유지)
            # ...
            if action == 'played_check_inquiry':
                return "테마를 [지역, 테마명] 형식으로 알려주세요.", {}, {}, action

            if not user_context:
                return "⚠️ 닉네임 입력이 필요합니다.", {}, {}, action
            
            items = intent_data.get('items', [])
            if not items and intent_data.get('theme'):
                 items.append({"location": intent_data.get('location'), "theme": intent_data.get('theme')})
            
            msg_list = []
            for item in items:
                loc = item.get('location')
                thm = item.get('theme')
                if thm:
                    tid = self.find_theme_id(loc, thm)
                    if tid:
                        res = self.update_play_history(user_context, tid, action)
                        msg_list.append(f"- {thm}: {res}")
                    else:
                        msg_list.append(f"- {thm}: ⚠️ 테마 못 찾음")
            return "\n".join(msg_list), {}, {}, action

        # 필터 설정
        filters = {
            'location': intent_data.get('location'),
            'keywords': intent_data.get('keywords', []),
            'mentioned_users': intent_data.get('mentioned_users', [])
        }
        
        # 그룹 멤버
        current_users = []
        if user_context:
            if ',' in user_context:
                current_users = [u.strip() for u in user_context.split(',')]
            else:
                current_users = [user_context.strip()]
        for u in filters['mentioned_users']:
            if u not in current_users:
                current_users.append(u)
        
        final_context = current_users if len(current_users) > 1 else (current_users[0] if current_users else None)
        log_msg(f"👥 최종 추천 대상: {final_context}")
        
        filters_to_use = {}
        exclude_ids = []
        if action == 'another_recommend':
            if session_context:
                filters_to_use = session_context.get('last_filters', {})
                exclude_ids = list(session_context.get('shown_ids', []))
                log_msg(f"🔄 '다른거' 요청 -> 이전 필터 사용, {len(exclude_ids)}개 제외")
        else:
            filters_to_use = filters
            exclude_ids = []

        final_results = {}
        
        # 1. RuleBased
        candidates_rule = self.rule_recommender.search_themes(
            filters_to_use, user_query, limit=3, nicknames=final_context, exclude_ids=exclude_ids
        )
        if candidates_rule: final_results['rule_based'] = candidates_rule

        # 2. Vector Personalized
        if final_context:
            candidates_vector = self.vector_recommender.recommend_by_user_search(
                final_context, limit=3, filters=filters_to_use, exclude_ids=exclude_ids
            )
            if candidates_vector:
                final_reranked = sort_candidates_by_query(candidates_vector, user_query)
                final_results['personalized'] = final_reranked

        # 3. Fallback
        if not final_results:
            candidates_text = self.vector_recommender.recommend_by_text(
                user_query, filters=filters_to_use, exclude_ids=exclude_ids
            )
            if candidates_text:
                final_results['text_search'] = sort_candidates_by_query(candidates_text, user_query)[:3]
            else:
                return "죄송합니다. 조건에 맞는 테마를 찾지 못했습니다.", {}, filters_to_use, action

        # 답변 생성
        context_str = ""
        # (생략: 컨텍스트 문자열 생성 부분은 기존과 동일)
        # ... 

        log_msg("📝 [LLM] 최종 답변 생성 요청...")
        system_prompt = f"당신은 방탈출 추천 AI입니다. 질문: {user_query}. 목록: {final_results}. 추천해주세요."
        response_text = self._call_llm(system_prompt)
        
        log_msg("✅ [BotEngine] 완료")
        return response_text, final_results, filters_to_use, action

# ==============================================================================
# [UI] Streamlit App
# ==============================================================================
def main():
    # 사이드바에 로그 창 만들기
    with st.sidebar:
        st.title("⚙️ 설정")
        # ... (설정 UI) ...
        
        with st.expander("🛠️ 디버그 로그 (실시간)"):
            if "app_logs" not in st.session_state:
                st.session_state.app_logs = []
            
            # 로그 출력 (최신순)
            for log in reversed(st.session_state.app_logs):
                st.caption(log)
    
    # ... (기존 메인 UI) ...

if __name__ == "__main__":
    main()
