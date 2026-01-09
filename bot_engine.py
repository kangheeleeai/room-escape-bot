import json
import re
from groq import Groq
from tavily import TavilyClient
from database import firestore, FieldFilter
from utils import sort_candidates_by_query

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

    def _clean_json_string(self, json_str):
        if not json_str: return ""
        cleaned = re.sub(r"```json\s*", "", json_str)
        cleaned = re.sub(r"```", "", cleaned)
        return cleaned.strip()

    def _call_llm(self, prompt, json_mode=False):
        if not self.groq_client: return None
        try:
            chat_completion = self.groq_client.chat.completions.create(
                messages=[
                    {
                        "role": "system",
                        "content": "You are a helpful assistant for Escape Room recommendations. Always respond in Korean." + (" Output JSON only." if json_mode else "")
                    },
                    {
                        "role": "user",
                        "content": prompt,
                    }
                ],
                model=self.model_name,
                temperature=0.1, 
                response_format={"type": "json_object"} if json_mode else None,
            )
            return chat_completion.choices[0].message.content
        except Exception as e:
            return None

    def find_theme_id(self, location, theme_name, on_log=None):
        if on_log: on_log(f"[DB] 테마 검색: {theme_name} (지역: {location})")
        
        try:
            themes_ref = self.db.collection('themes')
            query = themes_ref
            if location:
                query = query.where(filter=FieldFilter("location", "==", location))
            
            docs = list(query.limit(200).stream())
            target_name = theme_name.replace(" ", "")
            
            for doc in docs:
                data = doc.to_dict()
                title = data.get('title', '')
                letters = data.get('letters', '')
                if target_name in title.replace(" ", ""):
                    if on_log: on_log(f"   -> 발견: {title} (ID: {doc.id})")
                    return int(data.get('ref_id') or doc.id)
                if letters and target_name in letters.replace(" ", ""):
                    return int(data.get('ref_id') or doc.id)
            
            if on_log: on_log("   -> 검색 실패")
            return None
        except Exception as e:
            if on_log: on_log(f"   ⚠️ 검색 에러: {e}")
            return None

    def update_play_history(self, nickname, theme_id, action, on_log=None):
        try:
            users_ref = self.db.collection('users')
            q = users_ref.where(filter=FieldFilter("nickname", "==", nickname)).limit(1)
            docs = list(q.stream())
            
            if not docs: return "❌ 유저 미등록"
            
            user_doc = docs[0]
            if action == "played_check":
                user_doc.reference.update({"played": firestore.ArrayUnion([theme_id])})
                if on_log: on_log(f"[기록] {nickname}님 플레이 리스트에 {theme_id} 추가")
                return "추가 완료"
            elif action == "not_played_check":
                user_doc.reference.update({"played": firestore.ArrayRemove([theme_id])})
                if on_log: on_log(f"[기록] {nickname}님 플레이 리스트에서 {theme_id} 삭제")
                return "삭제 완료"
            return "알 수 없는 요청"
        except Exception as e:
            return f"에러: {e}"

    def analyze_user_intent(self, user_query, on_log=None):
        if on_log: on_log(f"[LLM] 사용자 의도 분석 중... ('{user_query}')")
        
        if not self.groq_client: return {}
        
        prompt = f"""
        사용자의 질문을 분석하여 방탈출 챗봇의 의도(Intent)와 정보를 추출하세요.
        질문: "{user_query}"
        [분석 규칙]
        1. "played_check_inquiry": 플레이 기록 방법 문의
        2. "played_check": 플레이 했다고 말함 (예: "강남 링 했어")
        3. "not_played_check": 안 했다고 취소함
        4. "recommend": 추천 요청
        5. "another_recommend": 다른 거 추천 요청
        
        [반환 필드] action, items(location, theme), location, keywords, mentioned_users
        JSON only.
        """
        try:
            result_str = self._call_llm(prompt, json_mode=True)
            cleaned_str = self._clean_json_string(result_str)
            result = json.loads(cleaned_str)
            if on_log: on_log(f"   -> 분석 완료: {result.get('action')}, 키워드: {result.get('keywords')}")
            return result
        except Exception as e:
            if on_log: on_log(f"   ❌ 의도 분석 실패: {e}")
            return {"action": "recommend", "keywords": [user_query]}

    def generate_reply(self, user_query, user_context=None, session_context=None, on_log=None):
        """
        on_log: 로그 메시지를 출력할 콜백 함수 (예: st.write)
        """
        if not self.groq_client:
            return "⚠️ API Key 설정 필요", {}, {}, "error", {}

        # 1. 의도 분석
        intent_data = self.analyze_user_intent(user_query, on_log)
        action = intent_data.get('action', 'recommend')
        debug_info = {"intent": intent_data, "query": user_query}

        # 플레이 기록 문의
        if action == "played_check_inquiry":
            msg = "플레이한 테마를 `[지역] [테마명] 했어` 라고 말씀해주시면 기록해 드립니다!"
            return msg, {}, {}, action, debug_info

        # 플레이 기록 추가/삭제
        if action in ['played_check', 'not_played_check']:
            if not user_context:
                return "⚠️ 닉네임을 먼저 설정해주세요.", {}, {}, action, debug_info
            
            items = intent_data.get('items', [])
            if not items and intent_data.get('theme'):
                items.append({"location": intent_data.get('location'), "theme": intent_data.get('theme')})

            results_msg = []
            success_count = 0
            
            for item in items:
                loc = item.get('location')
                theme = item.get('theme')
                if theme:
                    tid = self.find_theme_id(loc, theme, on_log)
                    if tid:
                        res = self.update_play_history(user_context, tid, action, on_log)
                        if "완료" in res: success_count += 1
                        results_msg.append(f"- **{theme}**: {res}")
                    else:
                        results_msg.append(f"- **{theme}**: ⚠️ 테마 못 찾음")
            
            return "\n".join(results_msg), {}, {}, action, debug_info

        # 3. 필터 설정
        current_filters = {
            'location': intent_data.get('location'),
            'keywords': intent_data.get('keywords', []),
            'mentioned_users': intent_data.get('mentioned_users', [])
        }
        
        # 유저 처리
        current_users = [u.strip() for u in str(user_context or "").split(',') if u.strip()]
        for u in current_filters['mentioned_users']:
            if u not in current_users: current_users.append(u)
        
        final_context = current_users if len(current_users) > 1 else (current_users[0] if current_users else None)

        filters_to_use = {}
        exclude_ids = []
        
        if action == 'another_recommend' and session_context:
            filters_to_use = session_context.get('last_filters', {})
            exclude_ids = list(session_context.get('shown_ids', []))
            if current_filters.get('location'): filters_to_use['location'] = current_filters['location']
        else:
            filters_to_use = current_filters
            exclude_ids = []

        if on_log: on_log(f"필터 적용: {filters_to_use}, 제외 ID: {len(exclude_ids)}개")

        # 4. 추천 실행
        final_results = {}
        
        # Rule-Based        
        candidates_rule = self.rule_recommender.search_themes(
            filters_to_use, user_query, limit=3, nicknames=final_context, exclude_ids=exclude_ids, log_func=on_log
        )
        if candidates_rule: final_results['rule_based'] = candidates_rule
        if on_log: on_log(f"exclude_ids: {len(exclude_ids)}개")

        # Personalized
        if on_log: on_log(f"final_context: {final_context}")
        
        if final_context:
            if on_log: on_log(f"exclude_ids: {len(exclude_ids)}개")
            # if on_log: on_log(f"exclude_ids:{exclude_ids}")
            
            candidates_vector = self.vector_recommender.recommend_by_user_search(
                final_context, limit=3, filters=filters_to_use, exclude_ids=exclude_ids, log_func=on_log
            )
            if candidates_vector:
                final_reranked = sort_candidates_by_query(candidates_vector, user_query)
                final_results['personalized'] = final_reranked

        # Fallback
        if not final_results:
            candidates_text = self.vector_recommender.recommend_by_text(
                user_query, filters=filters_to_use, exclude_ids=exclude_ids, log_func=on_log
            )
            if candidates_text:
                final_results['text_search'] = sort_candidates_by_query(candidates_text, user_query)[:3]
            else:
                return "조건에 맞는 테마를 찾지 못했습니다.", {}, filters_to_use, action, debug_info

        # LLM 설명 생성
        if on_log: on_log("📝 최종 답변 생성 중 (LLM)...")
        
        context_str = ""
        for k, v in final_results.items():
            context_str += f"\n[{k}]\n" + "\n".join([f"- {i['title']}" for i in v])

        system_prompt = f"""
        당신은 방탈출 추천 AI입니다. 질문: "{user_query}"
        [추천 목록] {context_str}
        위 목록에서 2~3개를 골라 친절하게 추천해주세요.
        """
        response_text = self._call_llm(system_prompt) or "답변 생성 오류"

        return response_text, final_results, filters_to_use, action, debug_info
