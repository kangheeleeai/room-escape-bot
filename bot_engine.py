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
        """LLM이 마크다운 코드 블록(```json ... ```)을 포함할 경우 제거"""
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
            print(f"❌ [Error] Groq API 호출 실패: {e}")
            return None

    def find_theme_id(self, location, theme_name):
        print(f"🔎 [DB] 테마 ID 검색: {theme_name} (지역: {location})")
        try:
            themes_ref = self.db.collection('themes')
            query = themes_ref
            
            if location:
                query = query.where(filter=FieldFilter("location", "==", location))
            
            docs = list(query.limit(500).stream())
            
            # 검색어 정규화 (공백 제거)
            target_name = theme_name.replace(" ", "")
            
            for doc in docs:
                data = doc.to_dict()
                title = data.get('title', '')
                letters = data.get('letters', '') # 약어/초성 등
                
                # 제목이나 약어에 검색어가 포함되면 매칭
                if target_name in title.replace(" ", ""):
                    tid = int(data.get('ref_id') or doc.id)
                    return tid
                if letters and target_name in letters.replace(" ", ""):
                    tid = int(data.get('ref_id') or doc.id)
                    return tid
            return None
        except Exception as e:
            print(f"   ⚠️ 검색 에러: {e}")
            return None

    def update_play_history(self, nickname, theme_id, action):
        print(f"✏️ [DB] 플레이 기록 업데이트: {nickname} -> {theme_id} ({action})")
        try:
            users_ref = self.db.collection('users')
            q = users_ref.where(filter=FieldFilter("nickname", "==", nickname)).limit(1)
            docs = list(q.stream())
            
            if not docs: return "❌ 유저 정보를 찾을 수 없습니다. (닉네임 등록 필요)"
            
            user_doc = docs[0]
            if action == "played_check":
                # 이미 있는지 확인 후 추가 (중복 방지 로직은 ArrayUnion이 처리해주지만 명시적으로)
                user_doc.reference.update({"played": firestore.ArrayUnion([theme_id])})
                return "✅ 기록 추가 완료"
            elif action == "not_played_check":
                user_doc.reference.update({"played": firestore.ArrayRemove([theme_id])})
                return "🗑️ 기록 삭제 완료"
            return "알 수 없는 요청"
        except Exception as e:
            return f"에러 발생: {e}"

    def analyze_user_intent(self, user_query):
        print(f"🧠 [LLM] 사용자 의도 분석 요청... Query: '{user_query}'")
        if not self.groq_client: return {}
        
        prompt = f"""
        사용자의 질문을 분석하여 방탈출 챗봇의 의도(Intent)와 정보를 추출하세요.

        질문: "{user_query}"

        [분석 규칙]
        1. "played_check_inquiry": 사용자가 플레이 기록을 어떻게 남기는지 묻거나, 단순히 "플레이한 테마", "기록 추가" 라고만 말했을 때.
        2. "played_check": 사용자가 특정 테마를 했다고 말할 때. (예: "강남 링 했어", "홍대 비트포비아 던전 다 했어")
        3. "not_played_check": 사용자가 안 했다고 하거나 취소할 때.
        4. "recommend": 새로운 추천 요청. (지역, 장르, 분위기 등을 물어볼 때)
        5. "another_recommend": 방금 추천해준 것 말고 다른 것을 원할 때.

        [추출 필드]
        - action: 위 규칙 중 하나.
        - items: 플레이 기록 관련일 때, {{"location": "지역", "theme": "테마명"}} 객체들의 리스트. 
        - location: (추천용) 지역명 (예: 강남, 홍대, 건대). 없으면 null.
        - keywords: (추천용) 장르, 분위기, 특징 등 키워드 리스트.
        - mentioned_users: 언급된 닉네임 리스트.

        JSON 형식으로만 반환하세요.
        예시: {{ "action": "played_check", "items": [{{"location": "강남", "theme": "링"}}] }}
        """
        try:
            result_str = self._call_llm(prompt, json_mode=True)
            cleaned_str = self._clean_json_string(result_str)
            result = json.loads(cleaned_str)
            print(f"   -> 분석 결과: {result}")
            return result
        except Exception as e:
            print(f"   ❌ 의도 분석 실패 (JSON 파싱 오류 등): {e}")
            # 실패 시 기본값: 일반 추천으로 간주하고 키워드만 쿼리로 사용
            return {"action": "recommend", "keywords": [user_query]}

    def generate_reply(self, user_query, user_context=None, session_context=None):
        if not self.groq_client:
            return "⚠️ Groq API Key가 설정되지 않았습니다.", {}, {}, "error", {}

        # 1. 의도 분석
        intent_data = self.analyze_user_intent(user_query)
        action = intent_data.get('action', 'recommend')

        # 디버깅용 정보
        debug_info = {
            "intent": intent_data,
            "query": user_query
        }

        # 플레이 기록 문의 처리
        if action == "played_check_inquiry":
            msg = "플레이한 테마를 **[지역, 테마명]** 형식으로 말씀해 주시면 기록해 드릴게요!\n예: `강남 링 했어`, `홍대 삐릿뽀 플레이했어`"
            return msg, {}, {}, action, debug_info

        # 플레이 기록 추가/삭제 처리
        if action in ['played_check', 'not_played_check']:
            if not user_context:
                return "⚠️ 플레이 기록을 관리하려면 사이드바에 **닉네임**을 먼저 입력해주세요.", {}, {}, action, debug_info
            
            items = intent_data.get('items', [])
            # items가 비어있지만 theme/location 필드가 있는 경우 처리 (LLM 출력 편차 대응)
            if not items and intent_data.get('theme'):
                items.append({
                    "location": intent_data.get('location'),
                    "theme": intent_data.get('theme')
                })

            if not items:
                return "⚠️ 테마 정보를 정확히 인식하지 못했습니다. '[지역] [테마명] 했어' 라고 다시 말씀해 주시겠어요?", {}, {}, action, debug_info

            results_msg = []
            success_count = 0
            
            for item in items:
                loc = item.get('location')
                theme = item.get('theme')
                if theme:
                    tid = self.find_theme_id(loc, theme)
                    if tid:
                        res = self.update_play_history(user_context, tid, action)
                        if "완료" in res: success_count += 1
                        results_msg.append(f"- **{theme}**: {res}")
                    else:
                        results_msg.append(f"- **{theme}**: ⚠️ 테마를 찾을 수 없음 (지역을 확인해주세요)")
            
            summary = f"**처리 결과 ({success_count}/{len(items)}건)**\n" + "\n".join(results_msg)
            return summary, {}, {}, action, debug_info

        # 3. 필터 설정 (추천 로직)
        current_filters = {
            'location': intent_data.get('location'),
            'keywords': intent_data.get('keywords', []),
            'mentioned_users': intent_data.get('mentioned_users', [])
        }

        # 유저 목록 정리
        current_users = []
        if user_context:
            # 콤마로 구분된 닉네임 처리
            current_users = [u.strip() for u in str(user_context).split(',') if u.strip()]
        
        # 쿼리에서 언급된 유저 추가
        for u in current_filters['mentioned_users']:
            if u not in current_users:
                current_users.append(u)
        
        final_context = current_users if len(current_users) > 1 else (current_users[0] if current_users else None)
        
        filters_to_use = {}
        exclude_ids = []
        
        # '다른거 추천'인 경우 이전 필터 재사용
        if action == 'another_recommend':
            if session_context:
                filters_to_use = session_context.get('last_filters', {})
                exclude_ids = list(session_context.get('shown_ids', []))
                # 위치정보가 덮어씌워지지 않도록 주의
                if current_filters.get('location'):
                     filters_to_use['location'] = current_filters['location']
        else:
            filters_to_use = current_filters
            exclude_ids = []

        # 4. 추천 실행
        final_results = {}
        
        # Rule-Based
        candidates_rule = self.rule_recommender.search_themes(
            filters_to_use, user_query, limit=3, nicknames=final_context, exclude_ids=exclude_ids
        )
        if candidates_rule:
            final_results['rule_based'] = candidates_rule

        # Vector (Personalized) - 유저 기록이 있을 때만
        if final_context:
            candidates_vector = self.vector_recommender.recommend_by_user_search(
                final_context, limit=3, filters=filters_to_use, exclude_ids=exclude_ids
            )
            if candidates_vector:
                final_reranked = sort_candidates_by_query(candidates_vector, user_query)
                final_results['personalized'] = final_reranked

        # Fallback (Text Search) - 결과가 없으면 텍스트 유사도로 검색
        if not final_results:
            candidates_text = self.vector_recommender.recommend_by_text(
                user_query, filters=filters_to_use, exclude_ids=exclude_ids
            )
            if candidates_text:
                final_results['text_search'] = sort_candidates_by_query(candidates_text, user_query)[:3]
            else:
                return "조건에 맞는 테마를 찾지 못했습니다. 😭\n지역이나 조건을 조금 더 넓혀서 다시 질문해 주시겠어요?", {}, filters_to_use, action, debug_info

        # LLM 설명 생성
        context_str = ""
        total_count = 0
        
        if 'personalized' in final_results:
            context_str += "\n[취향 맞춤 추천]\n"
            for i, item in enumerate(final_results['personalized']):
                context_str += f"- {item['title']} (평점 {item['rating']}): {item['desc'][:60]}...\n"
                total_count += 1
                
        if 'rule_based' in final_results:
            context_str += "\n[조건 부합 추천]\n"
            for i, item in enumerate(final_results['rule_based']):
                context_str += f"- {item['title']} (평점 {item['rating']}): {item['desc'][:60]}...\n"
                total_count += 1

        intro_msg = "이전에 추천드린 테마는 제외하고" if exclude_ids else "요청하신 내용과 플레이 성향을 바탕으로"
        
        system_prompt = f"""
        당신은 방탈출 추천 AI입니다.
        [상황] 질문: "{user_query}" / 근거: {intro_msg}
        [추천 목록]
        {context_str}
        
        [지시]
        위 목록에서 가장 적절한 2~3개를 골라 추천해주세요.
        테마의 특징을 매력적으로 설명하고, 왜 이 테마를 추천했는지 이유를 덧붙이세요.
        """

        response_text = self._call_llm(system_prompt)
        if not response_text:
            response_text = "죄송합니다. 답변 생성 중 일시적인 오류가 발생했습니다."

        debug_info['result_count'] = total_count
        return response_text, final_results, filters_to_use, action, debug_info
