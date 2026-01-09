import json
import copy
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
            
            target_name = theme_name.replace(" ", "")
            for doc in docs:
                data = doc.to_dict()
                title = data.get('title', '')
                if target_name in title.replace(" ", ""):
                    tid = int(data.get('ref_id') or doc.id)
                    print(f"   ✅ 찾음: {title} (ID: {tid})")
                    return tid
                letters = data.get('letters')
                if target_name in letters.replace(" ", ""):
                    tid = int(data.get('ref_id') or doc.id)
                    print(f"   ✅ 찾음: {title} (ID: {tid})")
                    return tid
            print("   ❌ 찾지 못함")
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
        print(f"🧠 [LLM] 의도 분석 요청: '{user_query}'")
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
            print(f"   -> 분석 결과: {result.get('action')}")
            return result
        except Exception as e:
            print(f"   ❌ 의도 분석 실패: {e}")
            return {"action": "recommend"}

    def generate_reply(self, user_query, user_context=None, session_context=None):
        if not self.groq_client:
            return "⚠️ Groq API Key가 설정되지 않았습니다.", {}, {}, "error"

        print("\n==================================================")
        print("🏁 [Generate Reply] 처리 시작")
        
        intent_data = self.analyze_user_intent(user_query)
        action = intent_data.get('action', 'recommend')

        # 플레이 기록 관리
        if action in ['played_check', 'not_played_check', 'played_check_inquiry']:
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
        print(f"👥 [Engine] 최종 추천 대상(Context): {final_context}")
        
        filters_to_use = {}
        exclude_ids = []
        if action == 'another_recommend':
            print("🔄 [Action] '다른거' 요청 감지 -> 이전 필터 복원")
            if session_context:
                filters_to_use = session_context.get('last_filters', {})
                exclude_ids = list(session_context.get('shown_ids', []))
        else:
            filters_to_use = filters
            exclude_ids = []

        print(f"⚙️ [Filter] 적용 필터: {filters_to_use}")
        print(f"🚫 [Exclude] 제외할 테마 수: {len(exclude_ids)}")

        final_results = {}
        
        # 1. RuleBased
        print("\n🚀 [Step 1] Rule-Based 검색 실행...")
        candidates_rule = self.rule_recommender.search_themes(
            filters_to_use, user_query, limit=3, nicknames=final_context, exclude_ids=exclude_ids
        )
        if candidates_rule:
            final_results['rule_based'] = candidates_rule
            print(f"   ✅ [Result] 룰베이스 {len(candidates_rule)}개 확보")

        # 2. Vector Personalized
        if final_context:
            print("\n🚀 [Step 2] Vector(개인화) 검색 실행...")
            candidates_vector = self.vector_recommender.recommend_by_user_search(
                final_context, limit=3, filters=filters_to_use, exclude_ids=exclude_ids
            )
            if candidates_vector:
                final_reranked = sort_candidates_by_query(candidates_vector, user_query)
                final_results['personalized'] = final_reranked
                print(f"   ✅ [Result] 개인화 벡터 {len(final_reranked)}개 확보")

        # 3. Fallback
        if not final_results:
            print("\n🚀 [Step 3] 결과 없음 -> 텍스트 벡터 검색 실행...")
            candidates_text = self.vector_recommender.recommend_by_text(
                user_query, filters=filters_to_use, exclude_ids=exclude_ids
            )
            if candidates_text:
                final_results['text_search'] = sort_candidates_by_query(candidates_text, user_query)[:3]
                print(f"   ✅ [Result] 텍스트 검색 {len(final_results['text_search'])}개 확보")
            else:
                print("   ❌ [Result] 검색 결과 없음")
                return "죄송합니다. 조건에 맞는 테마를 찾지 못했습니다.", {}, filters_to_use, action

        # 답변 생성
        context_str = ""
        # (생략: 컨텍스트 문자열 생성 부분은 기존과 동일)
        
        print("📝 [LLM] 최종 답변 생성 요청...")
        system_prompt = f"당신은 방탈출 추천 AI입니다. 질문: {user_query}. 목록: {final_results}. 추천해주세요."
        response_text = self._call_llm(system_prompt)
        
        if not response_text:
            response_text = "죄송합니다. 답변 생성 중 오류가 발생했습니다."

        print("✅ [BotEngine] 답변 생성 완료")
        print("==================================================\n")
        return response_text, final_results, filters_to_use, action
