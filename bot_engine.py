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
            self.model_name = "llama3-70b-8192"
        else:
            self.groq_client = None

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
                temperature=0.3,
                response_format={"type": "json_object"} if json_mode else None,
            )
            return chat_completion.choices[0].message.content
        except Exception as e:
            print(f"Groq API Error: {e}")
            return None

    def find_theme_id(self, location, theme_name):
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
                    return int(data.get('ref_id') or doc.id)
            return None
        except Exception:
            return None

    def update_play_history(self, nickname, theme_id, action):
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
        if not self.groq_client: return {}
        
        prompt = f"""
        사용자의 질문을 분석하여 방탈출 추천 서비스의 의도(Intent)와 파라미터를 추출하세요.

        질문: "{user_query}"

        다음 규칙에 따라 'action'을 결정하세요:
        1. "recommend": 새로운 추천을 요청함 (예: "강남 공포 테마 추천해줘", "재밌는거 추천좀").
        2. "another_recommend": 다른 추천을 요청함 (예: "다른거 추천해줘", "이거 말고", "다음").
        3. "played_check": 특정 테마를 플레이했다고 말함 (예: "강남 링 했어", "X 테마 해봤어").
        4. "not_played_check": 플레이하지 않았다고 정정하거나 취소함 (예: "링 안했어", "플레이 기록 취소해줘").

        다음 필드를 추출하세요 (반드시 한국어로):
        - location: 지역명 (예: 강남, 홍대, 건대) 또는 null.
        - theme: 언급된 테마명 (주로 플레이 기록 추가/삭제 시) 또는 null.
        - keywords: 추천을 위한 키워드 리스트 (장르, 분위기, 특징 등 예: "공포", "활동성", "스토리").
        - mentioned_users: 질문에 언급된 다른 유저 닉네임 리스트.

        JSON 형식으로만 반환하세요. 예시:
        {{ "action": "recommend", "location": "강남", "keywords": ["공포"], "theme": null, "mentioned_users": [] }}
        """
        try:
            result = self._call_llm(prompt, json_mode=True)
            return json.loads(result) if result else {"action": "recommend", "keywords": []}
        except Exception as e:
            print(f"Intent analysis error: {e}")
            return {"action": "recommend", "keywords": []}

    def generate_reply(self, user_query, user_context=None, session_context=None):
        if not self.groq_client:
            return "⚠️ Groq API Key가 설정되지 않았습니다.", {}, {}, "error"

        # 1. 의도 분석
        intent_data = self.analyze_user_intent(user_query)
        action = intent_data.get('action', 'recommend')
        print(f"🧠 [Intent] Action: {action}, Data: {intent_data}")

        # 2. 플레이 기록 관리
        if action in ['played_check', 'not_played_check']:
            if not user_context:
                return "⚠️ 플레이 기록을 관리하려면 닉네임 입력이 필요합니다.", {}, {}, action
            
            loc = intent_data.get('location')
            theme = intent_data.get('theme')
            
            if theme:
                tid = self.find_theme_id(loc, theme)
                if tid:
                    msg = self.update_play_history(user_context, tid, action)
                    return f"{msg} ({loc if loc else ''} {theme})", {}, {}, action
                else:
                    return f"⚠️ '{theme}' 테마를 찾을 수 없습니다. 지역 정보가 정확한지 확인해주세요.", {}, {}, action
            else:
                return "⚠️ 테마 이름을 인식하지 못했습니다.", {}, {}, action

        # 3. 필터 설정
        current_filters = {
            'location': intent_data.get('location'),
            'keywords': intent_data.get('keywords', []),
            'mentioned_users': intent_data.get('mentioned_users', [])
        }

        # 그룹 멤버 확인
        current_users = []
        if user_context:
            if ',' in user_context:
                current_users = [u.strip() for u in user_context.split(',')]
            else:
                current_users = [user_context.strip()]
        
        for u in current_filters['mentioned_users']:
            if u not in current_users:
                current_users.append(u)
        
        final_context = current_users if len(current_users) > 1 else (current_users[0] if current_users else None)
        
        # 필터 및 제외 ID 설정
        filters_to_use = {}
        exclude_ids = []
        
        if action == 'another_recommend':
            if session_context:
                filters_to_use = session_context.get('last_filters', {})
                exclude_ids = list(session_context.get('shown_ids', []))
        else:
            filters_to_use = current_filters
            exclude_ids = []

        print(f"🔍 [Engine] Filters: {filters_to_use}, Exclude count: {len(exclude_ids)}")

        # 4. 추천 실행
        final_results = {}
        
        # [Step 1] Rule-Based Candidates (Top 3)
        print("🚀 [Step 1] 룰 기반 검색 실행 중...")
        candidates_rule = self.rule_recommender.search_themes(
            filters_to_use, user_query, limit=3, nicknames=final_context, exclude_ids=exclude_ids
        )
        if candidates_rule:
            final_results['rule_based'] = candidates_rule
            print(f"   -> {len(candidates_rule)}개의 룰 기반 후보 발견.")

        # [Step 2] Personalized Vector Candidates (Top 3)
        if final_context:
            print("🚀 [Step 2] 벡터 검색(개인화) 실행 중...")
            candidates_vector = self.vector_recommender.recommend_by_user_search(
                final_context, limit=3, filters=filters_to_use, exclude_ids=exclude_ids
            )
            if candidates_vector:
                # 결과 미세조정 (정렬)
                final_reranked = sort_candidates_by_query(candidates_vector, user_query)
                final_results['personalized'] = final_reranked
                print(f"   -> {len(final_reranked)}개의 개인화 후보 발견.")

        # [Step 3] Fallback (Text Vector) - 둘 다 없을 때만
        if not final_results:
            print("🚀 [Step 3] 결과 없음. 대체 텍스트 검색 실행 중...")
            candidates_text = self.vector_recommender.recommend_by_text(
                user_query, filters=filters_to_use, exclude_ids=exclude_ids
            )
            if candidates_text:
                final_results['text_search'] = sort_candidates_by_query(candidates_text, user_query)[:3]
            else:
                return "죄송합니다. 조건에 맞는 테마를 찾지 못했습니다. 😭\n조건을 변경해서 다시 질문해 주시겠어요?", {}, filters_to_use, action

        # LLM Context 구성 (모든 결과 포함)
        context_str = ""
        
        if 'personalized' in final_results:
            context_str += "\n[취향 맞춤 추천 (Vector)]\n"
            for i, item in enumerate(final_results['personalized']):
                context_str += f"- {item['title']} (만족도 {item['rating']:.1f}, 공포 {item['fear']:.1f}): {item['desc'][:100]}...\n"
        
        if 'rule_based' in final_results:
            context_str += "\n[조건 부합 추천 (Rule-Based)]\n"
            for i, item in enumerate(final_results['rule_based']):
                context_str += f"- {item['title']} (만족도 {item['rating']:.1f}, 공포 {item['fear']:.1f}): {item['desc'][:100]}...\n"

        intro_msg = ""
        if exclude_ids: 
            intro_msg = "이전 추천을 제외하고,"
        
        if final_context:
            target_str = f"{final_context}님" if isinstance(final_context, str) else "그룹 멤버분들"
            intro_msg += f" {target_str}의 취향과 요청하신 조건을 모두 고려하여"
        else:
            intro_msg += " 요청하신 조건에 맞춰"
        
        system_prompt = f"""
        당신은 방탈출 추천 AI '코난'입니다.
        
        [상황]
        - 사용자 질문: "{user_query}"
        - 추천 근거: {intro_msg} 테마를 선정했습니다.
        
        [검색된 테마 목록]
        {context_str}
        
        [지시사항]
        1. 취향 맞춤 추천과 조건 부합 추천 결과를 종합하여 설명하세요.
        2. 각 추천의 특징(만족도, 공포도 등)을 언급하며 왜 추천했는지 알려주세요.
        3. 친절한 탐정 말투로 답변하세요.
        """

        # Groq 호출
        response_text = self._call_llm(system_prompt)
        if not response_text:
            response_text = "죄송합니다. 답변 생성 중 오류가 발생했습니다."

        return response_text, final_results, filters_to_use, action