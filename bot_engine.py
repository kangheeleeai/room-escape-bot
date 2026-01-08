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
                temperature=0.1, # 의도 분석은 정확도가 중요하므로 온도를 낮춤
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
            
            # 지역 정보가 있으면 필터링 (없으면 전체 검색 - 주의)
            if location:
                # DB의 location 값과 정확히 일치하지 않을 수 있으므로 (예: 강남구 vs 강남)
                # 여기서는 query를 느슨하게 하거나, 클라이언트 필터링 사용
                # 성능을 위해 일단 location 필터 적용
                query = query.where(filter=FieldFilter("location", "==", location))
            
            # 검색 범위 확장 (500개)
            docs = list(query.limit(500).stream())
            
            target_name = theme_name.replace(" ", "")
            for doc in docs:
                data = doc.to_dict()
                title = data.get('title', '')
                # 공백 제거 후 포함 여부 확인
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
            
            if not docs: return "❌ 유저 미등록"
            
            user_doc = docs[0]
            if action == "played_check":
                user_doc.reference.update({"played": firestore.ArrayUnion([theme_id])})
                return "추가 성공"
            elif action == "not_played_check":
                user_doc.reference.update({"played": firestore.ArrayRemove([theme_id])})
                return "삭제 성공"
            return "알 수 없는 요청"
        except Exception as e:
            return f"에러: {e}"

    def analyze_user_intent(self, user_query):
        print(f"🧠 [LLM] 사용자 의도 분석 요청... Query: '{user_query}'")
        if not self.groq_client: return {}
        
        prompt = f"""
        사용자의 질문을 분석하여 방탈출 챗봇의 의도(Intent)와 정보를 추출하세요.

        질문: "{user_query}"

        [분석 규칙]
        1. "played_check_inquiry": 사용자가 플레이 기록을 어떻게 남기는지 묻거나, 단순히 "플레이한 테마", "기록 추가" 라고만 말했을 때.
        2. "played_check": 사용자가 특정 테마를 했다고 말할 때. (예: "강남 링 했어", "[홍대, 삐릿뽀], [강남, 네드] 했어")
        3. "not_played_check": 사용자가 안 했다고 하거나 취소할 때.
        4. "recommend": 새로운 추천 요청.
        5. "another_recommend": 다른 거 추천 요청.

        [추출 필드]
        - items: 플레이 기록 관련일 때, {{"location": "지역", "theme": "테마명"}} 객체들의 리스트. 
                 사용자가 "[강남, 링]" 처럼 입력하면 이를 파싱해서 넣으세요.
        - location: (단일 추천용) 지역명.
        - keywords: (추천용) 장르, 분위기 등 키워드.
        - mentioned_users: 언급된 닉네임.

        JSON 형식으로만 반환하세요.
        예시 1: {{ "action": "played_check", "items": [{{"location": "강남", "theme": "링"}}, {{"location": "홍대", "theme": "비트포비아"}}] }}
        예시 2: {{ "action": "played_check_inquiry" }}
        """
        try:
            result_str = self._call_llm(prompt, json_mode=True)
            result = json.loads(result_str) if result_str else {"action": "recommend"}
            print(f"   -> 분석 결과: {result}")
            return result
        except Exception as e:
            print(f"   ❌ 의도 분석 실패: {e}")
            return {"action": "recommend"}

    def generate_reply(self, user_query, user_context=None, session_context=None):
        if not self.groq_client:
            return "⚠️ Groq API Key가 설정되지 않았습니다.", {}, {}, "error"

        # 1. 의도 분석
        intent_data = self.analyze_user_intent(user_query)
        action = intent_data.get('action', 'recommend')

        # 2. 플레이 기록 문의 처리 ("플레이한 테마")
        if action == "played_check_inquiry":
            msg = "플레이한 테마를 **[지역, 테마명], [지역, 테마명]** 과 같이 입력해주시면 기록해 드릴게요!\n예시: `[강남, 링], [홍대, 삐릿뽀]`"
            return msg, {}, {}, action

        # 3. 플레이 기록 추가/삭제 처리 (다중 처리 지원)
        if action in ['played_check', 'not_played_check']:
            if not user_context:
                return "⚠️ 플레이 기록을 관리하려면 닉네임 입력이 필요합니다.", {}, {}, action
            
            # items 리스트가 있으면 우선 사용, 없으면 단일 location/theme 사용
            items = intent_data.get('items', [])
            if not items and intent_data.get('theme'):
                items.append({
                    "location": intent_data.get('location'),
                    "theme": intent_data.get('theme')
                })

            if not items:
                return "⚠️ 테마 정보를 인식하지 못했습니다. '[지역, 테마명]' 형식으로 말씀해 주세요.", {}, {}, action

            results_msg = []
            success_count = 0
            
            for item in items:
                loc = item.get('location')
                theme = item.get('theme')
                if theme:
                    tid = self.find_theme_id(loc, theme)
                    if tid:
                        res = self.update_play_history(user_context, tid, action)
                        if "성공" in res: success_count += 1
                        results_msg.append(f"- {theme}: {res}")
                    else:
                        results_msg.append(f"- {theme}: ⚠️ 테마를 찾을 수 없음")
            
            summary = f"총 {len(items)}건 중 {success_count}건 처리 완료.\n" + "\n".join(results_msg)
            return summary, {}, {}, action

        # 3. 필터 설정 (추천 로직)
        current_filters = {
            'location': intent_data.get('location'),
            'keywords': intent_data.get('keywords', []),
            'mentioned_users': intent_data.get('mentioned_users', [])
        }

        # ... (이하 그룹 멤버 확인 및 추천 로직은 기존과 동일) ...
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
        
        filters_to_use = {}
        exclude_ids = []
        
        if action == 'another_recommend':
            if session_context:
                filters_to_use = session_context.get('last_filters', {})
                exclude_ids = list(session_context.get('shown_ids', []))
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

        # Vector (Personalized)
        if final_context:
            candidates_vector = self.vector_recommender.recommend_by_user_search(
                final_context, limit=3, filters=filters_to_use, exclude_ids=exclude_ids
            )
            if candidates_vector:
                final_reranked = sort_candidates_by_query(candidates_vector, user_query)
                final_results['personalized'] = final_reranked

        # Fallback
        if not final_results:
            candidates_text = self.vector_recommender.recommend_by_text(
                user_query, filters=filters_to_use, exclude_ids=exclude_ids
            )
            if candidates_text:
                final_results['text_search'] = sort_candidates_by_query(candidates_text, user_query)[:3]
            else:
                return "죄송합니다. 조건에 맞는 테마를 찾지 못했습니다. 😭\n조건을 변경해서 다시 질문해 주시겠어요?", {}, filters_to_use, action

        # LLM 설명 생성
        context_str = ""
        if 'personalized' in final_results:
            context_str += "\n[취향 맞춤 추천]\n"
            for i, item in enumerate(final_results['personalized']):
                context_str += f"- {item['title']} (만족도 {item['rating']:.1f}): {item['desc'][:80]}...\n"
        if 'rule_based' in final_results:
            context_str += "\n[조건 부합 추천]\n"
            for i, item in enumerate(final_results['rule_based']):
                context_str += f"- {item['title']} (만족도 {item['rating']:.1f}): {item['desc'][:80]}...\n"

        intro_msg = "이전 추천 제외" if exclude_ids else "요청하신 조건에 맞춰"
        
        system_prompt = f"""
        당신은 방탈출 추천 AI입니다.
        [상황] 질문: "{user_query}" / 근거: {intro_msg}
        [목록] {context_str}
        [지시] 위 목록에서 1~2개를 골라 추천 이유(특징, 평점)를 섞어 친절하게 설명하세요.
        """

        response_text = self._call_llm(system_prompt)
        if not response_text:
            response_text = "죄송합니다. 답변 생성 중 오류가 발생했습니다."

        return response_text, final_results, filters_to_use, action
