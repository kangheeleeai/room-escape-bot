import streamlit as st
import time
from database import init_firebase
from models import load_embed_model

# [수정됨] utils가 아니라 각각의 파일에서 클래스를 가져옵니다.
from recommenders import RuleBasedRecommender, VectorRecommender
from bot_engine import EscapeBotEngine
from config import GROQ_API_KEY, TAVILY_API_KEY

st.set_page_config(page_title="방탈출 AI 코난 (Hybrid)", page_icon="🕵️", layout="wide")

def main():
    # [LOG] 앱 실행 로그
    print("\n🚀 [App] Streamlit App Rerun")

    with st.sidebar:
        st.title("⚙️ 설정")
        # 실제 사용 시 Secrets 또는 입력창 활성화
        gemini_key = GROQ_API_KEY
        tavily_key = TAVILY_API_KEY
        
        # [LOG] 키 설정 확인
        print(f"   🔑 Keys Configured: Groq={'Yes' if gemini_key else 'No'}, Tavily={'Yes' if tavily_key else 'No'}")
        
        st.divider()
        st.subheader("👥 플레이어 설정")
        
        my_name = st.text_input("내 닉네임", placeholder="예: 방탈출러", key="my_name_input")
        group_names = st.text_input("같이 할 멤버 (쉼표로 구분)", placeholder="예: 친구1, 친구2", key="group_names_input")
        
        nickname = ""
        if my_name:
            nickname = my_name.strip()
            if group_names:
                nickname += f", {group_names.strip()}"
        elif group_names:
            nickname = group_names.strip()

        if nickname:
            if ',' in nickname:
                st.caption(f"✅ 그룹 모드: {nickname}")
                print(f"   👥 Group Mode: {nickname}")
            else:
                st.caption(f"✅ '{nickname}'님의 취향을 분석합니다.")
                print(f"   👤 User Mode: {nickname}")
        
        if st.button("대화 내용 지우기"):
            st.session_state.messages = []
            st.session_state.shown_theme_ids = set()
            st.session_state.last_filters = {}
            st.session_state.last_query = ""
            print("   🧹 Session Cleared")
            st.rerun()

    st.title("🕵️ 방탈출 AI 코난 (Hybrid)")
    st.caption("벡터 검색과 필터 검색을 결합하여 최적의 테마를 찾아드립니다.")

    # 상태 초기화
    if "shown_theme_ids" not in st.session_state:
        st.session_state.shown_theme_ids = set()
    if "last_filters" not in st.session_state:
        st.session_state.last_filters = {}
    if "last_query" not in st.session_state:
        st.session_state.last_query = ""
    
    # [LOG] 세션 상태
    print(f"   📊 Session State: Shown={len(st.session_state.shown_theme_ids)}, LastQuery='{st.session_state.last_query}'")

    db = init_firebase()
    embed_model = load_embed_model()

    if not db or not embed_model:
        st.error("시스템 초기화 실패.")
        print("   ❌ System Init Failed")
        st.stop()

    # 각 모듈에서 가져온 클래스로 인스턴스 생성
    vec_rec = VectorRecommender(db, embed_model)
    rule_rec = RuleBasedRecommender(db) 
    bot_engine = EscapeBotEngine(vec_rec, rule_rec, gemini_key, tavily_key)

    if "messages" not in st.session_state:
        st.session_state.messages = [{"role": "assistant", "content": "안녕하세요! 찾으시는 지역이나 장르가 있으신가요? (예: 강남 공포 테마)"}]

    # 채팅 기록 렌더링
    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])
            
            cards_data = msg.get("cards")
            if isinstance(cards_data, dict) and cards_data:
                if 'personalized' in cards_data:
                    st.success("🎯 취향 저격 추천 (Personalized)")
                    for item in cards_data['personalized']:
                        st.text(f"• {item['title']} ({item['store']}) - {item['rating']:.1f}")

                if 'rule_based' in cards_data:
                    st.info("🔎 조건 부합 추천 (Rule-Based)")
                    for item in cards_data['rule_based']:
                        st.text(f"• {item['title']} ({item['store']}) - {item['rating']:.1f}")
                
                if 'text_search' in cards_data:
                    st.warning("🧩 유사 테마 검색 (Text-Based)")
                    for item in cards_data['text_search']:
                        st.text(f"• {item['title']} ({item['store']}) - {item['rating']:.1f}")

    if prompt := st.chat_input("메시지를 입력하세요..."):
        # [LOG] 사용자 입력
        print(f"\n📨 [Input] User: '{prompt}'")
        
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)

        reply_text = ""
        result_cards = {}

        with st.chat_message("assistant"):
            if not gemini_key:
                reply_text = "⚠️ Groq API Key가 설정되지 않았습니다."
                st.warning(reply_text)
                print("   ⚠️ Missing API Key")
            else:
                with st.spinner("단서를 수집하고 추리하는 중... 🧐"):
                    session_context = {
                        'shown_ids': st.session_state.shown_theme_ids,
                        'last_filters': st.session_state.last_filters
                    }

                    # [LOG] 엔진 호출 전
                    print(f"   🤖 Engine Call: Query='{prompt}', Context={nickname}, Exclude={len(session_context['shown_ids'])}")

                    reply_text, result_cards, used_filters, action = bot_engine.generate_reply(
                        prompt, 
                        user_context=nickname,
                        session_context=session_context
                    )
                    
                    # [LOG] 엔진 응답 수신
                    card_count = sum(len(v) for v in result_cards.values()) if result_cards else 0
                    print(f"   ✅ Engine Response: Action={action}, Cards={card_count}")

                    st.markdown(reply_text)
                    
                    if result_cards:
                        if action == 'recommend': 
                            print("   🔄 New Recommendation -> Reset Shown IDs")
                            st.session_state.shown_theme_ids = set()
                        
                        st.session_state.last_filters = used_filters
                        if not action.startswith('played_'):
                             st.session_state.last_query = prompt
                        
                        count_new = 0
                        for key in result_cards:
                            for c in result_cards[key]:
                                st.session_state.shown_theme_ids.add(c['id'])
                                count_new += 1
                        
                        print(f"   💾 Updated Session: Filters={used_filters}, Total Shown={len(st.session_state.shown_theme_ids)} (+{count_new})")

                        # 렌더링
                        if 'personalized' in result_cards:
                            st.success("🎯 취향 저격 추천 (Personalized)")
                            for item in result_cards['personalized']:
                                st.text(f"• {item['title']} ({item['store']}) - {item['rating']:.1f}")

                        if 'rule_based' in result_cards:
                            st.info("🔎 조건 부합 추천 (Rule-Based)")
                            for item in result_cards['rule_based']:
                                st.text(f"• {item['title']} ({item['store']}) - {item['rating']:.1f}")
                        
                        if 'text_search' in result_cards:
                            st.warning("🧩 유사 테마 검색 (Text-Based)")
                            for item in result_cards['text_search']:
                                st.text(f"• {item['title']} ({item['store']}) - {item['rating']:.1f}")

        st.session_state.messages.append({
            "role": "assistant", 
            "content": reply_text,
            "cards": result_cards
        })

if __name__ == "__main__":
    main()
