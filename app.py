import streamlit as st
import time
from database import init_firebase
from models import load_embed_model

from recommenders import RuleBasedRecommender, VectorRecommender
from bot_engine import EscapeBotEngine
from config import GROQ_API_KEY, TAVILY_API_KEY

st.set_page_config(page_title="방탈출 AI 코난 (Hybrid)", page_icon="🕵️", layout="wide")

def show_guide():
    """사용자 가이드 페이지를 표시합니다."""
    st.markdown("""
    ## 🕵️ 방탈출 AI 코난 사용 설명서
    
    반갑습니다! 저는 여러분의 취향에 딱 맞는 방탈출 테마를 찾아드리는 AI 탐정, 코난입니다.
    저를 200% 활용하는 방법을 알려드릴게요!

    ---

    ### 1️⃣ 기본 추천 받기 (누구나)
    채팅창에 원하시는 지역과 장르, 분위기를 자유롭게 말씀해주세요.
    
    * **"강남에서 무서운 공포 테마 추천해줘"**
    * **"홍대 활동성 많은거 있어?"**
    * **"건대 감성 테마 추천좀"**
    * **"스토리 좋은 인생 테마 찾고 있어"**
    
    ### 2️⃣ 나만의 맞춤 추천 (닉네임 입력)
    왼쪽 사이드바에 **'내 닉네임'**을 입력하시면, 그동안의 플레이 기록과 취향을 분석해 **저격 추천**을 해드립니다.
    
    * **Rule-Based:** 여러분의 질문 조건에 딱 맞는 테마를 찾고,
    * **Personalized:** 여러분이 좋아할 만한 숨겨진 명작을 찾아냅니다.
    
    ### 3️⃣ 친구와 함께! (그룹 추천)
    같이 갈 친구가 있나요? 사이드바의 **'같이 할 멤버'** 칸에 친구 닉네임을 적어주세요. (쉼표 `,`로 구분)
    
    * **그룹 취향 분석:** 멤버들의 공통적인 취향(교집합)을 찾아 모두가 만족할 테마를 추천합니다.
    * **안 해본 테마만:** 멤버 중 한 명이라도 플레이한 기록이 있다면 추천에서 제외합니다.
    
    ### 4️⃣ 플레이 기록 관리 ("했어/안했어")
    추천받은 테마를 이미 하셨나요? 채팅으로 바로 알려주세요.
    
    * **"강남 링 테마는 이미 했어"** -> 플레이 목록에 추가하고 다음부터 추천하지 않습니다.
    * **"홍대 삐릿뽀 안했어"** -> 실수로 추가된 기록을 취소합니다.
    
    ### 5️⃣ 마음에 안 드시나요? ("다른거")
    추천 결과가 별로라면 **"다른거 추천해줘"**라고 말해보세요.
    
    * 이전에 보여드린 테마는 제외하고, 차순위의 새로운 테마들을 보여드립니다.
    * 기존 검색 조건(지역, 장르 등)은 그대로 유지됩니다.

    ---
    **자, 그럼 이제 사건을 의뢰하러 가보실까요? 🧐**
    """)

def main():
    with st.sidebar:
        st.title("⚙️ 설정")
        
        # [NEW] 페이지 선택 메뉴
        page = st.radio("이동", ["🤖 챗봇 사용하기", "📖 사용 가이드"])
        
        st.divider()
        
        # 실제 사용 시 Secrets 또는 입력창 활성화
        gemini_key = GROQ_API_KEY
        tavily_key = TAVILY_API_KEY
        
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
            else:
                st.caption(f"✅ '{nickname}'님의 취향을 분석합니다.")
        
        if st.button("대화 내용 지우기"):
            st.session_state.messages = []
            st.session_state.shown_theme_ids = set()
            st.session_state.last_filters = {}
            st.session_state.last_query = ""
            st.rerun()

    if page == "📖 사용 가이드":
        show_guide()
        return  # 가이드만 보여주고 함수 종료

    # --- 기존 챗봇 UI 시작 ---
    st.title("🕵️ 방탈출 AI 코난 (Hybrid)")
    st.caption("벡터 검색과 필터 검색을 결합하여 최적의 테마를 찾아드립니다.")

    # 상태 초기화
    if "shown_theme_ids" not in st.session_state:
        st.session_state.shown_theme_ids = set()
    if "last_filters" not in st.session_state:
        st.session_state.last_filters = {}
    if "last_query" not in st.session_state:
        st.session_state.last_query = ""

    db = init_firebase()
    embed_model = load_embed_model()

    if not db or not embed_model:
        st.error("시스템 초기화 실패.")
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
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)

        reply_text = ""
        result_cards = {}

        with st.chat_message("assistant"):
            if not gemini_key:
                reply_text = "⚠️ Groq API Key가 설정되지 않았습니다."
                st.warning(reply_text)
            else:
                with st.spinner("단서를 수집하고 추리하는 중... 🧐"):
                    session_context = {
                        'shown_ids': st.session_state.shown_theme_ids,
                        'last_filters': st.session_state.last_filters
                    }

                    reply_text, result_cards, used_filters, action = bot_engine.generate_reply(
                        prompt, 
                        user_context=nickname,
                        session_context=session_context
                    )
                    st.markdown(reply_text)
                    
                    if result_cards:
                        if action == 'recommend': 
                            st.session_state.shown_theme_ids = set()
                        
                        st.session_state.last_filters = used_filters
                        if not action.startswith('played_'):
                             st.session_state.last_query = prompt
                        
                        for key in result_cards:
                            for c in result_cards[key]:
                                st.session_state.shown_theme_ids.add(c['id'])

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
