import streamlit as st
import time
from database import init_firebase
from models import load_embed_model

from recommenders import RuleBasedRecommender, VectorRecommender
from bot_engine import EscapeBotEngine
from config import GROQ_API_KEY, TAVILY_API_KEY

st.set_page_config(page_title="방탈출 AI 코난 (Hybrid)", page_icon="🕵️", layout="wide")

# CSS 스타일 주입 (카드 디자인 등)
st.markdown("""
<style>
    .theme-card {
        background-color: #f0f2f6;
        padding: 15px;
        border-radius: 10px;
        margin-bottom: 10px;
        border-left: 5px solid #ff4b4b;
    }
    .theme-title { font-weight: bold; font-size: 1.1em; }
    .theme-meta { font-size: 0.9em; color: #555; }
</style>
""", unsafe_allow_html=True)

def show_guide():
    """사용자 가이드 페이지를 표시합니다."""
    st.markdown("""
    ## 🕵️ 방탈출 AI 코난 사용 설명서
    
    ### 1️⃣ 기본 추천
    * "강남 공포 테마 추천해줘"
    * "홍대 활동성 많은거"
    
    ### 2️⃣ 닉네임 맞춤 추천
    * 왼쪽 사이드바에 닉네임을 입력하면 **내 플레이 기록**을 제외하고 추천합니다.
    * 친구들과 함께라면 쉼표(`,`)로 여러 명을 입력하세요.
    
    ### 3️⃣ 기록 관리
    * "**강남 링 했어**" -> 플레이 목록에 추가
    * "**홍대 삐릿뽀 안했어**" -> 기록 취소
    """)

def main():
    with st.sidebar:
        st.title("⚙️ 설정 & 프로필")
        
        page = st.radio("이동", ["🤖 챗봇", "📖 가이드"])
        st.divider()
        
        st.subheader("👥 플레이어 정보")
        my_name = st.text_input("내 닉네임", placeholder="예: 코난", key="my_name_input")
        group_names = st.text_input("같이 할 멤버 (옵션)", placeholder="예: 미란이, 장미", key="group_names_input")
        
        nickname = my_name.strip()
        if group_names:
            nickname = f"{nickname}, {group_names}".strip(", ") if nickname else group_names

        if nickname:
            st.success(f"로그인: {nickname}")
        else:
            st.info("닉네임을 입력하면 맞춤 추천이 가능합니다.")
            
        st.divider()
        debug_mode = st.toggle("🐛 디버그 모드", value=False, help="봇의 의도 분석 결과와 필터 정보를 보여줍니다.")
        
        if st.button("🗑️ 대화 초기화"):
            st.session_state.messages = []
            st.session_state.shown_theme_ids = set()
            st.session_state.last_filters = {}
            st.rerun()

    if page == "📖 가이드":
        show_guide()
        return

    # --- 메인 챗봇 로직 ---
    st.title("🕵️ 방탈출 AI 코난")
    st.caption("Hybrid Recommender System (Rule-based + Vector)")

    # 초기화
    if "messages" not in st.session_state:
        st.session_state.messages = [{"role": "assistant", "content": "어떤 방탈출 테마를 찾으시나요? 지역이나 장르를 말씀해주세요!"}]
    if "shown_theme_ids" not in st.session_state:
        st.session_state.shown_theme_ids = set()
    if "last_filters" not in st.session_state:
        st.session_state.last_filters = {}

    # 리소스 로드
    db = init_firebase()
    embed_model = load_embed_model()

    if not db:
        st.error("🔥 Firebase 연결 실패. 서비스 계정 키 또는 Secrets 설정을 확인하세요.")
        st.stop()
    
    # 인스턴스 생성
    vec_rec = VectorRecommender(db, embed_model)
    rule_rec = RuleBasedRecommender(db) 
    bot_engine = EscapeBotEngine(vec_rec, rule_rec, GROQ_API_KEY, TAVILY_API_KEY)

    # 채팅 기록 표시
    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])
            # 카드 정보가 있으면 렌더링
            cards = msg.get("cards", {})
            debug_info = msg.get("debug_info", {})
            
            if cards:
                # 탭으로 추천 유형 분리
                tab1, tab2, tab3 = st.tabs(["🎯 맞춤 추천", "🔎 조건 추천", "🧩 유사 검색"])
                
                with tab1:
                    if 'personalized' in cards:
                        for item in cards['personalized']:
                            st.markdown(f"""
                            <div class='theme-card'>
                                <div class='theme-title'>{item['title']} <span style='font-size:0.8em; color:gray'>({item['store']})</span></div>
                                <div class='theme-meta'>⭐ 평점: {item['rating']} | 📍 {item['location']}</div>
                                <div style='font-size:0.9em; margin-top:5px;'>{item['desc']}</div>
                            </div>
                            """, unsafe_allow_html=True)
                    else:
                        st.caption("맞춤 추천 결과가 없습니다.")
                        
                with tab2:
                    if 'rule_based' in cards:
                        for item in cards['rule_based']:
                            st.markdown(f"**{item['title']}** ({item['store']}) - ⭐{item['rating']}")
                    else:
                        st.caption("조건 검색 결과가 없습니다.")
                
                with tab3:
                    if 'text_search' in cards:
                        for item in cards['text_search']:
                            st.markdown(f"- {item['title']}")
                    else:
                        st.caption("유사 검색 결과가 없습니다.")
            
            # 디버그 정보 표시 (토글이 켜져있을 때만)
            if debug_mode and debug_info:
                with st.expander("🛠️ 디버그 정보"):
                    st.json(debug_info)

    # 입력 처리
    if prompt := st.chat_input("메시지를 입력하세요..."):
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)

        with st.chat_message("assistant"):
            if not GROQ_API_KEY:
                st.error("API Key가 없습니다. 설정 파일(secrets.toml)을 확인해주세요.")
            else:
                with st.spinner("단서를 분석하고 있습니다... 🧐"):
                    session_ctx = {
                        'shown_ids': st.session_state.shown_theme_ids,
                        'last_filters': st.session_state.last_filters
                    }

                    # 봇 엔진 호출 (debug_info 리턴값 추가됨)
                    reply_text, result_cards, used_filters, action, debug_data = bot_engine.generate_reply(
                        prompt, 
                        user_context=nickname,
                        session_context=session_ctx
                    )
                    
                    st.markdown(reply_text)
                    
                    if debug_mode:
                        with st.expander("🛠️ 실시간 분석 로그"):
                            st.json(debug_data)
                            st.write(f"Action: {action}")
                            st.write(f"Applied Filters: {used_filters}")

                    # 상태 업데이트
                    if result_cards:
                        if action == 'recommend': 
                            st.session_state.shown_theme_ids = set() # 새 추천이면 리셋
                        
                        st.session_state.last_filters = used_filters
                        
                        # 보여준 ID 저장 (중복 추천 방지)
                        for key in result_cards:
                            for c in result_cards[key]:
                                st.session_state.shown_theme_ids.add(c['id'])

        # 메시지 기록 저장 (디버그 정보 포함)
        st.session_state.messages.append({
            "role": "assistant", 
            "content": reply_text,
            "cards": result_cards,
            "debug_info": debug_data if debug_mode else {}
        })
        st.rerun() # UI 즉시 갱신

if __name__ == "__main__":
    main()
