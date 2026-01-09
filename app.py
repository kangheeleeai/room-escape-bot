import streamlit as st
import time
import logging
from database import init_firebase
from models import load_embed_model

from recommenders import RuleBasedRecommender, VectorRecommender
from bot_engine import EscapeBotEngine
from config import GROQ_API_KEY, TAVILY_API_KEY

# 기본 로깅 설정 (터미널용)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler()]
)
logger = logging.getLogger(__name__)

st.set_page_config(page_title="방탈출 AI 코난 (Hybrid)", page_icon="🕵️", layout="wide")

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

    if "messages" not in st.session_state:
        st.session_state.messages = [{"role": "assistant", "content": "어떤 방탈출 테마를 찾으시나요? 지역이나 장르를 말씀해주세요!"}]
    if "shown_theme_ids" not in st.session_state:
        st.session_state.shown_theme_ids = set()
    if "last_filters" not in st.session_state:
        st.session_state.last_filters = {}

    db = init_firebase()
    embed_model = load_embed_model()

    if not db:
        st.error("🔥 Firebase 연결 실패. 서비스 계정 키 또는 Secrets 설정을 확인하세요.")
        st.stop()
    
    vec_rec = VectorRecommender(db, embed_model)
    rule_rec = RuleBasedRecommender(db) 
    bot_engine = EscapeBotEngine(vec_rec, rule_rec, GROQ_API_KEY, TAVILY_API_KEY)

    # 채팅 기록 표시
    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])
            
            cards = msg.get("cards", {})
            debug_info = msg.get("debug_info", {})
            logs = msg.get("logs", []) # 저장된 로그 확인

            if logs:
                with st.expander("📜 처리 과정 로그 보기"):
                    for l in logs:
                        st.text(l)

            if cards:
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
                        st.caption("결과 없음")
                with tab2:
                    if 'rule_based' in cards:
                        for item in cards['rule_based']:
                            st.markdown(f"**{item['title']}** ({item['store']}) - ⭐{item['rating']}")
                    else:
                        st.caption("결과 없음")
                with tab3:
                    if 'text_search' in cards:
                        for item in cards['text_search']:
                            st.markdown(f"- {item['title']}")
                    else:
                        st.caption("결과 없음")
            
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
                st.error("API Key가 없습니다.")
            else:
                # [핵심 변경] st.status를 사용하여 실시간 로그 출력
                process_logs = []
                with st.status("🕵️ 코난이 추리 중입니다...", expanded=True) as status:
                    
                    # UI에 로그를 찍고 리스트에도 저장하는 콜백 함수
                    def ui_logger(msg):
                        st.write(f"🔹 {msg}") # status 컨테이너 안에 출력
                        process_logs.append(msg)
                        logger.info(msg) # 터미널에도 출력

                    session_ctx = {
                        'shown_ids': st.session_state.shown_theme_ids,
                        'last_filters': st.session_state.last_filters
                    }

                    # bot_engine에 로거 전달
                    reply_text, result_cards, used_filters, action, debug_data = bot_engine.generate_reply(
                        prompt, 
                        user_context=nickname,
                        session_context=session_ctx,
                        on_log=ui_logger  # <--- 콜백 전달
                    )
                    
                    status.update(label="추리 완료!", state="complete", expanded=False)

                st.markdown(reply_text)
                
                # 상태 업데이트
                if result_cards:
                    if action == 'recommend': 
                        st.session_state.shown_theme_ids = set()
                    st.session_state.last_filters = used_filters
                    for key in result_cards:
                        for c in result_cards[key]:
                            st.session_state.shown_theme_ids.add(c['id'])

        st.session_state.messages.append({
            "role": "assistant", 
            "content": reply_text,
            "cards": result_cards,
            "debug_info": debug_data if debug_mode else {},
            "logs": process_logs # 로그도 기록에 저장
        })
        st.rerun()

if __name__ == "__main__":
    main()
