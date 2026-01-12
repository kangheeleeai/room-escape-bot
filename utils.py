def sort_candidates_by_query(candidates, user_query):
    """
    사용자 쿼리(user_query)에 포함된 키워드(공포, 활동성 등)를 분석하여
    후보 리스트(candidates)를 재정렬합니다.
    
    모든 정렬은 기본적으로 '조건 충족도 우선' -> '만족도(평점) 차순' 입니다.
    """
    if not candidates: return []
    query_text = user_query if user_query else ""
    
    # 딕셔너리 키 안전 접근을 위한 헬퍼 (기본값 0)
    def get_val(item, key):
        try:
            val = item.get(key)
            if val is None: return 0.0
            return float(val)
        except:
            return 0.0

    # 1. 공포/비공포
    if "안무서운" in query_text or "무섭지 않은" in query_text or "겁쟁이" in query_text or "극쫄" in query_text:
        # 공포도 낮음(ASC) -> 만족도 높음
        # reverse=True(내림차순) 정렬이므로, 작은 값이 먼저 오게 하려면 음수(-)를 취함
        candidates.sort(key=lambda x: (-get_val(x, 'fear'), get_val(x, 'rating')), reverse=True)
        
    elif "공포" in query_text or "무서운" in query_text or "호러" in query_text or "스릴러" in query_text:
        # 공포도 높음 -> 만족도 높음
        candidates.sort(key=lambda x: (get_val(x, 'fear'), get_val(x, 'rating')), reverse=True)
        
    # 2. 난이도
    elif "쉬운" in query_text or "안어려운" in query_text or "입문" in query_text or "초보" in query_text:
        # 난이도 낮음 -> 만족도 높음
        candidates.sort(key=lambda x: (-get_val(x, 'difficulty'), get_val(x, 'rating')), reverse=True)
        
    elif "문제방" in query_text or "어려운" in query_text or "문제" in query_text or "숙련자" in query_text:
        # 문제방(문제점수) or 난이도 높음 -> 만족도 높음
        if "문제방" in query_text or "문제" in query_text:
             # 문제 점수 우선
             candidates.sort(key=lambda x: (get_val(x, 'problem'), get_val(x, 'difficulty'), get_val(x, 'rating')), reverse=True)
        else:
             # 난이도 점수 우선
             candidates.sort(key=lambda x: (get_val(x, 'difficulty'), get_val(x, 'rating')), reverse=True)
        
    # 3. 활동성
    elif "활동적이지 않은" in query_text or "치마" in query_text or "힐" in query_text or "걷는" in query_text:
        # 활동성 낮음 -> 만족도 높음
        candidates.sort(key=lambda x: (-get_val(x, 'activity'), get_val(x, 'rating')), reverse=True)
        
    elif "활동" in query_text or "동적인" in query_text or "바지" in query_text or "체력" in query_text:
        # 활동성 높음 -> 만족도 높음
        candidates.sort(key=lambda x: (get_val(x, 'activity'), get_val(x, 'rating')), reverse=True)
        
    # 4. 기타 요소 (스토리, 인테리어, 장치)
    elif "스토리" in query_text or "드라마" in query_text or "감성" in query_text or "서사" in query_text:
        candidates.sort(key=lambda x: (get_val(x, 'story'), get_val(x, 'rating')), reverse=True)
        
    elif "인테리어" in query_text or "리얼리티" in query_text or "실제같은" in query_text or "배경" in query_text:
        candidates.sort(key=lambda x: (get_val(x, 'interior'), get_val(x, 'rating')), reverse=True)
        
    elif "연출" in query_text or "장치" in query_text or "화려" in query_text or "스케일" in query_text:
        candidates.sort(key=lambda x: (get_val(x, 'act'), get_val(x, 'rating')), reverse=True)
    
    else:
        # 기본: 만족도 순
        candidates.sort(key=lambda x: get_val(x, 'rating'), reverse=True)
    
    return candidates
