def sort_candidates_by_query(candidates, user_query):
    """
    사용자 쿼리(user_query)에 포함된 키워드(공포, 활동성 등)를 분석하여
    후보 리스트(candidates)를 재정렬합니다.
    """
    print(user_query)
    if not candidates: return []
    query_text = user_query if user_query else ""
    
    # 기본: 만족도(rating) 높은 순 (내림차순)
    
    if "안무서운" in query_text or "무섭지 않은" in query_text:
        # 공포도 낮고(ASC), 만족도가 높은 순
        candidates.sort(key=lambda x: (x['rating'], -x['fear']), reverse=True)
        
    elif "공포" in query_text or "무서운" in query_text or "호러" in query_text:
        # 공포도 높고, 만족도 높은 순
        candidates.sort(key=lambda x: (x['fear'], x['rating']), reverse=True)
        
    elif "쉬운" in query_text or "안어려운" in query_text:
        candidates.sort(key=lambda x: (x['rating'], -x['difficulty']), reverse=True)
        
    elif "문제방" in query_text or "어려운" in query_text or "문제" in query_text:
        candidates.sort(key=lambda x: (x['problem'], x['difficulty'], x['rating']), reverse=True)
        
    elif "활동적이지 않은" in query_text or "치마" in query_text:
        candidates.sort(key=lambda x: (x['rating'], -x['activity']), reverse=True)
        
    elif "활동" in query_text or "동적인" in query_text or "바지" in query_text:
        candidates.sort(key=lambda x: (x['activity'], x['rating']), reverse=True)
        
    elif "스토리" in query_text or "드라마" in query_text or "감성" in query_text:
        candidates.sort(key=lambda x: (x['story'], x['rating']), reverse=True)
        
    elif "인테리어" in query_text or "리얼리티" in query_text or "실제같은" in query_text:
        candidates.sort(key=lambda x: (x['interior'], x['rating']), reverse=True)
        
    elif "연출" in query_text or "장치" in query_text or "화려" in query_text or "스케일" in query_text:
        candidates.sort(key=lambda x: (x['act'], x['rating']), reverse=True)
    
    else:
        # 기본: 만족도 순
        candidates.sort(key=lambda x: x['rating'], reverse=True)
    
    return candidates
