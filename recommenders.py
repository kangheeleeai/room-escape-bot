import numpy as np
from database import firestore, Vector, DistanceMeasure, FieldFilter
from utils import sort_candidates_by_query
from config import PROJECT_ID

class RuleBasedRecommender:
    def __init__(self, db):
        self.db = db

    def search_themes(self, criteria, user_query="", limit=30, nicknames=None, exclude_ids=None, log_func=None):
        if log_func: log_func(f"[Rule] 검색 시작 (지역: {criteria.get('location', '전체')})")
        
        played_theme_ids = set()
        
        # 1. 이력 조회
        target_users = []
        if isinstance(nicknames, str):
            target_users = [n.strip() for n in nicknames.split(',') if n.strip()]
        elif isinstance(nicknames, list):
            target_users = nicknames

        if target_users:
            try:
                users_ref = self.db.collection('users')
                if len(target_users) > 10: target_users = target_users[:10]
                user_q = users_ref.where(filter=FieldFilter("nickname", "in", target_users))
                user_docs = list(user_q.stream())
                
                for u_doc in user_docs:
                    played = u_doc.to_dict().get('played', [])
                    for pid in played: played_theme_ids.add(int(pid))
                
                if log_func: log_func(f"   -> {len(target_users)}명 플레이 기록 {len(played_theme_ids)}개 제외")
            except Exception as e:
                if log_func: log_func(f"   ⚠️ 이력 조회 에러: {e}")

        total_exclude_ids = set(exclude_ids) if exclude_ids else set()
        total_exclude_ids.update(played_theme_ids)

        # 2. DB 쿼리
        themes_ref = self.db.collection('themes')
        # RuleBased는 평점순 정렬이 기본
        query = themes_ref.order_by('satisfyTotalRating', direction=firestore.Query.DESCENDING).limit(200)

        docs = list(query.stream())
        raw_candidates = []
        loc_input = criteria.get('location', '').replace(" ", "") if criteria.get('location') else ""

        for doc in docs:
            data = doc.to_dict()
            
            # ID 제외
            try:
                tid = int(data.get('ref_id') or doc.id)
                if tid in total_exclude_ids or str(tid) in total_exclude_ids: continue
            except: 
                if doc.id in total_exclude_ids: continue

            # 지역 필터
            if loc_input:
                db_loc = f"{data.get('location', '')} {data.get('store_name', '')}".replace(" ", "")
                if loc_input not in db_loc: continue

            # 벡터 저장
            vec_obj = data.get('embedding_field')
            vector = None
            try:
                if vec_obj: vector = vec_obj.to_map()['value'] if hasattr(vec_obj, 'to_map') else list(vec_obj)
            except: pass

            raw_candidates.append({
                'id': doc.id,
                'title': data.get('title'),
                'store': data.get('store_name'),
                'location': data.get('location'),
                'desc': data.get('description', '')[:100],
                'rating': float(data.get('satisfyTotalRating') or 0),
                'fear': float(data.get('fearTotalRating') or 0),
                'activity': float(data.get('activityTotalRating') or 0),
                'difficulty': float(data.get('difficultyTotalRating') or 0),
                'vector': vector
            })

        sorted_candidates = sort_candidates_by_query(raw_candidates, user_query)
        if log_func: log_func(f"   -> [Rule] 필터링 후 {len(sorted_candidates)}개 후보 발견")
        
        return sorted_candidates[:limit]


class VectorRecommender:
    def __init__(self, db, model):
        self.db = db
        self.model = model

    def get_group_vector(self, nicknames, log_func=None):
        target_users = [n.strip() for n in nicknames if n.strip()] if isinstance(nicknames, list) else ([n.strip() for n in nicknames.split(',')] if nicknames else [])
        if not target_users: return None

        try:
            users_ref = self.db.collection('users')
            if len(target_users) > 10: target_users = target_users[:10]
            docs = list(users_ref.where(filter=FieldFilter("nickname", "in", target_users)).stream())
            
            vectors = []
            for doc in docs:
                vec_obj = doc.to_dict().get('embedding_field')
                if vec_obj:
                    try:
                        v = vec_obj.to_map()['value'] if hasattr(vec_obj, 'to_map') else list(vec_obj)
                        vectors.append(v)
                    except: pass
            
            if not vectors: return None
            
            mean_vector = np.mean(np.array(vectors), axis=0)
            norm = np.linalg.norm(mean_vector)
            return (mean_vector / norm).tolist() if norm > 0 else mean_vector.tolist()

        except Exception as e:
            if log_func: log_func(f"   ⚠️ 벡터 계산 실패: {e}")
            return None

    def _get_played_ids_internal(self, user_context, log_func=None):
        """RuleBasedRecommender와 동일한 방식으로 플레이 이력을 조회"""
        played_ids = set()
        if not user_context: return played_ids
        
        target_users = []
        if isinstance(user_context, str):
            target_users = [n.strip() for n in user_context.split(',') if n.strip()]
        elif isinstance(user_context, list):
            target_users = user_context
            
        if not target_users: return played_ids

        try:
            users_ref = self.db.collection('users')
            if len(target_users) > 10: target_users = target_users[:10]
            # IN query
            user_q = users_ref.where(filter=FieldFilter("nickname", "in", target_users))
            docs = user_q.stream()
            for doc in docs:
                played = doc.to_dict().get('played', [])
                for pid in played:
                    played_ids.add(int(pid))
            if log_func: log_func(f"   -> [Vector] {len(target_users)}명 이력 {len(played_ids)}개 로드")
        except Exception as e:
            if log_func: log_func(f"   ⚠️ [Vector] 이력 조회 에러: {e}")
            
        return played_ids

    def _execute_vector_search(self, vector, limit=20, filters=None, exclude_ids=None, log_func=None):
        """
        [수정된 로직]
        1. 지역 기준(또는 전체)으로 테마 로드
        2. 제외 ID 필터링
        3. 메모리 상에서 Cosine 유사도 계산 및 정렬
        """
        try:
            themes_ref = self.db.collection('themes')
            query = themes_ref
            
            # 1. 지역 기준 로드 (DB 필터)
            # 정확한 지역명이 있으면 DB에서 1차 필터링하여 성능 최적화
            loc_filter = filters.get('location') if filters else None
            if loc_filter:
                # Firestore에서는 정확한 일치만 where로 가능. 
                # (예: "강남" 입력 시 DB에 "강남"이어야 함. "서울 강남구"면 못 찾음)
                # 데이터가 정규화되어 있지 않다면 전체 로드 후 메모리 필터링이 안전함.
                # 여기서는 '전체 로드' 전략 사용 (데이터가 아주 많지 않다고 가정)
                pass 
            
            # 제한 없이 전체 로드 (전수 조사)
            docs = list(query.stream())
            
            candidates = []
            
            # 타겟 벡터 정규화 (유사도 계산용)
            target_vec = np.array(vector)
            target_norm = np.linalg.norm(target_vec)
            
            loc_input = loc_filter.replace(" ", "") if loc_filter else ""
            total_exclude_ids = set(exclude_ids) if exclude_ids else set()

            for doc in docs:
                data = doc.to_dict()
                
                # 2. 제외 ID 필터링
                try:
                    tid = int(data.get('ref_id') or doc.id)
                    if tid in total_exclude_ids or str(tid) in total_exclude_ids: continue
                except:
                    if doc.id in total_exclude_ids: continue

                # 지역 필터 (메모리 유연 검색)
                if loc_input:
                    db_loc = f"{data.get('location', '')} {data.get('store_name', '')}".replace(" ", "")
                    if loc_input not in db_loc: continue
                
                # 벡터 유사도 계산
                vec_obj = data.get('embedding_field')
                if not vec_obj: continue # 벡터 없으면 패스
                
                try:
                    theme_vec = vec_obj.to_map()['value'] if hasattr(vec_obj, 'to_map') else list(vec_obj)
                    theme_vec = np.array(theme_vec)
                    theme_norm = np.linalg.norm(theme_vec)
                    
                    if target_norm > 0 and theme_norm > 0:
                        score = np.dot(target_vec, theme_vec) / (target_norm * theme_norm)
                    else:
                        score = 0
                except:
                    score = 0
                
                candidates.append({
                    'id': doc.id,
                    'title': data.get('title'),
                    'store': data.get('store_name'),
                    'location': data.get('location'),
                    'desc': data.get('description', '')[:100],
                    'rating': float(data.get('satisfyTotalRating') or 0),
                    'fear': float(data.get('fearTotalRating') or 0),
                    'difficulty': float(data.get('difficultyTotalRating') or 0),
                    'score': score
                })

            # 3. 유사도 정렬
            candidates.sort(key=lambda x: x['score'], reverse=True)
            
            if log_func: log_func(f"   -> [Vector] {len(candidates)}개 후보 중 Top {limit} 추출")
            return candidates[:limit]

        except Exception as e:
            if log_func: log_func(f"   ❌ [Error] Vector Search 실패: {e}")
            return []

    def recommend_by_text(self, query_text, filters=None, exclude_ids=None, log_func=None):
        """텍스트 기반 검색 (이력 제외는 외부 exclude_ids에 의존)"""
        if not self.model: return []
        if log_func: log_func(f"[Text] '{query_text}' 임베딩 검색 시작")
        query_vector = self.model.encode(query_text).tolist()
        return self._execute_vector_search(query_vector, limit=10, filters=filters, exclude_ids=exclude_ids, log_func=log_func)

    def recommend_by_user_search(self, user_context, limit=3, filters=None, exclude_ids=None, log_func=None):
        """유저/그룹 벡터 검색 (내부에서 이력 조회하여 추가 제외)"""
        if log_func: log_func(f"[Person] '{user_context}' 그룹 벡터 분석")
        
        # 1. 그룹 벡터 계산
        target_vec = self.get_group_vector(user_context, log_func)
        if not target_vec: return []
        
        # 2. 이력 조회 및 병합
        played_ids = self._get_played_ids_internal(user_context, log_func)
        final_exclude = set(exclude_ids) if exclude_ids else set()
        final_exclude.update(played_ids)
        
        # 3. 검색 실행
        return self._execute_vector_search(target_vec, limit=limit, filters=filters, exclude_ids=final_exclude, log_func=log_func)
