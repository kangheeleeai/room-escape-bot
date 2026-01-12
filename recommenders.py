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

    def _execute_vector_search(self, vector, limit=20, filters=None, exclude_ids=None, log_func=None):
        try:
            fetch_limit = limit + (len(exclude_ids) if exclude_ids else 0) + 20
            themes_ref = self.db.collection('themes')
            if fetch_limit < 500:
                fetch_limit = 500
            if log_func: log_func(f"[fetch_limit, exclude_ids] '{fetch_limit}', '{exclude_ids}")
            
            
            vector_query = themes_ref.find_nearest(
                vector_field="embedding_field",
                query_vector=Vector(vector),
                distance_measure=DistanceMeasure.COSINE,
                limit=fetch_limit
            )
            docs = vector_query.get()
            
            results = []
            loc_filter = filters.get('location', '').replace(" ", "") if filters else ""

            for doc in docs:
                if exclude_ids and (doc.id in exclude_ids or str(doc.id) in exclude_ids): continue
                data = doc.to_dict()
                if loc_filter:
                    db_loc = f"{data.get('location', '')} {data.get('store_name', '')}".replace(" ", "")
                    if loc_filter not in db_loc: continue
                    
                results.append({
                    'id': doc.id,
                    'title': data.get('title'),
                    'store': data.get('store_name'),
                    'location': data.get('location'),
                    'desc': data.get('description', '')[:100],
                    'rating': float(data.get('satisfyTotalRating') or 0),
                    'fear': float(data.get('fearTotalRating') or 0),
                    'difficulty': float(data.get('difficultyTotalRating') or 0),
                })
                if len(results) >= limit: break
            
            if log_func: log_func(f"   -> [Vector] 유사 테마 {len(results)}개 찾음")
            return results
        except Exception as e:
            if log_func: log_func(f"   ❌ Vector Search 실패: {e}")
            return []

    def recommend_by_text(self, query_text, filters=None, exclude_ids=None, log_func=None):
        if not self.model: return []
        if log_func: log_func(f"[Text] '{query_text}' 임베딩 변환 및 검색")
        query_vector = self.model.encode(query_text).tolist()
        return self._execute_vector_search(query_vector, filters=filters, exclude_ids=exclude_ids, log_func=log_func)

    def recommend_by_user_search(self, user_context, limit=3, filters=None, exclude_ids=None, log_func=None):
        if log_func: log_func(f"[Person] '{user_context}' 그룹 취향 분석")
        target_vec = self.get_group_vector(user_context, log_func)
        if not target_vec: return []
        return self._execute_vector_search(target_vec, limit=limit, filters=filters, exclude_ids=exclude_ids, log_func=log_func)
