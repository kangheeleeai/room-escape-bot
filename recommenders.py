import numpy as np
from database import firestore, Vector, DistanceMeasure, FieldFilter
from utils import sort_candidates_by_query
from config import PROJECT_ID

class RuleBasedRecommender:
    def __init__(self, db):
        self.db = db

    def search_themes(self, criteria, user_query="", limit=30, nicknames=None, exclude_ids=None):
        print(f"\n🔎 [RuleBased] 검색 시작 | 조건: {criteria}")
        
        played_theme_ids = set()
        
        # 1. 유저/그룹 플레이 이력 조회
        target_users = []
        if isinstance(nicknames, str):
            target_users = [n.strip() for n in nicknames.split(',') if n.strip()]
        elif isinstance(nicknames, list):
            target_users = nicknames

        if target_users:
            try:
                users_ref = self.db.collection('users')
                # Firestore IN query 제한 (최대 10)
                if len(target_users) > 10: target_users = target_users[:10]
                
                user_q = users_ref.where(filter=FieldFilter("nickname", "in", target_users))
                user_docs = list(user_q.stream())
                
                for u_doc in user_docs:
                    u_data = u_doc.to_dict()
                    played = u_data.get('played', [])
                    for pid in played:
                        played_theme_ids.add(int(pid))
                
                print(f"   -> 제외할 플레이 이력: {len(played_theme_ids)}개")
            except Exception as e:
                print(f"   ⚠️ [Error] 이력 조회 실패: {e}")

        # 제외할 ID 합치기
        total_exclude_ids = set(exclude_ids) if exclude_ids else set()
        total_exclude_ids.update(played_theme_ids)

        # 2. DB 쿼리 구성
        themes_ref = self.db.collection('themes')
        query = themes_ref

        # 지역 필터: 정확도 향상을 위해 DB 쿼리 단계에서 필터링 권장
        # 단, DB의 location이 "서울 강남구"이고 입력이 "강남"인 경우 등 부분 일치를 위해
        # 파이썬 레벨에서 필터링하는 전략을 유지하되, 메모리 효율을 위해 limit을 넉넉히 잡음
        
        # 여기서는 만족도 순 정렬을 기본으로
        query = query.order_by('satisfyTotalRating', direction=firestore.Query.DESCENDING).limit(200)

        # 3. 데이터 가져오기 및 메모리 필터링
        docs = list(query.stream())
        print(f"   📦 후보군 Fetch: {len(docs)}개")

        raw_candidates = []
        
        loc_input = criteria.get('location', '').replace(" ", "") if criteria.get('location') else ""

        for doc in docs:
            data = doc.to_dict()
            
            # (1) ID 제외
            try:
                ref_id = data.get('ref_id')
                tid = int(ref_id) if ref_id is not None else int(doc.id)
                # 안전하게 문자열/정수 모두 비교
                if tid in total_exclude_ids or str(tid) in total_exclude_ids:
                    continue
            except:
                if doc.id in total_exclude_ids:
                    continue

            # (2) 필터링 (지역) - 공백 제거 후 포함 여부 확인 (유연한 검색)
            if loc_input:
                db_loc = f"{data.get('location', '')} {data.get('store_name', '')}".replace(" ", "")
                if loc_input not in db_loc:
                    continue

            # 벡터 추출 (나중에 재정렬용)
            vec_obj = data.get('embedding_field')
            vector = None
            try:
                if vec_obj:
                    if hasattr(vec_obj, 'to_map'): 
                        vector = vec_obj.to_map()['value']
                    else:
                        vector = list(vec_obj)
            except: pass

            raw_candidates.append({
                'id': doc.id,
                'title': data.get('title'),
                'store': data.get('store_name'),
                'location': data.get('location'),
                'genre': data.get('genre'),
                'desc': data.get('description', '')[:150],
                'rating': float(data.get('satisfyTotalRating') or 0),
                'fear': float(data.get('fearTotalRating') or 0),
                'difficulty': float(data.get('difficultyTotalRating') or 0),
                'activity': float(data.get('activityTotalRating') or 0),
                'problem': float(data.get('problemTotalRating') or 0),
                'story': float(data.get('storyTotalRating') or 0),
                'interior': float(data.get('interiorTotalRating') or 0),
                'act': float(data.get('actTotalRating') or 0),
                'vector': vector
            })

        # 4. 동적 정렬 (키워드 반영)
        sorted_candidates = sort_candidates_by_query(raw_candidates, user_query)
        
        return sorted_candidates[:limit]


class VectorRecommender:
    def __init__(self, db, model):
        self.db = db
        self.model = model

    def get_group_vector(self, nicknames):
        target_users = []
        if isinstance(nicknames, str):
            target_users = [n.strip() for n in nicknames.split(',') if n.strip()]
        elif isinstance(nicknames, list):
            target_users = nicknames
            
        if not target_users: return None

        try:
            users_ref = self.db.collection('users')
            if len(target_users) > 10: target_users = target_users[:10]
            
            query = users_ref.where(filter=FieldFilter("nickname", "in", target_users))
            docs = list(query.stream())
            
            vectors = []
            for doc in docs:
                user_data = doc.to_dict()
                vec_obj = user_data.get('embedding_field')
                if vec_obj:
                    try:
                        if hasattr(vec_obj, 'to_map'):
                            v = vec_obj.to_map()['value']
                        else:
                            v = list(vec_obj)
                        vectors.append(v)
                    except: pass
            
            if not vectors: 
                return None
            
            # 평균 벡터 계산
            matrix = np.array(vectors)
            mean_vector = np.mean(matrix, axis=0)
            
            norm = np.linalg.norm(mean_vector)
            if norm > 0:
                mean_vector = mean_vector / norm
            
            return mean_vector.tolist()

        except Exception as e:
            print(f"   ⚠️ [Error] 그룹 벡터 계산 실패: {e}")
            return None

    def _execute_vector_search(self, vector, limit=20, filters=None, exclude_ids=None):
        themes_ref = self.db.collection('themes')
        query = themes_ref
        
        # 벡터 검색에서는 필터 제약이 있을 수 있으므로 주의 (Composite Index 필요)
        # 여기서는 필터 없이 벡터 검색 후 메모리 필터링 방식 사용 (데이터가 아주 많지 않다고 가정)
        # 데이터가 많으면 filters['location']을 DB 쿼리에 넣어야 함
        
        try:
            fetch_limit = limit + (len(exclude_ids) if exclude_ids else 0) + 10 # 여유분
            
            vector_query = query.find_nearest(
                vector_field="embedding_field",
                query_vector=Vector(vector),
                distance_measure=DistanceMeasure.COSINE,
                limit=fetch_limit
            )
            docs = vector_query.get()
            
            results = []
            loc_filter = filters.get('location', '').replace(" ", "") if filters else ""

            for doc in docs:
                # 1. ID 제외
                if exclude_ids and (doc.id in exclude_ids or str(doc.id) in exclude_ids):
                    continue
                
                data = doc.to_dict()
                
                # 2. 지역 필터 (메모리 단에서 유연하게 검사)
                if loc_filter:
                    db_loc = f"{data.get('location', '')} {data.get('store_name', '')}".replace(" ", "")
                    if loc_filter not in db_loc:
                        continue
                    
                results.append({
                    'id': doc.id,
                    'title': data.get('title'),
                    'store': data.get('store_name'),
                    'location': data.get('location'),
                    'desc': data.get('description', '')[:100],
                    'rating': float(data.get('satisfyTotalRating') or 0),
                    # 필요한 필드만 최소한으로
                    'fear': float(data.get('fearTotalRating') or 0),
                    'activity': float(data.get('activityTotalRating') or 0),
                    'difficulty': float(data.get('difficultyTotalRating') or 0),
                })
                
                if len(results) >= limit:
                    break
            
            return results
        except Exception as e:
            print(f"   ❌ [Error] Vector Search 실패: {e}")
            return []

    def recommend_by_text(self, query_text, filters=None, exclude_ids=None):
        if not self.model: return []
        print(f"🔤 [TextVector] 텍스트 임베딩 생성...")
        query_vector = self.model.encode(query_text).tolist()
        return self._execute_vector_search(query_vector, filters=filters, exclude_ids=exclude_ids)

    def recommend_by_user_search(self, user_context, limit=3, filters=None, exclude_ids=None):
        target_vec = self.get_group_vector(user_context)
        if not target_vec: return []
        print("🚀 [UserVector] 유저 벡터로 DB 검색")
        return self._execute_vector_search(target_vec, limit=limit, filters=filters, exclude_ids=exclude_ids)
