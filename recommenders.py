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
            target_users = [n.strip() for n in nicknames.split(',')]
        elif isinstance(nicknames, list):
            target_users = nicknames

        if target_users:
            print(f"   👤 플레이 이력 조회 대상: {target_users}")
            try:
                users_ref = self.db.collection('users')
                # Firestore IN query 제한 고려 (최대 10명)
                if len(target_users) > 10: target_users = target_users[:10]
                
                user_q = users_ref.where(filter=FieldFilter("nickname", "in", target_users))
                user_docs = list(user_q.stream())
                
                found_count = 0
                for u_doc in user_docs:
                    u_data = u_doc.to_dict()
                    played = u_data.get('played', [])
                    for pid in played:
                        played_theme_ids.add(int(pid))
                    found_count += 1
                
                print(f"   -> DB에서 유저 {found_count}명 발견, 총 {len(played_theme_ids)}개 테마 제외 예정")
            except Exception as e:
                print(f"   ⚠️ [Error] 그룹 기록 조회 실패: {e}")

        # 제외할 ID 합치기 ("다른거 추천" + "플레이한거")
        total_exclude_ids = set(exclude_ids) if exclude_ids else set()
        total_exclude_ids.update(played_theme_ids)

        # 2. DB 쿼리 구성
        themes_ref = self.db.collection('themes')
        query = themes_ref

        if criteria.get('location'):
            # 지역 필터가 있으면 적용
            query = query.where(filter=FieldFilter('location', '==', criteria['location']))
            print(f"   📌 DB 쿼리 필터: 지역 == {criteria['location']}")
        else:
            # 없으면 만족도 순 정렬
            query = query.order_by('satisfyTotalRating', direction=firestore.Query.DESCENDING).limit(100)
            print("   📌 DB 쿼리 필터: 없음 (만족도순 전체 검색)")

        # 3. 데이터 가져오기 및 메모리 필터링
        docs = list(query.stream())
        print(f"   📦 DB 가져온 문서 수: {len(docs)}개")

        raw_candidates = []
        count_excluded_by_id = 0
        count_excluded_by_filter = 0

        for doc in docs:
            data = doc.to_dict()
            
            # (1) ID 제외 체크
            try:
                ref_id = data.get('ref_id')
                tid = int(ref_id) if ref_id is not None else int(doc.id)
                # int/str 형변환 유연하게 비교
                if tid in total_exclude_ids or str(tid) in total_exclude_ids or doc.id in total_exclude_ids:
                    count_excluded_by_id += 1
                    continue
            except:
                if doc.id in total_exclude_ids:
                    count_excluded_by_id += 1
                    continue

            # (2) 메모리 내 추가 필터링 (지역 등)
            is_match = True
            if criteria.get('location'):
                loc_input = criteria['location']
                db_loc = f"{data.get('location', '')} {data.get('store_name', '')}"
                if loc_input not in db_loc:
                    is_match = False
                    count_excluded_by_filter += 1

            if is_match:
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

        print(f"   ✂️ 필터링: 이력 제외 {count_excluded_by_id}개, 조건 불일치 {count_excluded_by_filter}개")
        
        # 4. 동적 정렬 (키워드 반영)
        sorted_candidates = sort_candidates_by_query(raw_candidates, user_query)
        
        if sorted_candidates:
            print(f"   ✅ [RuleBased] 최종 후보 {len(sorted_candidates)}개 확보. (1위: {sorted_candidates[0]['title']})")
        else:
            print("   ⚠️ [RuleBased] 최종 후보가 없습니다.")

        return sorted_candidates[:limit]


class VectorRecommender:
    def __init__(self, db, model):
        self.db = db
        self.model = model

    def get_group_vector(self, nicknames):
        target_users = []
        if isinstance(nicknames, str):
            target_users = [n.strip() for n in nicknames.split(',')]
        elif isinstance(nicknames, list):
            target_users = nicknames
            
        if not target_users: return None

        print(f"👥 [Vector] 그룹 벡터 계산 시도: {target_users}")

        try:
            users_ref = self.db.collection('users')
            if len(target_users) > 10: target_users = target_users[:10]
            
            # IN 쿼리
            query = users_ref.where(filter=FieldFilter("nickname", "in", target_users))
            docs = list(query.stream())
            
            vectors = []
            found_nicknames = []
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
                        found_nicknames.append(user_data.get('nickname'))
                    except: pass
            
            print(f"   -> DB에서 벡터 확보 성공: {found_nicknames} ({len(vectors)}/{len(target_users)}명)")
            
            if not vectors: 
                print("   ⚠️ 유효한 유저 벡터가 하나도 없습니다.")
                return None
            
            # 평균 벡터 계산 (Centroid)
            matrix = np.array(vectors)
            mean_vector = np.mean(matrix, axis=0)
            
            # 정규화
            norm = np.linalg.norm(mean_vector)
            if norm > 0:
                mean_vector = mean_vector / norm
            
            return mean_vector.tolist()

        except Exception as e:
            print(f"   ⚠️ [Error] 그룹 벡터 계산 실패: {e}")
            return None

    def rerank_candidates(self, candidates, user_context):
        print(f"\n🔄 [Vector] 재정렬(Re-ranking) 시작 | 후보 {len(candidates)}개 | 대상: {user_context}")
        
        # 벡터 생성
        if isinstance(user_context, list) or (isinstance(user_context, str) and ',' in user_context):
            target_vec = self.get_group_vector(user_context)
        else:
            target_vec = self.get_group_vector([user_context])

        if not target_vec:
            print("   -> 타겟 벡터 없음. 재정렬 건너뜀.")
            return candidates

        try:
            u_v = np.array(target_vec)
            u_norm = np.linalg.norm(u_v)
            if u_norm == 0: return candidates

            # 코사인 유사도 계산
            for c in candidates:
                if c.get('vector'):
                    c_v = np.array(c['vector'])
                    c_norm = np.linalg.norm(c_v)
                    if c_norm > 0:
                        sim = np.dot(u_v, c_v) / (u_norm * c_norm)
                    else:
                        sim = 0
                else:
                    sim = -1 # 벡터 없는 테마는 최하위로
                c['score_vec'] = sim
            
            # 유사도 순 정렬
            candidates.sort(key=lambda x: x.get('score_vec', -1), reverse=True)
            
            # 로그 출력
            if candidates:
                print(f"   📊 재정렬 Top 3:")
                for i, c in enumerate(candidates[:3]):
                    print(f"      {i+1}. {c['title']} (유사도: {c.get('score_vec', 0):.4f})")
            
            return candidates
            
        except Exception as e:
            print(f"   ⚠️ [Error] 재정렬 중 오류: {e}")
            return candidates

    def _execute_vector_search(self, vector, limit=20, filters=None, exclude_ids=None):
        themes_ref = self.db.collection('themes')
        query = themes_ref
        
        if filters and filters.get('location'):
            query = query.where(filter=FieldFilter("location", "==", filters['location']))

        try:
            # 제외할 개수만큼 더 가져옴
            fetch_limit = limit + len(exclude_ids) if exclude_ids else limit
            
            print(f"\n🚀 [Vector] DB 벡터 검색 실행 (Limit: {fetch_limit})")
            
            vector_query = query.find_nearest(
                vector_field="embedding_field",
                query_vector=Vector(vector),
                distance_measure=DistanceMeasure.COSINE,
                limit=fetch_limit
            )
            results = []
            docs = vector_query.get()
            
            print(f"   -> DB 반환 문서 수: {len(docs)}개")
            
            for doc in docs:
                if exclude_ids and doc.id in exclude_ids:
                    continue
                    
                data = doc.to_dict()
                results.append({
                    'id': doc.id,
                    'title': data.get('title'),
                    'store': data.get('store_name'),
                    'location': data.get('location'),
                    'genre': data.get('genre'),
                    'desc': data.get('search_text', '')[:150],
                    'rating': float(data.get('satisfyTotalRating') or 0),
                    'fear': float(data.get('fearTotalRating') or 0),
                    'activity': float(data.get('activityTotalRating') or 0),
                    'difficulty': float(data.get('difficultyTotalRating') or 0),
                    'interior': float(data.get('interiorTotalRating') or 0),
                    'problem': float(data.get('problemTotalRating') or 0),
                    'story': float(data.get('storyTotalRating') or 0),
                    'act': float(data.get('actTotalRating') or 0)
                })
                
                if len(results) >= limit:
                    break
            
            print(f"   ✅ 최종 유효 결과: {len(results)}개")
            return results
        except Exception as e:
            print(f"   ❌ [Error] Vector Search 실패: {e}")
            if "Missing vector index configuration" in str(e):
                 print(f"   ⚠️ 인덱스 생성 필요: gcloud firestore indexes composite create --project={PROJECT_ID} --collection-group=themes --query-scope=COLLECTION --field-config=order=ASCENDING,field-path=location --field-config=vector-config='{{\"dimension\":384,\"flat\": \"{{}}\"}}',field-path=embedding_field")
            return []

    def recommend_by_text(self, query_text, filters=None, exclude_ids=None):
        if not self.model: return []
        print(f"🔤 [TextVector] 텍스트 임베딩: '{query_text}'")
        query_vector = self.model.encode(query_text).tolist()
        return self._execute_vector_search(query_vector, filters=filters, exclude_ids=exclude_ids)

    def recommend_by_user_search(self, user_context, limit=3, filters=None, exclude_ids=None):
        # 유저 벡터 가져오기
        if isinstance(user_context, list) or (isinstance(user_context, str) and ',' in user_context):
            target_vec = self.get_group_vector(user_context)
        else:
            target_vec = self.get_group_vector([user_context])
            
        if not target_vec: return []
        
        print("🚀 [UserVector] 유저 벡터로 DB 검색 실행")
        return self._execute_vector_search(target_vec, limit=limit, filters=filters, exclude_ids=exclude_ids)