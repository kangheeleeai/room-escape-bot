import numpy as np
from database import firestore, Vector, DistanceMeasure, FieldFilter
from utils import sort_candidates_by_query
from config import PROJECT_ID

class RuleBasedRecommender:
    def __init__(self, db):
        self.db = db

    def search_themes(self, criteria, user_query="", limit=30, nicknames=None, exclude_ids=None):
        played_theme_ids = set()
        
        # 유저 플레이 이력 조회
        target_users = []
        if isinstance(nicknames, str):
            target_users = [n.strip() for n in nicknames.split(',')]
        elif isinstance(nicknames, list):
            target_users = nicknames

        if target_users:
            try:
                users_ref = self.db.collection('users')
                if len(target_users) > 10: target_users = target_users[:10]
                user_q = users_ref.where(filter=FieldFilter("nickname", "in", target_users))
                user_docs = list(user_q.stream())
                
                for u_doc in user_docs:
                    u_data = u_doc.to_dict()
                    played = u_data.get('played', [])
                    for pid in played:
                        played_theme_ids.add(int(pid))
            except Exception as e:
                print(f"Error fetching group history: {e}")

        total_exclude_ids = set(exclude_ids) if exclude_ids else set()
        total_exclude_ids.update(played_theme_ids)

        themes_ref = self.db.collection('themes')
        query = themes_ref

        if criteria.get('location'):
            query = query.where(filter=FieldFilter('location', '==', criteria['location']))
        else:
            query = query.order_by('satisfyTotalRating', direction=firestore.Query.DESCENDING).limit(100)

        docs = query.stream()
        raw_candidates = []

        for doc in docs:
            data = doc.to_dict()
            try:
                ref_id = data.get('ref_id')
                tid = int(ref_id) if ref_id is not None else int(doc.id)
                if tid in total_exclude_ids or str(tid) in total_exclude_ids or doc.id in total_exclude_ids:
                    continue
            except:
                if doc.id in total_exclude_ids:
                    continue

            is_match = True
            if criteria.get('location'):
                loc_input = criteria['location']
                db_loc = f"{data.get('location', '')} {data.get('store_name', '')}"
                if loc_input not in db_loc:
                    is_match = False

            if is_match:
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

        sorted_candidates = sort_candidates_by_query(raw_candidates, user_query)
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
            
            if not vectors: return None
            matrix = np.array(vectors)
            mean_vector = np.mean(matrix, axis=0)
            norm = np.linalg.norm(mean_vector)
            if norm > 0:
                mean_vector = mean_vector / norm
            return mean_vector.tolist()

        except Exception as e:
            print(f"Group vector error: {e}")
            return None

    def rerank_candidates(self, candidates, user_context):
        if isinstance(user_context, list) or (isinstance(user_context, str) and ',' in user_context):
            target_vec = self.get_group_vector(user_context)
        else:
            target_vec = self.get_group_vector([user_context])

        if not target_vec:
            return candidates

        try:
            u_v = np.array(target_vec)
            u_norm = np.linalg.norm(u_v)
            if u_norm == 0: return candidates

            for c in candidates:
                if c.get('vector'):
                    c_v = np.array(c['vector'])
                    c_norm = np.linalg.norm(c_v)
                    if c_norm > 0:
                        sim = np.dot(u_v, c_v) / (u_norm * c_norm)
                    else:
                        sim = 0
                else:
                    sim = -1
                c['score_vec'] = sim
            
            candidates.sort(key=lambda x: x.get('score_vec', -1), reverse=True)
            return candidates
            
        except Exception as e:
            print(f"Rerank Error: {e}")
            return candidates

    def _execute_vector_search(self, vector, limit=20, filters=None, exclude_ids=None):
        themes_ref = self.db.collection('themes')
        query = themes_ref
        
        if filters and filters.get('location'):
            query = query.where(filter=FieldFilter("location", "==", filters['location']))

        try:
            fetch_limit = limit + len(exclude_ids) if exclude_ids else limit
            
            vector_query = query.find_nearest(
                vector_field="embedding_field",
                query_vector=Vector(vector),
                distance_measure=DistanceMeasure.COSINE,
                limit=fetch_limit
            )
            results = []
            for doc in vector_query.get():
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
            return results
        except Exception as e:
            print(f"Vector Search Error: {e}")
            if "Missing vector index configuration" in str(e):
                 print(f"⚠️ 인덱스 생성 필요: gcloud firestore indexes composite create --project={PROJECT_ID} --collection-group=themes --query-scope=COLLECTION --field-config=order=ASCENDING,field-path=location --field-config=vector-config='{{\"dimension\":384,\"flat\": \"{{}}\"}}',field-path=embedding_field")
            return []

    def recommend_by_text(self, query_text, filters=None, exclude_ids=None):
        if not self.model: return []
        query_vector = self.model.encode(query_text).tolist()
        return self._execute_vector_search(query_vector, filters=filters, exclude_ids=exclude_ids)

    def recommend_by_user_search(self, user_context, limit=3, filters=None, exclude_ids=None):
        if isinstance(user_context, list) or (isinstance(user_context, str) and ',' in user_context):
            target_vec = self.get_group_vector(user_context)
        else:
            target_vec = self.get_group_vector([user_context])
            
        if not target_vec: return []
        return self._execute_vector_search(target_vec, limit=limit, filters=filters, exclude_ids=exclude_ids)