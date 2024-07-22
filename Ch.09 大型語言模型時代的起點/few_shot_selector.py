from sentence_transformers import SentenceTransformer, util
import torch

model = SentenceTransformer('paraphrase-MiniLM-L6-v2')

def select_few_shot(few_shot, user_query, top_k=5):
    """
    選擇最相關的Few-shot樣本來幫助回答使用者的問題。
    """
    query_embedding = model.encode(user_query)
    corpus_embeddings = model.encode(few_shot)

    cos_scores = util.pytorch_cos_sim(query_embedding, corpus_embeddings)[0]
    top_results = torch.topk(cos_scores, k=top_k)

    selected_few_shot = [few_shot[idx] for idx in top_results.indices]
    return selected_few_shot
