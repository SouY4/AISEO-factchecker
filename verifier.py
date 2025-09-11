import json
import numpy as np
from openai import OpenAI
from cohere import Client as CohereClient
from typing import List, Dict, Any
from knowledge_base import KnowledgeBase

class FactChecker:
    """
    記事中の単一の文を検証し、スコア、判定、理由を返すクラス。
    """
    def __init__(self, openai_client: OpenAI, cohere_client: CohereClient):
        self.openai_client = openai_client
        self.cohere_client = cohere_client

    def _get_embedding(self, text: str) -> List[float]:
        """ヘルパー関数：単一のテキストをベクトル化する"""
        res = self.openai_client.embeddings.create(input=[text], model="text-embedding-3-small")
        return res.data[0].embedding

    def verify_sentence(self, sentence: str, knowledge_base: KnowledgeBase) -> Dict[str, Any]:
        """
        単一の文を3段階のプロセスで検証する。

        Args:
            sentence: 検証したい単一の文。
            knowledge_base: 構築済みのKnowledgeBaseインスタンス。

        Returns:
            検証結果を含む辞書。
        """
        print(f"\n--- Verifying sentence: '{sentence}' ---")
        
        # === Stage 1: Retrieval (HyDE) ===
        print("Stage 1: Retrieving relevant documents with HyDE...")
        try:
            hyde_prompt = f"以下の文について、その主張を詳細に解説し、背景を説明するような理想的な解説文を生成してください。事実は不正確でも構いません。文脈を豊かにすることが目的です。\n\n文: \"{sentence}\""
            response = self.openai_client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[{"role": "user", "content": hyde_prompt}]
            )
            hypothetical_document = response.choices[0].message.content
            
            query_vector = self._get_embedding(hypothetical_document)
            retrieved_docs = knowledge_base.search(query_vector, top_k=10)
            if not retrieved_docs:
                return {"decision": "Not Enough Information", "reason": "No relevant documents found.", "score": 0}
        except Exception as e:
            return {"decision": "Error", "reason": f"Error in Retrieval stage: {e}", "score": 0}

        # === Stage 2: Reranking ===
        print("Stage 2: Reranking documents with Cohere...")
        try:
            doc_texts = [doc['text'] for doc in retrieved_docs]
            rerank_response = self.cohere_client.rerank(
                query=sentence,
                documents=doc_texts,
                top_n=3,
                model="rerank-multilingual-v3.0"
            )
            
            evidence_docs = []
            for hit in rerank_response.results:
                original_doc = retrieved_docs[hit.index]
                evidence_docs.append({
                    "text": original_doc['text'],
                    "source": original_doc['source']
                })
        except Exception as e:
            return {"decision": "Error", "reason": f"Error in Reranking stage: {e}", "score": 0}

        # === Stage 3: Verification ===
        print("Stage 3: Final verification with LLM...")
        try:
            # a. Quantitative Scoring
            sentence_vector = np.array(self._get_embedding(sentence))
            top_evidence_vector = np.array(self._get_embedding(evidence_docs[0]['text']))
            
            cosine_similarity = np.dot(sentence_vector, top_evidence_vector) / (np.linalg.norm(sentence_vector) * np.linalg.norm(top_evidence_vector))
            quantitative_score = int((cosine_similarity + 1) / 2 * 100)

            # b. LLM Verification
            evidence_text_for_prompt = "\n".join([f"{i+1}. \"{doc['text']}\" (Source: {doc['source']})" for i, doc in enumerate(evidence_docs)])
            
            verification_prompt = f"""
            役割: あなたは客観的で厳密なファクトチェッカーです。
            指示: 以下の「検証対象の文」が、「証拠文書」の内容によってどの程度裏付けられるかを評価してください。評価として判定（支持/矛盾/判断不能）、そしてその理由をJSON形式で出力してください。

            コンテキスト:
            - 定量的類似度スコア: {quantitative_score} / 100
              - このスコアは、文と最も関連性の高い証拠が意味的にどれだけ近いかを示す客観的な指標です。スコアが高いほど内容は意味的に似ていますが、矛盾する場合もあるため、必ず文書の内容を吟味してください。

            検証対象の文:
            "{sentence}"

            証拠文書 (関連度順):
            {evidence_text_for_prompt}

            出力JSONフォーマット:
            {{
              "decision": "<Supported/Refuted/Not Enough Information>",
              "reason": "<定量的スコアと証拠文書の内容を基に、なぜその判定に至ったのかを具体的に説明してください。>"
            }}
            """
            
            final_response = self.openai_client.chat.completions.create(
                model="gpt-4o",
                messages=[{"role": "user", "content": verification_prompt}],
                response_format={"type": "json_object"}
            )
            
            llm_result = json.loads(final_response.choices[0].message.content)
            
            # 最終結果を組み立てる
            final_result = {
                "original_sentence": sentence,
                "score": quantitative_score,
                "decision": llm_result.get("decision", "Error"),
                "reason": llm_result.get("reason", "No reason provided."),
                "evidence": evidence_docs
            }
            return final_result

        except Exception as e:
            return {"decision": "Error", "reason": f"Error in Verification stage: {e}", "score": 0}
