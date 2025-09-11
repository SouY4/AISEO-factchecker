import lancedb
import pandas as pd
from openai import OpenAI
from typing import List, Dict, Any

class KnowledgeBase:
    """
    テキストチャンクから一時的なベクトルデータベースを構築し、検索機能を提供するクラス。
    """
    def __init__(self, openai_client: OpenAI):
        """
        コンストラクタ。APIクライアントとインメモリDBを初期化する。
        """
        self.client = openai_client
        self.db = lancedb.connect("/tmp/lancedb") # 一時的なインメモリDBとして使用
        self.table = None

    def build(self, chunks: List[Dict[str, str]]):
        """
        チャンクのリストからベクトルデータベースを構築する。

        Args:
            chunks: 'source'と'text'のキーを持つチャンクの辞書のリスト。
        """
        if not chunks:
            print("No chunks to build knowledge base from.")
            return

        # チャンクのテキスト部分のみをリストとして抽出
        texts_to_embed = [chunk['text'] for chunk in chunks]
        
        print(f"Embedding {len(texts_to_embed)} chunks...")
        try:
            # OpenAIのEmbedding APIでベクトルを一括取得
            res = self.client.embeddings.create(input=texts_to_embed, model="text-embedding-3-small")
            embeddings = [embedding.embedding for embedding in res.data]
            
            # データフレームを作成
            df_data = []
            for i, chunk in enumerate(chunks):
                df_data.append({
                    "vector": embeddings[i],
                    "text": chunk['text'],
                    "source": chunk['source']
                })
            df = pd.DataFrame(df_data)

            # LanceDBテーブルを作成（既に存在する場合は上書き）
            table_name = "fact_checker_kb"
            if table_name in self.db.table_names():
                self.db.drop_table(table_name)
            
            self.table = self.db.create_table(table_name, data=df)
            print("Knowledge base built successfully.")

        except Exception as e:
            print(f"Failed to build knowledge base: {e}")
            # テーブルをNoneに設定して、後続処理で適切にハンドリングできるようにする
            self.table = None
            # より詳細なエラー情報を提供
            if "429" in str(e):
                print("💡 ヒント: これはレート制限エラーです。OpenAI APIの使用量制限に達している可能性があります。")
            elif "401" in str(e):
                print("💡 ヒント: これは認証エラーです。OPENAI_API_KEYが正しく設定されているか確認してください。")
            elif "insufficient_quota" in str(e):
                print("💡 ヒント: OpenAI APIのクォータが不足しています。課金プランを確認してください。")

    def search(self, query_vector: List[float], top_k: int = 10) -> List[Dict[str, Any]]:
        """
        クエリベクトルに類似したチャンクを検索する。

        Args:
            query_vector: 検索クエリのベクトル。
            top_k: 取得したい上位件数。

        Returns:
            検索結果（テキスト、ソースURL、類似度スコアなど）を含む辞書のリスト。
        """
        if self.table is None:
            print("Knowledge base is not built yet.")
            return []

        try:
            search_results = self.table.search(query_vector).limit(top_k).to_list()
            return search_results
        except Exception as e:
            print(f"Error during search: {e}")
            return []
