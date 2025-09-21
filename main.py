import os
from dotenv import load_dotenv
from openai import OpenAI
from cohere import Client as CohereClient

# 各モジュールから必要な関数/クラスをインポート
from scraper import extract_text_from_urls
from text_processor import chunk_text, split_into_sentences
from knowledge_base import KnowledgeBase
from verifier import FactChecker

def main():
    """
    ファクトチェックの全プロセスを実行するメイン関数。
    """
    # .envファイルから環境変数を読み込む
    load_dotenv()

    # --- 1. 初期設定 ---
    print("Initializing API clients...")
    try:
        openai_client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
        cohere_client = CohereClient(api_key=os.getenv("COHERE_API_KEY"))
    except Exception as e:
        print(f"Error initializing API clients. Make sure your .env file is set up correctly. Error: {e}")
        return

    # --- 2. 入力データの定義 ---
    # ここに検証したい記事と参考文献URLを入力してください
    article_to_verify = """
    大規模言語モデル（LLM）は、自然言語処理の分野で革命をもたらしました。
    これらのモデルは、数十億のパラメータを持ち、インターネット上の膨大なテキストデータから学習します。
    その結果、人間のような文章を生成したり、質問に答えたり、テキストを要約したりする能力を獲得しました。
    しかし、LLMは時として事実に基づかない情報を生成する「ハルシネーション」という課題も抱えています。
    この問題に対処するため、RAG（Retrieval-Augmented Generation）という技術が注目されています。
    """
    
    reference_urls = [
        "https://ja.wikipedia.org/wiki/%E5%A4%A7%E8%A6%8F%E6%A8%A1%E8%A8%80%E8%AA%9E%E3%83%A2%E3%83%87%E3%83%AB",
        "https://ja.wikipedia.org/wiki/%E5%BC%B7%E5%8C%96%E5%AD%A6%E7%BF%92" # 少し関連性の低い記事を混ぜる
    ]

    # --- 3. 動的知識ベースの構築 ---
    print("\n--- Building Knowledge Base from References ---")
    documents = extract_text_from_urls(reference_urls)
    
    all_chunks = []
    for doc in documents:
        chunks = chunk_text(doc, chunk_size=500, chunk_overlap=50)
        all_chunks.extend(chunks)
        
    knowledge_base = KnowledgeBase(openai_client)
    
    # 知識ベースの構築を試行し、失敗した場合は適切にエラーハンドリングする
    try:
        knowledge_base.build(all_chunks)
        if knowledge_base.table is None:
            print("⚠️  知識ベースの構築に失敗しました。")
            print("これは以下の理由が考えられます：")
            print("1. OpenAI APIのクォータ制限に達している")
            print("2. APIキーが無効または設定されていない")
            print("3. ネットワーク接続の問題")
            print("\n解決策：")
            print("- OpenAI APIの課金プランを確認してください")
            print("- .envファイルのOPENAI_API_KEYが正しく設定されているか確認してください")
            print("- しばらく時間を置いてから再試行してください")
            return
    except Exception as e:
        print(f"⚠️  知識ベース構築中にエラーが発生しました: {e}")
        if "429" in str(e) or "quota" in str(e).lower():
            print("\n💡 これはOpenAI APIのクォータ制限エラーです。")
            print("解決策：")
            print("1. OpenAIアカウントの課金情報を確認してください")
            print("2. 使用量制限を確認し、必要に応じてプランをアップグレードしてください")
            print("3. 月の使用量がリセットされるまで待つか、クレジットを追加してください")
        return

    # --- 4. 記事の各文を検証 ---
    print("\n--- Verifying Each Sentence of the Article ---")
    sentences = split_into_sentences(article_to_verify)
    fact_checker = FactChecker(openai_client, cohere_client)
    
    final_results = []
    for sentence in sentences:
        if not sentence.strip():  # 空の文はスキップ
            continue
        result = fact_checker.verify_sentence(sentence, knowledge_base)
        final_results.append(result)

    # --- 5. 最終結果の表示 ---
    print("\n\n==========================================")
    print("          Fact-Checking Report")
    print("==========================================")
    for result in final_results:
        print(f"\n[Sentence] {result.get('original_sentence', 'N/A')}")
        print(f"  - Score: {result.get('score', 'N/A')}/100")
        print(f"  - Decision: {result.get('decision', 'N/A')}")
        print(f"  - Reason: {result.get('reason', 'N/A')}")
        
        # 類似度スコアの詳細表示
        if 'score' in result and result['score'] is not None:
            similarity_score = result['score'] / 100.0  # 0-1の範囲に変換
            print(f"  - Similarity Score: {similarity_score:.3f} (Cosine similarity between sentence and top evidence)")
        
        # 参照元文章の詳細表示
        if 'evidence' in result and result['evidence']:
            print(f"  - Top Evidence Source: {result['evidence'][0]['source']}")
            print(f"  - Top Evidence Text: \"{result['evidence'][0]['text'][:200]}...\"")  # 最初の200文字を表示
            
            # 追加の証拠があれば表示
            if len(result['evidence']) > 1:
                print(f"  - Additional Evidence Sources ({len(result['evidence'])-1} more):")
                for i, evidence in enumerate(result['evidence'][1:], 2):
                    print(f"    {i}. {evidence['source']}")
                    print(f"       \"{evidence['text'][:150]}...\"")  # 最初の150文字を表示


if __name__ == "__main__":
    main()
