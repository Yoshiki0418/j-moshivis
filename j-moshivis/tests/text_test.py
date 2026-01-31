import sentencepiece as spm
import os


def check_tokens():
    # 1. 確認したいトークンIDのリスト
    token_ids =  [ 9,    9,    8,     3,     3,     3,     3,     0,     9,  1400,     3,     0,
     9,    11,     9, 25879,     3,     3,     3,     0,     9,  1560,     3,     3,
     0,     9,     7,     0,     9,   7668
      ]
    # 2. トークナイザのパス（ご提示いただいたパス）
    tokenizer_path = "/workspace/j-moshivis/jmoshivis/tokenizer_spm_32k_3.model"

    # ファイルの存在確認
    if not os.path.exists(tokenizer_path):
        print(f"❌ エラー: トークナイザファイルが見つかりません: {tokenizer_path}")
        return

    # 3. モデルのロード
    sp = spm.SentencePieceProcessor()
    sp.load(tokenizer_path)

    # 4. 全体をデコードしてテキスト表示
    decoded_text = sp.decode(token_ids)

    print("="*40)
    print("=== Input IDs ===")
    print(token_ids)
    print("\n=== Decoded Text (Result) ===")
    print(f"👉 {decoded_text}")
    print("="*40)

    # 5. (参考) どのIDがどの文字に対応しているか内訳を表示
    print("\n=== Token-by-Token Breakdown ===")
    print(f"{'ID':<8} | {'Piece (Raw String)':<20}")
    print("-" * 35)
    for tid in token_ids:
        # id_to_piece で生のトークン表現（アンダースコア _ など含む）を確認
        piece = sp.id_to_piece(tid)
        print(f"{tid:<8} | {piece:<20}")


if __name__ == "__main__":
    check_tokens()