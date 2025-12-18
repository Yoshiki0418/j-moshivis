# test_tokenize_text.py

import sentencepiece as spm


TEXT = (
    "これはどんなぬいぐるみですか？ これはトミー・タイガースのカードナルという鳥のぬいぐるみですね。"
    "赤い体と黒いマスク、そして赤い首輪が特徴的です。 "
    "どうやってこのぬいぐるみを作っているのでしょうか？ "
    "一般的に、このようなぬいぐるみは布や毛糸を使って作られています。"
    "その上に刺繍や染色を施して形を作り、最後に目や口などを付けて仕上げます。 "
    "これは何歳の鳥なんでしょうか？ "
    "これは実際の鳥ではなく、ぬいぐるみなので具体的な年齢はありません。"
    "しかし、デザインはリアルな鳥を模倣しているので、その種類の鳥の特徴を反映しています。 "
    "これはどんな場所で売られているのでしょうか？ "
    "タイガースの公式サイトや直営店、また玩具販売店などで販売されていることが多いです。"
    "オンラインショッピングサイトでも購入できるかもしれません。 "
    "これはどんな用途で使われているのでしょうか？ "
    "これは主に子供向けの玩具として使われますが、大人も手放せない可愛らしいデザインなので、"
    "大人も楽しむことができます。 "
    "ありがとう、とても参考になりました。 "
    "様々な情報を提供できればうれしいです。他にも教えてほしいことがありましたら、お知らせください。"
    "何歳"
)


def main():
    model_path = "/workspace/j-moshivis/jmoshivis/tokenizer_spm_32k_3.model"

    sp = spm.SentencePieceProcessor()
    sp.load(model_path)

    print(f"Loaded SentencePiece model from: {model_path}\n")

    print("=== Original Text ===")
    print(TEXT)
    print()

    # 1. ID 列 & piece 列を取得
    ids = sp.EncodeAsIds(TEXT)
    pieces = sp.EncodeAsPieces(TEXT)

    print("=== Token IDs ===")
    print(ids)
    print(f"Total tokens: {len(ids)}")
    print()

    print("=== ID ↔ Piece 対応 ===")
    for i, (tid, piece) in enumerate(zip(ids, pieces)):
        print(f"{i:3d}: ID={tid:5d}, piece={repr(piece)}")
    print()

    # 2. '▁' をスペースに変換して「人間が読める復元テキスト」にする
    reconstructed = "".join(pieces).replace("▁", " ")
    print("=== Reconstructed text (▁ → space) ===")
    print(reconstructed)
    print()

    # 3. ID 区切りで piece を並べた文字列（デバッグ用）
    #    例: [2767: 'です'] [3418: 'ます'] ...
    print("=== ID付きトークン列（プレゼン用） ===")
    id_piece_str = " ".join([f"[{tid}:{piece}]" for tid, piece in zip(ids, pieces)])
    print(id_piece_str)


if __name__ == "__main__":
    main()
