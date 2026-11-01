import argparse
import sentencepiece as spm


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model",
        type=str,
        default="/workspace/j-moshivis/jmoshivis/tokenizer_spm_32k_3.model",
        help="SentencePiece model path",
    )
    parser.add_argument(
        "--ids",
        type=int,
        nargs="+",
        default=[0, 3, 9, 8, 7, 2767, 10, 3418, 11, 26, 1, 3704, 25062, 20833, 58],
        help="Token IDs to inspect",
    )
    args = parser.parse_args()

    sp = spm.SentencePieceProcessor()
    sp.load(args.model)

    print(f"Loaded SentencePiece model from: {args.model}")
    print("")

    vocab_size = sp.get_piece_size()
    print(f"Vocab size: {vocab_size}")
    print("")

    for tid in args.ids:
        if tid < 0 or tid >= vocab_size:
            print(f"[ID {tid}] ❌ out of range (0 ~ {vocab_size-1})")
            continue

        piece = sp.id_to_piece(tid)
        score = sp.get_score(tid) if hasattr(sp, "get_score") else None

        print(f"[ID {tid}]")
        print(f"  piece: {repr(piece)}")
        if score is not None:
            print(f"  score: {score}")
        print("")


if __name__ == "__main__":
    main()
