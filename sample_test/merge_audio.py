import sys
import numpy as np
import soundfile as sf

def load_mono(path: str):
    data, sr = sf.read(path)  # data shape: (N,) or (N, C)
    # ステレオなどの場合はモノラルに変換
    if data.ndim == 2:
        data = data.mean(axis=1)
    return data, sr


def main(user_path: str, assistant_path: str):
    # 読み込み
    user, sr_user = load_mono(user_path)
    assistant, sr_assistant = load_mono(assistant_path)

    # サンプリングレートチェック
    if sr_user != sr_assistant:
        raise ValueError(f"Sample rate mismatch: {sr_user} vs {sr_assistant}. "
                         f"事前に同じサンプリングレートに揃えてください。")

    # ① 時系列結合（user → assistant）
    concat = np.concatenate([user, assistant], axis=0)
    sf.write("merged_concat.wav", concat, sr_user)
    print("保存しました: merged_concat.wav (mono, user→assistant)")

    # ② ステレオ結合（L=user, R=assistant）
    max_len = max(len(user), len(assistant))

    # 長さを揃える（足りない部分は無音）
    user_pad = np.zeros(max_len, dtype=np.float32)
    assistant_pad = np.zeros(max_len, dtype=np.float32)

    user_pad[:len(user)] = user.astype(np.float32)
    assistant_pad[:len(assistant)] = assistant.astype(np.float32)

    # ステレオ化: shape (N, 2) => [L, R] = [user, assistant]
    stereo = np.stack([user_pad, assistant_pad], axis=1)
    sf.write("merged_stereo.wav", stereo, sr_user)
    print("保存しました: merged_stereo.wav (stereo, L=user, R=assistant)")


if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("使い方: python merge_audio.py user.wav assistant.wav")
        sys.exit(1)

    main(sys.argv[1], sys.argv[2])
