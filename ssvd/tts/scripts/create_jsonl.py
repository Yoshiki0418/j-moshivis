import sphn
import json
from pathlib import Path
import argparse


def create_jsonl(wavdir_path: str, output_dir: str) -> None:
    """Create a JSONL file with audio paths and durations."""
    paths = [str(f) for f in Path(wavdir_path).glob("**/*.wav")]
    durations = sphn.durations(paths)

    if not Path(output_dir).exists():
        Path(output_dir).mkdir(parents=True, exist_ok=True)
    with open(Path(output_dir) / "data.jsonl", "w") as fobj:
        for p, d in zip(paths, durations):
            if d is None:
                continue
            json.dump({"path": p, "duration": d}, fobj)
            fobj.write("\n")
            print(f"✅ {p} (duration: {d:.2f} sec)")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Create JSONL file with audio paths and durations.")
    parser.add_argument("--wav-dir", type=str, default="/workspace/data/speech/data_stereo", help="Directory containing WAV files.")
    parser.add_argument("--out-dir", type=str, default="/workspace/data/speech", help="Output JSONL file.")
    args = parser.parse_args()
    create_jsonl(args.wav_dir, args.out_dir)    
