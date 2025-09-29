import torch
from torch.utils.data import Dataset
from PIL import Image
from typing import List, Dict, Any

# --- プレースホルダー: 実際のプロジェクトのモジュールを想定 ---

# Moshiモデルのテキストトークナイザのダミー実装
class TextTokenizer:
    def __init__(self, vocab_size=32000):
        self.vocab_size = vocab_size
        self.pad_token_id = 3
        self.zero_token_id = -1
        
    def encode(self, text: str) -> List[int]:
        # テキストをトークンIDのリストに変換する処理
        return [hash(c) % self.vocab_size for c in text]

# 画像プロセッサのダミー実装
class ImageProcessor:
    def __call__(self, image: Image.Image) -> torch.Tensor:
        # 画像を前処理してテンソルに変換する処理
        return torch.randn(3, 224, 224)

# --- ここまでプレースホルダー ---

class MoshiVisDataset(Dataset):
    """
    Mimiで事前エンコードされた音声トークンを扱う、より実践的なMoshiVis用データセットクラス。

    Args:
        data (List[Dict[str, Any]]): データセットのメタ情報。
            - 'image_path': 画像ファイルパス
            - 'type': 'speech' または 'speechless'
            - 'text': (speechlessの場合) テキスト
            - 'assistant_text': (speechの場合) アシスタントのテキスト
            - 'user_audio_codes_path': (speechの場合) mimiでエンコード済みのユーザー音声トークンのファイルパス (.pt)
            - 'assistant_audio_codes_path': (speechの場合) mimiでエンコード済みのアシスタント音声トークンのファイルパス (.pt)
        text_tokenizer (TextTokenizer): テキスト用のトークナイザ。
        image_processor (ImageProcessor): 画像用のプロセッサ。
        n_audio_codebooks (int): 音声コーデックのコードブック数 (mimiの場合は8)。
    """
    def __init__(self, data: List[Dict[str, Any]], text_tokenizer: TextTokenizer, image_processor: ImageProcessor, n_audio_codebooks: int = 8):
        self.data = data
        self.text_tokenizer = text_tokenizer
        self.image_processor = image_processor
        self.n_audio_codebooks = n_audio_codebooks

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        sample = self.data[idx]
        
        # 1. 画像の処理
        image = Image.open(sample['image_path']).convert("RGB")
        processed_image = self.image_processor(image)

        # 2. サンプルの種類に応じてinput_idsを生成
        if sample['type'] == 'speech':
            # アシスタントのテキストをトークン化
            text_tokens = torch.tensor(self.text_tokenizer.encode(sample['assistant_text']), dtype=torch.long)
            
            # 事前にエンコードされた音声トークンをファイルから読み込む
            user_audio_codes = torch.load(sample['user_audio_codes_path'])         # 形状: (n_q, S_user)
            assistant_audio_codes = torch.load(sample['assistant_audio_codes_path']) # 形状: (n_q, S_assistant)

            # 論文のアーキテクチャに基づき、ストリームを合計する
            # ここでは単純な加算を仮定。実際のMoshiの実装ではより複雑な可能性がある
            audio_codes = user_audio_codes + assistant_audio_codes
            
            # 時系列をテキストと音声で揃える（短い方に合わせる）
            seq_len = min(text_tokens.size(0), audio_codes.size(1))
            text_tokens = text_tokens[:seq_len].unsqueeze(0)    # 形状: (1, S)
            audio_codes = audio_codes[:, :seq_len]              # 形状: (n_q, S)
            
            input_ids = torch.cat([text_tokens, audio_codes], dim=0) # 形状: (1 + n_q, S)

        elif sample['type'] == 'speechless':
            # テキスト全体をトークン化
            text_tokens = torch.tensor(self.text_tokenizer.encode(sample['text']), dtype=torch.long)
            seq_len = text_tokens.size(0)

            # テキストストリームと、ダミーの音声ストリームを作成
            text_stream = text_tokens.unsqueeze(0) # 形状: (1, S)
            audio_stream = torch.full(
                size=(self.n_audio_codebooks, seq_len),
                fill_value=self.text_tokenizer.zero_token_id,
                dtype=torch.long
            )
            input_ids = torch.cat([text_stream, audio_stream], dim=0) # 形状: (1 + n_q, S)
        else:
            raise ValueError(f"Unknown sample type: {sample['type']}")

        return {
            "input_ids": input_ids,
            "image": processed_image
        }


if __name__ == '__main__':
    # --- 使用例 ---
    
    # 0. (事前準備) mimiで.wavをエンコードして.ptファイルとして保存する（この処理はデータセット作成時に一度だけ行う）
    # `mimi.encode(waveform)` の結果を torch.save で保存するイメージ
    # dummy_user_audio_codes.pt, dummy_assistant_audio_codes.pt を作成したと仮定
    dummy_audio_codes = torch.randint(0, 1024, (8, 150))
    torch.save(dummy_audio_codes, 'dummy_user_audio_codes.pt')
    torch.save(dummy_audio_codes, 'dummy_assistant_audio_codes.pt')
    Image.new('RGB', (100, 100)).save('dummy_image1.jpg')
    
    # 1. ダミーのメタデータを作成
    dummy_metadata = [
        {'image_path': 'dummy_image1.jpg', 'type': 'speechless', 'text': 'A beautiful cat is sitting on the table.'},
        {'image_path': 'dummy_image1.jpg', 'type': 'speech', 'assistant_text': 'I see a cat.', 
         'user_audio_codes_path': 'dummy_user_audio_codes.pt', 
         'assistant_audio_codes_path': 'dummy_assistant_audio_codes.pt'}
    ]

    # 2. 初期化
    tokenizer = TextTokenizer()
    processor = ImageProcessor()
    
    # 3. データセットをインスタンス化
    dataset = MoshiVisDataset(
        data=dummy_metadata,
        text_tokenizer=tokenizer,
        image_processor=processor,
        n_audio_codebooks=8
    )
    
    # 4. データを取得して確認
    speechless_item = dataset[0]
    print("--- Speechless Sample ---")
    print("Input IDs Shape:", speechless_item['input_ids'].shape) # -> torch.Size([9, 40])  (1+8, text_len)
    print("Image Tensor Shape:", speechless_item['image'].shape)

    speech_item = dataset[1]
    print("\n--- Speech Sample ---")
    print("Input IDs Shape:", speech_item['input_ids'].shape) # -> torch.Size([9, 12])  (1+8, min(text_len, audio_len))
    print("Image Tensor Shape:", speech_item['image'].shape)