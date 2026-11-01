import sentencepiece as spm

sp = spm.SentencePieceProcessor()
sp.load("/workspace/j-moshivis/jmoshivis/tokenizer_spm_32k_3.model")

print("ID 9  :", sp.id_to_piece(635))
print("ID 3418 :", sp.id_to_piece(9621))