import json
from pathlib import Path

from datasets import load_dataset
from tokenizers import Tokenizer, models, normalizers, pre_tokenizers, decoders, processors
from tokenizers.trainers import BpeTrainer
from transformers import PreTrainedTokenizerFast

# Konfigurasi
DATASET_NAME = "AksaraLLM/aksara-pretrain-clean-v1.1"
OUTPUT_DIR = Path("./aurallm/aura_tokenizer_v3")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

VERSION = "3.0.0"
VOCAB_SIZE = 32773
MIN_FREQUENCY = 2
END_OF_WORD_SUFFIX = "</w>"

SPECIAL_TOKENS = [
    "[PAD]",
    "[EOS]",
    "[BOS]",
    "[UNK]",
    "[SEP]",
    "[MASK]",
    "[INST]",
    "[/INST]",
    "[SYS]",
    "[USER]",
    "[ASST]",
    "[TURN]",
    "[LANG_ID]",
    "[LANG_JV]",
    "[LANG_SU]",
    "[LANG_EN]",
]

BRAND_TOKENS = [
    "AuraLLM",
    "aurallm",
    "AURALLM",
    "Indonesia",
    "indonesia",
    "INDONESIA",
    "Pancasila",
    "pancasila",
    "Nusantara",
    "nusantara",
]

print("📥 Memuat dataset...")
dataset = load_dataset(DATASET_NAME, split="train")
print(f"✅ Loaded train split: {len(dataset):,} rows")

def batch_iterator(batch_size=1000):
    for i in range(0, len(dataset), batch_size):
        batch = dataset[i:i + batch_size]["text"]
        batch = [x for x in batch if isinstance(x, str) and x.strip()]
        if batch:
            yield batch

print("🧠 Membangun tokenizer...")

tokenizer = Tokenizer(
    models.BPE(
        unk_token="[UNK]",
        end_of_word_suffix=END_OF_WORD_SUFFIX
    )
)

# Normalizer: NFKC, no lowercase
tokenizer.normalizer = normalizers.NFKC()

# Pre-tokenizer: split by whitespace
tokenizer.pre_tokenizer = pre_tokenizers.Whitespace()

# Decoder: restore end-of-word suffix back into spaces
tokenizer.decoder = decoders.BPEDecoder(suffix=END_OF_WORD_SUFFIX)

trainer = BpeTrainer(
    vocab_size=VOCAB_SIZE,
    min_frequency=MIN_FREQUENCY,
    special_tokens=SPECIAL_TOKENS,
    end_of_word_suffix=END_OF_WORD_SUFFIX,
    show_progress=True,
)

tokenizer.train_from_iterator(batch_iterator(), trainer=trainer)

added_count = tokenizer.add_tokens(BRAND_TOKENS)
print(f"✅ Menambahkan {added_count} brand token sebagai token biasa")

special_token_ids = {tok: tokenizer.token_to_id(tok) for tok in SPECIAL_TOKENS}

for tok, tok_id in special_token_ids.items():
    if tok_id is None:
        raise ValueError(f"Special token hilang dari vocab: {tok}")

print("✅ Special token IDs:")
for tok, tok_id in special_token_ids.items():
    print(f"   {tok}: {tok_id}")

tokenizer.post_processor = processors.TemplateProcessing(
    single="[BOS] $A [EOS]",
    pair="[BOS] $A [SEP] $B [EOS]",
    special_tokens=[
        ("[BOS]", special_token_ids["[BOS]"]),
        ("[EOS]", special_token_ids["[EOS]"]),
        ("[SEP]", special_token_ids["[SEP]"]),
    ],
)

# Simpan raw tokenizer
tokenizer_json_path = OUTPUT_DIR / "tokenizer.json"
tokenizer.save(str(tokenizer_json_path))
print(f"✅ Saved tokenizer.json -> {tokenizer_json_path}")

fast_tokenizer = PreTrainedTokenizerFast(
    tokenizer_file=str(tokenizer_json_path),
    unk_token="[UNK]",
    pad_token="[PAD]",
    bos_token="[BOS]",
    eos_token="[EOS]",
    sep_token="[SEP]",
    mask_token="[MASK]",
    additional_special_tokens=[
        "[INST]", "[/INST]", "[SYS]", "[USER]", "[ASST]",
        "[TURN]", "[LANG_ID]", "[LANG_JV]", "[LANG_SU]", "[LANG_EN]"
    ],
)

# tambahkan brand tokens sebagai token biasa
fast_tokenizer.add_tokens(BRAND_TOKENS)

# simpan tokenizer HF
fast_tokenizer.save_pretrained(str(OUTPUT_DIR))
print(f"✅ Saved HF tokenizer files -> {OUTPUT_DIR}")

tokenizer_config = {
    "version": VERSION,
    "vocab_size": fast_tokenizer.vocab_size + len(BRAND_TOKENS),
    "pre_tokenizer": "Whitespace",
    "normalizer": "NFKC (NO lowercase)",
    "model": "BPE",
    "end_of_word_suffix": END_OF_WORD_SUFFIX,
    "brand_tokens": BRAND_TOKENS,
    "special_tokens": {
        tok: special_token_ids[tok] for tok in SPECIAL_TOKENS
    },
    "tokenizer_class": "PreTrainedTokenizerFast",
    "unk_token": "[UNK]",
    "pad_token": "[PAD]",
    "bos_token": "[BOS]",
    "eos_token": "[EOS]",
    "sep_token": "[SEP]",
    "mask_token": "[MASK]",
    "additional_special_tokens": [
        "[INST]", "[/INST]", "[SYS]", "[USER]", "[ASST]",
        "[TURN]", "[LANG_ID]", "[LANG_JV]", "[LANG_SU]", "[LANG_EN]"
    ],
    "clean_up_tokenization_spaces": False,
    "model_max_length": 2048
}

with open(OUTPUT_DIR / "tokenizer_config.json", "w", encoding="utf-8") as f:
    json.dump(tokenizer_config, f, ensure_ascii=False, indent=2)

special_tokens_map = {
    "unk_token": "[UNK]",
    "pad_token": "[PAD]",
    "bos_token": "[BOS]",
    "eos_token": "[EOS]",
    "sep_token": "[SEP]",
    "mask_token": "[MASK]",
    "additional_special_tokens": [
        "[INST]", "[/INST]", "[SYS]", "[USER]", "[ASST]",
        "[TURN]", "[LANG_ID]", "[LANG_JV]", "[LANG_SU]", "[LANG_EN]"
    ]
}

with open(OUTPUT_DIR / "special_tokens_map.json", "w", encoding="utf-8") as f:
    json.dump(special_tokens_map, f, ensure_ascii=False, indent=2)

print("🎉 Tokenizer v3 selesai dibuat!")
print(f"📁 Output folder: {OUTPUT_DIR.resolve()}")