import json
from pathlib import Path

from datasets import load_dataset
from tokenizers import Tokenizer, models, normalizers, pre_tokenizers, decoders, processors
from tokenizers.trainers import BpeTrainer
from transformers import PreTrainedTokenizerFast

# =========================================================
# CONFIG
# =========================================================
DATASET_NAME = "AksaraLLM/aksara-pretrain-id"
OUTPUT_DIR = Path("./aura_tokenizer_v2")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

VERSION = "2.0.0"
VOCAB_SIZE = 32773
MIN_FREQUENCY = 2

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

ALL_ADDED_TOKENS = SPECIAL_TOKENS + BRAND_TOKENS

# =========================================================
# LOAD DATASET
# =========================================================
dataset = load_dataset(DATASET_NAME)

train_ds = dataset["train"]

def batch_iterator(batch_size=1000):
    for i in range(0, len(train_ds), batch_size):
        batch = train_ds[i:i + batch_size]
        texts = batch["text"]
        # filter null/empty
        texts = [t for t in texts if isinstance(t, str) and t.strip()]
        if texts:
            yield texts

# =========================================================
# BUILD TOKENIZER
# =========================================================
tokenizer = Tokenizer(models.BPE(unk_token="[UNK]"))

# NFKC tanpa lowercase
tokenizer.normalizer = normalizers.NFKC()

# Pre-tokenizer Whitespace
tokenizer.pre_tokenizer = pre_tokenizers.Whitespace()

# Decoder BPE
tokenizer.decoder = decoders.BPEDecoder()

trainer = BpeTrainer(
    vocab_size=VOCAB_SIZE,
    min_frequency=MIN_FREQUENCY,
    special_tokens=ALL_ADDED_TOKENS,
    show_progress=True,
)

tokenizer.train_from_iterator(batch_iterator(), trainer=trainer)

# =========================================================
# SPECIAL TOKEN IDS
# =========================================================
special_token_ids = {tok: tokenizer.token_to_id(tok) for tok in SPECIAL_TOKENS}

# Validasi dasar
for tok in SPECIAL_TOKENS:
    if special_token_ids[tok] is None:
        raise ValueError(f"Special token {tok} tidak masuk vocab.")

# =========================================================
# POST PROCESSOR
# Single: [BOS] X [EOS]
# Pair  : [BOS] A [SEP] B [EOS]
# =========================================================
tokenizer.post_processor = processors.TemplateProcessing(
    single="[BOS] $A [EOS]",
    pair="[BOS] $A [SEP] $B [EOS]",
    special_tokens=[
        ("[BOS]", special_token_ids["[BOS]"]),
        ("[EOS]", special_token_ids["[EOS]"]),
        ("[SEP]", special_token_ids["[SEP]"]),
    ],
)

# Save raw tokenizer
tokenizer_json_path = OUTPUT_DIR / "tokenizer.json"
tokenizer.save(str(tokenizer_json_path))

# =========================================================
# WRAP AS PreTrainedTokenizerFast
# =========================================================
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

# Save HF tokenizer files
fast_tokenizer.save_pretrained(str(OUTPUT_DIR))

# =========================================================
# CUSTOM tokenizer_config.json
# =========================================================
tokenizer_config = {
    "version": VERSION,
    "vocab_size": VOCAB_SIZE,
    "pre_tokenizer": "Whitespace",
    "normalizer": "NFKC (NO lowercase)",
    "model": "BPE",
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

# =========================================================
# SPECIAL TOKENS MAP
# =========================================================
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

print(f"Tokenizer selesai disimpan di: {OUTPUT_DIR.resolve()}")
print("Special token IDs:")
for k, v in special_token_ids.items():
    print(f"{k}: {v}")