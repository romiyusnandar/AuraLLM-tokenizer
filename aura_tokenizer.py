import os
from datasets import load_dataset
from tokenizers import Tokenizer, models, trainers, pre_tokenizers, decoders, processors, normalizers
from transformers import PreTrainedTokenizerFast

# Ganti dengan path dataset yang kamu temukan di Hugging Face
DATASET_NAME = "AksaraLLM/aksara-pretrain-id" 
TEXT_COLUMN = "text"

print("📥 Loading dataset...")
dataset = load_dataset(DATASET_NAME, split="train")

def batch_iterator(batch_size=1000):
    for i in range(0, len(dataset), batch_size):
        yield dataset[i : i + batch_size][TEXT_COLUMN]

tokenizer = Tokenizer(models.BPE(unk_token="[UNK]"))

tokenizer.normalizer = normalizers.NFKC()

# Menggunakan read_number=False agar angka dipisah per digit untuk logika matematika yang lebih baik
tokenizer.pre_tokenizer = pre_tokenizers.Sequence([
    pre_tokenizers.Digits(individual_digits=True),
    pre_tokenizers.ByteLevel(add_prefix_space=False),
])

special_tokens = [
    "[PAD]", "[EOS]", "[BOS]", "[UNK]", "[SEP]", "[MASK]",
    "[INST]", "[/INST]", "[SYS]", "[USER]", "[ASST]", "[TURN]",
    "[LANG_ID]", "[LANG_JV]", "[LANG_SU]", "[LANG_EN]",
    "AuraLLM", "Indonesia", "Pancasila", "Nusantara" 
]

# Vocab size 32.768 adalah standar yang pas untuk efisiensi vs performa
trainer = trainers.BpeTrainer(
    vocab_size=32768, 
    min_frequency=2,
    initial_alphabet=pre_tokenizers.ByteLevel.alphabet(),
    special_tokens=special_tokens
)

print("🚀 Memulai training tokenizer (AuraLLM Style)...")
tokenizer.train_from_iterator(batch_iterator(), trainer=trainer)

tokenizer.decoder = decoders.ByteLevel()
tokenizer.post_processor = processors.ByteLevel(trim_offsets=True)

hf_tokenizer = PreTrainedTokenizerFast(
    tokenizer_object=tokenizer,
    bos_token="[BOS]",
    eos_token="[EOS]",
    unk_token="[UNK]",
    pad_token="[PAD]",
    sep_token="[SEP]",
    mask_token="[MASK]",
    additional_special_tokens=["[INST]", "[/INST]", "[SYS]", "[USER]", "[ASST]"]
)

OUTPUT_DIR = "aura-tokenizer-custom"
os.makedirs(OUTPUT_DIR, exist_ok=True)
hf_tokenizer.save_pretrained(OUTPUT_DIR)

print(f"\n✅ Selesai! Tokenizer tersimpan di folder: {OUTPUT_DIR}")
print(f"   Daftar File: {os.listdir(OUTPUT_DIR)}")

# --- TEST TOKENIZER ---
print("\n--- TEST HASIL ---")
test_text = "Selamat pagi Nusantara! 123 + 456 = 579. [INST] Apa kabar? [/INST]"
encoded = hf_tokenizer.encode(test_text)
print(f"Teks Asli : {test_text}")
print(f"Tokens    : {hf_tokenizer.convert_ids_to_tokens(encoded)}")
print(f"Total IDs : {len(encoded)} tokens")