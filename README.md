# 🎯 AuraLLM Custom Tokenizer

Tokenizer kustom yang dioptimalkan untuk **Bahasa Indonesia dan bahasa regional** dengan performa superior untuk training dan inference model Language Model.

## ✨ Mengapa Tokenizer Kustom?

Tokenizer standar (GPT-4, LLaMA) tidak dioptimalkan untuk bahasa Indonesia. AuraLLM Tokenizer memberikan:

| Aspek | GPT-4 | LLaMA-2 | **AuraLLM** |
|-------|-------|---------|-----------|
| "Selamat pagi" | Sel amat pagi (3 tokens) | Se lam at p agi (5 tokens) | **Selamat pagi (2 tokens)** ✅ |
| **Hasil** | Inference Lambat | Training Mahal | **Lebih Cepat & Murah** |

### Manfaat:
- ⚡ **Faster Inference** - Fewer tokens = faster generation
- 💰 **Cheaper Training** - Reduce token count significantly
- 🎓 **Better Model Quality** - Language-specific optimization
- 🌍 **Multi-Regional Support** - Indonesian + regional languages

---

## 🧪 Testing


```
tok = PreTrainedTokenizerFast.from_pretrained("./aura_tokenizer_v2")

sample = "AuraLLM adalah model bahasa untuk Indonesia dan Nusantara."
enc = tok(sample)

print(tok.tokenize(sample))
print(enc["input_ids"])
print(tok.convert_ids_to_tokens(enc["input_ids"]))
print(tok.decode(enc["input_ids"]))

# Output
['AuraLLM', 'adalah', 'model', 'bahasa', 'untuk', 'Indonesia', 'dan', 'Nusantara', '.']
[2, 16, 12979, 15111, 13382, 12960, 19, 12913, 24, 45, 1]
['[BOS]', 'AuraLLM', 'adalah', 'model', 'bahasa', 'untuk', 'Indonesia', 'dan', 'Nusantara', '.', '[EOS]']
[BOS]AuraLLMadalahmodelbahasauntukIndonesiadanNusantara.[EOS]

```

---

**Last Updated:** April 2026  
**Version:** 2.0
