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

Script sudah include automatic testing:

```
--- TEST HASIL ---
Teks Asli : Selamat pagi Nusantara! 123 + 456 = 579. [INST] Apa kabar? [/INST]
Tokens    : ['Selamat', 'Ġpagi', 'Ġ', 'Nusantara', '!', 'Ġ', '1', '2', '3', 'Ġ+', 'Ġ', '4', '5', '6', 'Ġ=', 'Ġ', '5', '7', '9', '.', 'Ġ', '[INST]', 'ĠApa', 'Ġkabar', '?', 'Ġ', '[/INST]']
Total IDs : 27 tokens
```

---

**Last Updated:** April 2026  
**Version:** 1.0
