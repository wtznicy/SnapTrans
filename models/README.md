# SnapTrans Models

SnapTrans uses MarianMT ONNX models with KV Cache autoregressive decoding.

## Quick Download / Export

### 1. Export with Optimum CLI
```bash
pip install optimum[onnxruntime] transformers sentencepiece

# Export English -> Chinese
optimum-cli export onnx \
    --model Helsinki-NLP/opus-mt-en-zh \
    --task text2text-generation-with-past \
    models/snaptrans/en_zh/
```

### 2. Generate tokenizer.json
```bash
python tools/make_tokenizer.py models/snaptrans/en_zh/source.spm models/snaptrans/en_zh/vocab.json models/snaptrans/en_zh/tokenizer.json
```

### 3. (Optional) Dynamic INT8 Quantization
```bash
python -c "
from onnxruntime.quantization import quantize_dynamic, QuantType
quantize_dynamic('models/snaptrans/en_zh/encoder_model.onnx', 'models/snaptrans/en_zh/encoder_model_int8.onnx', weight_type=QuantType.QInt8)
"
```
