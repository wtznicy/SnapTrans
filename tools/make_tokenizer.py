"""
Helper script to build tokenizer.json for HuggingFace `tokenizers` crate from MarianMT source.spm and vocab.json.
"""

import sys
import json
from tokenizers import Tokenizer, decoders, AddedToken
from tokenizers.models import Unigram
from tokenizers.processors import TemplateProcessing
from transformers.convert_slow_tokenizer import SpmConverter, import_protobuf

def build_tokenizer(spm_path: str, vocab_path: str, output_path: str):
    print(f"Loading protobuf from {spm_path}...")
    model_pb2 = import_protobuf()
    m = model_pb2.ModelProto()
    with open(spm_path, 'rb') as f:
        m.ParseFromString(f.read())
    spm_scores = {p.piece: p.score for p in m.pieces}

    print(f"Loading vocab mapping from {vocab_path}...")
    with open(vocab_path, 'r', encoding='utf-8') as f:
        vocab = json.load(f)

    # Create vocab list mapped to MarianMT joint vocab IDs
    vocab_list = [None] * len(vocab)
    for piece, idx in vocab.items():
        score = spm_scores.get(piece, -100.0)
        vocab_list[idx] = (piece, score)

    unk_id = vocab.get('<unk>', 1)

    dummy = type('Dummy', (), {'vocab_file': spm_path, 'add_prefix_space': True})()
    c = SpmConverter(dummy)
    norm = c.normalizer(m)
    pre_tok = c.pre_tokenizer('\u2581', True)
    decoder = decoders.Metaspace(replacement='\u2581', prepend_scheme='always')

    tok_model = Unigram(vocab_list, unk_id=unk_id)
    tok = Tokenizer(tok_model)
    tok.normalizer = norm
    tok.pre_tokenizer = pre_tok
    tok.decoder = decoder

    tok.add_special_tokens([
        AddedToken('</s>', special=True),
        AddedToken('<unk>', special=True),
        AddedToken('<pad>', special=True)
    ])

    tok.post_processor = TemplateProcessing(
        single="$A </s>",
        pair="$A </s> $B </s>",
        special_tokens=[("</s>", vocab.get('</s>', 0))]
    )

    tok.save(output_path)
    print(f"Successfully generated {output_path}")

    # Test encode
    enc = tok.encode("Hello, world!")
    print("Test encode 'Hello, world!':", enc.ids)

if __name__ == '__main__':
    spm = sys.argv[1] if len(sys.argv) > 1 else 'models/snaptrans/en_zh/source.spm'
    vocab = sys.argv[2] if len(sys.argv) > 2 else 'models/snaptrans/en_zh/vocab.json'
    out = sys.argv[3] if len(sys.argv) > 3 else 'models/snaptrans/en_zh/tokenizer.json'
    build_tokenizer(spm, vocab, out)
