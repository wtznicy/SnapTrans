//! Benchmarks for SnapTrans pipeline stages.

use criterion::{black_box, criterion_group, criterion_main, Criterion};

fn bench_text_normalize(c: &mut Criterion) {
    let input = "  Hello，World！  This is\n  a test。  Full-width：ＡＢＣ  ";

    c.bench_function("text_normalize", |b| {
        b.iter(|| {
            let mut result = String::with_capacity(input.len());
            let mut prev_space = false;
            for ch in input.chars() {
                let ch = match ch {
                    '\u{FF01}'..='\u{FF5E}' => (ch as u32 - 0xFF01 + 0x21) as u8 as char,
                    '\u{3000}' => ' ',
                    _ => ch,
                };
                match ch {
                    '\n' | '\r' | '\t' | ' ' => {
                        if !prev_space {
                            result.push(' ');
                            prev_space = true;
                        }
                    }
                    _ => {
                        result.push(ch);
                        prev_space = false;
                    }
                }
            }
            black_box(result);
        });
    });
}

fn bench_lang_detect(c: &mut Criterion) {
    let english = "This is a sample English sentence for language detection benchmarking.";
    let chinese = "这是一个用于语言检测基准测试的中文示例句子。";
    let mixed = "这是一个Mixed混合Language语言Detection检测的Test测试。";

    c.bench_function("lang_detect_english", |b| {
        b.iter(|| {
            let mut cjk = 0u32;
            let mut total = 0u32;
            for ch in english.chars() {
                if !ch.is_whitespace() {
                    total += 1;
                    if matches!(ch, '\u{4E00}'..='\u{9FFF}') {
                        cjk += 1;
                    }
                }
            }
            black_box(cjk as f32 / total.max(1) as f32 > 0.3);
        });
    });

    c.bench_function("lang_detect_chinese", |b| {
        b.iter(|| {
            let mut cjk = 0u32;
            let mut total = 0u32;
            for ch in chinese.chars() {
                if !ch.is_whitespace() {
                    total += 1;
                    if matches!(ch, '\u{4E00}'..='\u{9FFF}') {
                        cjk += 1;
                    }
                }
            }
            black_box(cjk as f32 / total.max(1) as f32 > 0.3);
        });
    });

    c.bench_function("lang_detect_mixed", |b| {
        b.iter(|| {
            let mut cjk = 0u32;
            let mut total = 0u32;
            for ch in mixed.chars() {
                if !ch.is_whitespace() {
                    total += 1;
                    if matches!(ch, '\u{4E00}'..='\u{9FFF}') {
                        cjk += 1;
                    }
                }
            }
            black_box(cjk as f32 / total.max(1) as f32 > 0.3);
        });
    });
}

fn bench_postprocess(c: &mut Criterion) {
    let chinese_output = "你 好 ， 世 界 。 这 是 一 个 测 试 。";

    c.bench_function("postprocess_chinese", |b| {
        b.iter(|| {
            let chars: Vec<char> = chinese_output.chars().collect();
            let mut result = String::with_capacity(chinese_output.len());
            let mut i = 0;
            while i < chars.len() {
                if chars[i] == ' ' {
                    let prev_cjk = i > 0 && matches!(chars[i-1], '\u{4E00}'..='\u{9FFF}');
                    let mut j = i;
                    while j < chars.len() && chars[j] == ' ' { j += 1; }
                    let next_cjk = j < chars.len() && matches!(chars[j], '\u{4E00}'..='\u{9FFF}');
                    if !(prev_cjk && next_cjk) {
                        result.push(' ');
                    }
                    i = j;
                } else {
                    result.push(chars[i]);
                    i += 1;
                }
            }
            black_box(result);
        });
    });
}

fn bench_repetition_penalty(c: &mut Criterion) {
    let mut logits: Vec<f32> = (0..58101).map(|i| (i as f32 - 29000.0) / 1000.0).collect();
    let generated: Vec<i64> = (0..50).collect();

    c.bench_function("repetition_penalty_58k_vocab", |b| {
        b.iter(|| {
            let penalty = 1.15f32;
            for &token_id in &generated {
                let idx = token_id as usize;
                if idx < logits.len() {
                    if logits[idx] > 0.0 {
                        logits[idx] /= penalty;
                    } else {
                        logits[idx] *= penalty;
                    }
                }
            }
            black_box(&logits);
        });
    });
}

criterion_group!(
    benches,
    bench_text_normalize,
    bench_lang_detect,
    bench_postprocess,
    bench_repetition_penalty,
);
criterion_main!(benches);
