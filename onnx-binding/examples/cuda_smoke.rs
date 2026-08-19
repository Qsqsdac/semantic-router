//! Temporary smoke test for ONNX Runtime CUDA execution provider.
//! Verifies classifier (Auto) and embedding (use_cpu=false) both land on CUDA EP.

use onnx_semantic_router::{
    ClassifierExecutionProvider, MmBertEmbeddingModel, MmBertSequenceClassifier,
};

fn main() {
    let args: Vec<String> = std::env::args().collect();
    let classifier_path = args
        .get(1)
        .expect("usage: cuda_smoke <classifier_model_path> <embedding_model_path>");
    let embedding_path = args
        .get(2)
        .expect("usage: cuda_smoke <classifier_model_path> <embedding_model_path>");

    println!("=== Classifier (Auto) ===");
    let mut classifier =
        MmBertSequenceClassifier::load(classifier_path, ClassifierExecutionProvider::Auto)
            .expect("classifier load failed");
    let result = classifier
        .classify("What is gradient descent?")
        .expect("classify failed");
    println!(
        "classify: label={} conf={:.4}",
        result.label, result.confidence
    );

    println!("=== Embedding (use_cpu=false) ===");
    let mut embedding = MmBertEmbeddingModel::load(embedding_path, false)
        .expect("embedding load failed");
    let vec = embedding
        .encode_single("Hello world", None, None)
        .expect("encode failed");
    println!("embedding: dims={}", vec.len());

    println!("SMOKE OK");
}
