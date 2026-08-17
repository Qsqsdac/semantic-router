use std::io::Cursor;

use fasttext_pure_rs::FastText;

pub struct FastTextClassifier {
    model: FastText,
}

#[derive(Debug, Clone)]
pub struct FastTextPrediction {
    pub label: String,
    pub probability: f32,
}

impl FastTextClassifier {
    pub fn load(model_path: &str) -> Result<Self, String> {
        let bytes = std::fs::read(model_path)
            .map_err(|e| format!("failed to read fastText model {model_path:?}: {e}"))?;
        let model = FastText::load_from_reader(Cursor::new(bytes))
            .map_err(|e| format!("failed to load fastText model {model_path:?}: {e}"))?;
        Ok(Self { model })
    }

    pub fn predict(
        &self,
        text: &str,
        k: usize,
        threshold: f32,
    ) -> Result<Option<FastTextPrediction>, String> {
        let predictions = self
            .model
            .predict(text, k.max(1), threshold.max(0.0))
            .map_err(|e| format!("fastText prediction failed: {e}"))?;
        Ok(predictions.first().map(|prediction| FastTextPrediction {
            label: prediction.label.clone(),
            probability: prediction.probability,
        }))
    }
}
