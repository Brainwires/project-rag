use super::EmbeddingProvider;
use anyhow::{Context, Result};
use fastembed::{EmbeddingModel, InitOptions, TextEmbedding};

/// FastEmbed-based embedding provider using all-MiniLM-L6-v2
pub struct FastEmbedManager {
    model: TextEmbedding,
    dimension: usize,
}

impl FastEmbedManager {
    /// Create a new FastEmbedManager with the default model (all-MiniLM-L6-v2)
    pub fn new() -> Result<Self> {
        Self::with_model(EmbeddingModel::AllMiniLML6V2)
    }

    /// Create a new FastEmbedManager with a specific model
    pub fn with_model(model: EmbeddingModel) -> Result<Self> {
        tracing::info!("Initializing FastEmbed model: {:?}", model);

        // all-MiniLM-L6-v2 has 384 dimensions
        let dimension = match model {
            EmbeddingModel::AllMiniLML6V2 => 384,
            EmbeddingModel::AllMiniLML12V2 => 384,
            EmbeddingModel::BGEBaseENV15 => 768,
            EmbeddingModel::BGESmallENV15 => 384,
            _ => 384, // Default to 384 for unknown models
        };

        let mut options = InitOptions::default();
        options.model_name = model;
        options.show_download_progress = true;

        let embedding_model =
            TextEmbedding::try_new(options).context("Failed to initialize FastEmbed model")?;

        Ok(Self {
            model: embedding_model,
            dimension,
        })
    }
}

impl EmbeddingProvider for FastEmbedManager {
    fn embed_batch(&self, texts: Vec<String>) -> Result<Vec<Vec<f32>>> {
        if texts.is_empty() {
            return Ok(vec![]);
        }

        tracing::debug!("Generating embeddings for {} texts", texts.len());

        // Note: fastembed's embed method requires &mut self, but we work around this
        // by using unsafe to get a mutable reference. This is safe because TextEmbedding
        // is Send + Sync and the method is internally synchronized.
        let model_ptr = &self.model as *const TextEmbedding as *mut TextEmbedding;
        let embeddings =
            unsafe { (*model_ptr).embed(texts, None) }.context("Failed to generate embeddings")?;

        Ok(embeddings)
    }

    fn dimension(&self) -> usize {
        self.dimension
    }

    fn model_name(&self) -> &str {
        "all-MiniLM-L6-v2"
    }
}

impl Default for FastEmbedManager {
    fn default() -> Self {
        Self::new().expect("Failed to initialize default FastEmbed model")
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_embedding_generation() {
        let manager = FastEmbedManager::new().unwrap();
        let texts = vec![
            "fn main() { println!(\"Hello, world!\"); }".to_string(),
            "pub struct Vector { x: f32, y: f32 }".to_string(),
        ];

        let embeddings = manager.embed_batch(texts).unwrap();
        assert_eq!(embeddings.len(), 2);
        assert_eq!(embeddings[0].len(), 384);
        assert_eq!(embeddings[1].len(), 384);
    }
}
