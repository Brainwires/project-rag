use crate::types::{ChunkMetadata, SearchResult};
use crate::vector_db::{DatabaseStats, VectorDatabase};
use anyhow::{Context, Result};
use arrow_array::{
    types::Float32Type, Array, FixedSizeListArray, Float32Array, RecordBatch, RecordBatchIterator,
    StringArray, UInt32Array,
};
use arrow_schema::{DataType, Field, Schema};
use futures::stream::TryStreamExt;
use lancedb::connection::Connection;
use lancedb::query::{ExecutableQuery, QueryBase};
use lancedb::Table;
use std::collections::HashMap;
use std::sync::Arc;

/// LanceDB vector database implementation (embedded, no server required)
pub struct LanceVectorDB {
    connection: Connection,
    table_name: String,
    db_path: String,
}

impl LanceVectorDB {
    /// Create a new LanceDB instance with default path
    pub async fn new() -> Result<Self> {
        let db_path = Self::default_path();
        Self::with_path(&db_path).await
    }

    /// Create a new LanceDB instance with custom path
    pub async fn with_path(db_path: &str) -> Result<Self> {
        tracing::info!("Connecting to LanceDB at: {}", db_path);

        let connection = lancedb::connect(db_path)
            .execute()
            .await
            .context("Failed to connect to LanceDB")?;

        Ok(Self {
            connection,
            table_name: "code_embeddings".to_string(),
            db_path: db_path.to_string(),
        })
    }

    /// Get default database path
    fn default_path() -> String {
        let data_dir = if cfg!(target_os = "windows") {
            std::env::var("LOCALAPPDATA").unwrap_or_else(|_| ".".to_string())
        } else if cfg!(target_os = "macos") {
            format!(
                "{}/Library/Application Support",
                std::env::var("HOME").unwrap_or_else(|_| ".".to_string())
            )
        } else {
            // Linux/Unix
            std::env::var("XDG_DATA_HOME").unwrap_or_else(|_| {
                format!(
                    "{}/.local/share",
                    std::env::var("HOME").unwrap_or_else(|_| ".".to_string())
                )
            })
        };

        format!("{}/project-rag/lancedb", data_dir)
    }

    /// Create schema for the embeddings table
    fn create_schema(dimension: usize) -> Arc<Schema> {
        Arc::new(Schema::new(vec![
            Field::new(
                "vector",
                DataType::FixedSizeList(
                    Arc::new(Field::new("item", DataType::Float32, true)),
                    dimension as i32,
                ),
                false,
            ),
            Field::new("id", DataType::Utf8, false),
            Field::new("file_path", DataType::Utf8, false),
            Field::new("start_line", DataType::UInt32, false),
            Field::new("end_line", DataType::UInt32, false),
            Field::new("language", DataType::Utf8, false),
            Field::new("extension", DataType::Utf8, false),
            Field::new("file_hash", DataType::Utf8, false),
            Field::new("indexed_at", DataType::Utf8, false),
            Field::new("content", DataType::Utf8, false),
            Field::new("project", DataType::Utf8, true),
        ]))
    }

    /// Get or create table
    async fn get_table(&self) -> Result<Table> {
        self.connection
            .open_table(&self.table_name)
            .execute()
            .await
            .context("Failed to open table")
    }

    /// Convert embeddings and metadata to RecordBatch
    fn create_record_batch(
        embeddings: Vec<Vec<f32>>,
        metadata: Vec<ChunkMetadata>,
        contents: Vec<String>,
        schema: Arc<Schema>,
    ) -> Result<RecordBatch> {
        let num_rows = embeddings.len();
        let dimension = embeddings[0].len();

        // Create FixedSizeListArray for vectors
        let vector_array = FixedSizeListArray::from_iter_primitive::<Float32Type, _, _>(
            embeddings.into_iter().map(|v| Some(v.into_iter().map(Some))),
            dimension as i32,
        );

        // Create arrays for each field
        let id_array = StringArray::from(
            (0..num_rows)
                .map(|i| format!("{}:{}", metadata[i].file_path, metadata[i].start_line))
                .collect::<Vec<_>>(),
        );
        let file_path_array = StringArray::from(
            metadata
                .iter()
                .map(|m| m.file_path.as_str())
                .collect::<Vec<_>>(),
        );
        let start_line_array = UInt32Array::from(
            metadata
                .iter()
                .map(|m| m.start_line as u32)
                .collect::<Vec<_>>(),
        );
        let end_line_array = UInt32Array::from(
            metadata
                .iter()
                .map(|m| m.end_line as u32)
                .collect::<Vec<_>>(),
        );
        let language_array = StringArray::from(
            metadata
                .iter()
                .map(|m| m.language.as_deref().unwrap_or("Unknown"))
                .collect::<Vec<_>>(),
        );
        let extension_array = StringArray::from(
            metadata
                .iter()
                .map(|m| m.extension.as_deref().unwrap_or(""))
                .collect::<Vec<_>>(),
        );
        let file_hash_array = StringArray::from(
            metadata
                .iter()
                .map(|m| m.file_hash.as_str())
                .collect::<Vec<_>>(),
        );
        let indexed_at_array = StringArray::from(
            metadata
                .iter()
                .map(|m| m.indexed_at.to_string())
                .collect::<Vec<_>>(),
        );
        let content_array =
            StringArray::from(contents.iter().map(|s| s.as_str()).collect::<Vec<_>>());
        let project_array = StringArray::from(
            metadata
                .iter()
                .map(|m| m.project.as_deref())
                .collect::<Vec<_>>(),
        );

        RecordBatch::try_new(
            schema,
            vec![
                Arc::new(vector_array),
                Arc::new(id_array),
                Arc::new(file_path_array),
                Arc::new(start_line_array),
                Arc::new(end_line_array),
                Arc::new(language_array),
                Arc::new(extension_array),
                Arc::new(file_hash_array),
                Arc::new(indexed_at_array),
                Arc::new(content_array),
                Arc::new(project_array),
            ],
        )
        .context("Failed to create RecordBatch")
    }
}

impl VectorDatabase for LanceVectorDB {
    async fn initialize(&self, dimension: usize) -> Result<()> {
        tracing::info!(
            "Initializing LanceDB with dimension {} at {}",
            dimension,
            self.db_path
        );

        // Check if table exists
        let table_names = self
            .connection
            .table_names()
            .execute()
            .await
            .context("Failed to list tables")?;

        if table_names.contains(&self.table_name) {
            tracing::info!("Table '{}' already exists", self.table_name);
            return Ok(());
        }

        // Create empty table with schema
        let schema = Self::create_schema(dimension);

        // Create empty RecordBatch
        let empty_batch = RecordBatch::new_empty(schema.clone());

        // Need to wrap in iterator that returns Result<RecordBatch>
        let batches = RecordBatchIterator::new(
            vec![empty_batch].into_iter().map(Ok),
            schema.clone(),
        );

        self.connection
            .create_table(&self.table_name, Box::new(batches))
            .execute()
            .await
            .context("Failed to create table")?;

        tracing::info!("Created table '{}'", self.table_name);
        Ok(())
    }

    async fn store_embeddings(
        &self,
        embeddings: Vec<Vec<f32>>,
        metadata: Vec<ChunkMetadata>,
        contents: Vec<String>,
    ) -> Result<usize> {
        if embeddings.is_empty() {
            return Ok(0);
        }

        let dimension = embeddings[0].len();
        let schema = Self::create_schema(dimension);

        let batch = Self::create_record_batch(embeddings, metadata, contents, schema.clone())?;
        let count = batch.num_rows();

        let table = self.get_table().await?;

        let batches = RecordBatchIterator::new(vec![batch].into_iter().map(Ok), schema);

        table
            .add(Box::new(batches))
            .execute()
            .await
            .context("Failed to add records to table")?;

        tracing::info!("Stored {} embeddings", count);
        Ok(count)
    }

    async fn search(
        &self,
        query_vector: Vec<f32>,
        _query_text: &str,
        limit: usize,
        min_score: f32,
        project: Option<String>,
        _hybrid: bool,
    ) -> Result<Vec<SearchResult>> {
        let table = self.get_table().await?;

        let query = table
            .vector_search(query_vector)
            .context("Failed to create vector search")?
            .limit(limit);

        // Add project filter if specified and execute
        let stream = if let Some(ref project_name) = project {
            query.only_if(format!("project = '{}'", project_name)).execute().await.context("Failed to execute search")?
        } else {
            query.execute().await.context("Failed to execute search")?
        };

        let results: Vec<RecordBatch> = stream
            .try_collect()
            .await
            .context("Failed to collect search results")?;

        let mut search_results = Vec::new();

        for batch in results {
            let file_path_array = batch
                .column_by_name("file_path")
                .context("Missing file_path column")?
                .as_any()
                .downcast_ref::<StringArray>()
                .context("Invalid file_path type")?;

            let start_line_array = batch
                .column_by_name("start_line")
                .context("Missing start_line column")?
                .as_any()
                .downcast_ref::<UInt32Array>()
                .context("Invalid start_line type")?;

            let end_line_array = batch
                .column_by_name("end_line")
                .context("Missing end_line column")?
                .as_any()
                .downcast_ref::<UInt32Array>()
                .context("Invalid end_line type")?;

            let language_array = batch
                .column_by_name("language")
                .context("Missing language column")?
                .as_any()
                .downcast_ref::<StringArray>()
                .context("Invalid language type")?;

            let content_array = batch
                .column_by_name("content")
                .context("Missing content column")?
                .as_any()
                .downcast_ref::<StringArray>()
                .context("Invalid content type")?;

            let project_array = batch
                .column_by_name("project")
                .context("Missing project column")?
                .as_any()
                .downcast_ref::<StringArray>()
                .context("Invalid project type")?;

            // LanceDB returns distance, we need to convert to similarity score
            let distance_array = batch
                .column_by_name("_distance")
                .context("Missing _distance column")?
                .as_any()
                .downcast_ref::<Float32Array>()
                .context("Invalid _distance type")?;

            for i in 0..batch.num_rows() {
                let distance = distance_array.value(i);
                // Convert L2 distance to similarity score (0-1 range)
                let score = 1.0 / (1.0 + distance);

                if score >= min_score {
                    search_results.push(SearchResult {
                        score,
                        vector_score: score,
                        keyword_score: None,
                        file_path: file_path_array.value(i).to_string(),
                        start_line: start_line_array.value(i) as usize,
                        end_line: end_line_array.value(i) as usize,
                        language: language_array.value(i).to_string(),
                        content: content_array.value(i).to_string(),
                        project: if project_array.is_null(i) {
                            None
                        } else {
                            Some(project_array.value(i).to_string())
                        },
                    });
                }
            }
        }

        Ok(search_results)
    }

    async fn search_filtered(
        &self,
        query_vector: Vec<f32>,
        _query_text: &str,
        limit: usize,
        min_score: f32,
        project: Option<String>,
        _hybrid: bool,
        file_extensions: Vec<String>,
        languages: Vec<String>,
        path_patterns: Vec<String>,
    ) -> Result<Vec<SearchResult>> {
        let table = self.get_table().await?;

        let query = table
            .vector_search(query_vector)
            .context("Failed to create vector search")?
            .limit(limit * 2); // Get more results for filtering

        // Build filter expression
        let mut filters = Vec::new();

        if let Some(ref project_name) = project {
            filters.push(format!("project = '{}'", project_name));
        }

        if !file_extensions.is_empty() {
            let ext_filters: Vec<String> = file_extensions
                .iter()
                .map(|ext| format!("extension = '{}'", ext))
                .collect();
            filters.push(format!("({})", ext_filters.join(" OR ")));
        }

        if !languages.is_empty() {
            let lang_filters: Vec<String> = languages
                .iter()
                .map(|lang| format!("language = '{}'", lang))
                .collect();
            filters.push(format!("({})", lang_filters.join(" OR ")));
        }

        // Execute with or without filters
        let stream = if !filters.is_empty() {
            query.only_if(filters.join(" AND ")).execute().await.context("Failed to execute filtered search")?
        } else {
            query.execute().await.context("Failed to execute filtered search")?
        };

        let results: Vec<RecordBatch> = stream
            .try_collect()
            .await
            .context("Failed to collect filtered results")?;

        let mut search_results = Vec::new();

        for batch in results {
            let file_path_array = batch
                .column_by_name("file_path")
                .context("Missing file_path column")?
                .as_any()
                .downcast_ref::<StringArray>()
                .context("Invalid file_path type")?;

            let start_line_array = batch
                .column_by_name("start_line")
                .context("Missing start_line column")?
                .as_any()
                .downcast_ref::<UInt32Array>()
                .context("Invalid start_line type")?;

            let end_line_array = batch
                .column_by_name("end_line")
                .context("Missing end_line column")?
                .as_any()
                .downcast_ref::<UInt32Array>()
                .context("Invalid end_line type")?;

            let language_array = batch
                .column_by_name("language")
                .context("Missing language column")?
                .as_any()
                .downcast_ref::<StringArray>()
                .context("Invalid language type")?;

            let content_array = batch
                .column_by_name("content")
                .context("Missing content column")?
                .as_any()
                .downcast_ref::<StringArray>()
                .context("Invalid content type")?;

            let project_array = batch
                .column_by_name("project")
                .context("Missing project column")?
                .as_any()
                .downcast_ref::<StringArray>()
                .context("Invalid project type")?;

            let distance_array = batch
                .column_by_name("_distance")
                .context("Missing _distance column")?
                .as_any()
                .downcast_ref::<Float32Array>()
                .context("Invalid _distance type")?;

            for i in 0..batch.num_rows() {
                let file_path = file_path_array.value(i).to_string();

                // Apply path pattern filter (post-processing)
                if !path_patterns.is_empty() {
                    if !path_patterns
                        .iter()
                        .any(|pattern| file_path.contains(pattern))
                    {
                        continue;
                    }
                }

                let distance = distance_array.value(i);
                let score = 1.0 / (1.0 + distance);

                if score >= min_score {
                    search_results.push(SearchResult {
                        score,
                        vector_score: score,
                        keyword_score: None,
                        file_path,
                        start_line: start_line_array.value(i) as usize,
                        end_line: end_line_array.value(i) as usize,
                        language: language_array.value(i).to_string(),
                        content: content_array.value(i).to_string(),
                        project: if project_array.is_null(i) {
                            None
                        } else {
                            Some(project_array.value(i).to_string())
                        },
                    });
                }

                if search_results.len() >= limit {
                    break;
                }
            }
        }

        // Truncate to limit
        search_results.truncate(limit);

        Ok(search_results)
    }

    async fn delete_by_file(&self, file_path: &str) -> Result<usize> {
        let table = self.get_table().await?;

        // LanceDB uses SQL-like delete
        let filter = format!("file_path = '{}'", file_path);

        table
            .delete(&filter)
            .await
            .context("Failed to delete records")?;

        tracing::info!("Deleted embeddings for file: {}", file_path);

        // LanceDB doesn't return count directly, return 0 as placeholder
        Ok(0)
    }

    async fn clear(&self) -> Result<()> {
        // Drop and recreate table
        self.connection
            .drop_table(&self.table_name)
            .await
            .context("Failed to drop table")?;

        tracing::info!("Cleared all embeddings");
        Ok(())
    }

    async fn get_statistics(&self) -> Result<DatabaseStats> {
        let table = self.get_table().await?;

        // Count total vectors
        let count_result = table
            .count_rows(None)
            .await
            .context("Failed to count rows")?;

        // Get language breakdown by scanning the table
        let stream = table
            .query()
            .select(lancedb::query::Select::Columns(vec![
                "language".to_string(),
            ]))
            .execute()
            .await
            .context("Failed to query languages")?;

        let query_result: Vec<RecordBatch> = stream
            .try_collect()
            .await
            .context("Failed to collect language data")?;

        let mut language_counts: HashMap<String, usize> = HashMap::new();

        for batch in query_result {
            let language_array = batch
                .column_by_name("language")
                .context("Missing language column")?
                .as_any()
                .downcast_ref::<StringArray>()
                .context("Invalid language type")?;

            for i in 0..batch.num_rows() {
                let language = language_array.value(i);
                *language_counts.entry(language.to_string()).or_insert(0) += 1;
            }
        }

        let mut language_breakdown: Vec<(String, usize)> =
            language_counts.into_iter().collect();
        language_breakdown.sort_by(|a, b| b.1.cmp(&a.1));

        Ok(DatabaseStats {
            total_points: count_result,
            total_vectors: count_result,
            language_breakdown,
        })
    }

    async fn flush(&self) -> Result<()> {
        // LanceDB persists automatically, no explicit flush needed
        Ok(())
    }
}
