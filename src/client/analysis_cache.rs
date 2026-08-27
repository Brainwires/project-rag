//! Generation-scoped caches for authoritative analysis responses.

use std::collections::HashMap;

use crate::types::{GetCallGraphResponse, QueryResponse};

#[derive(Clone)]
struct CachedGraph {
    generation: u64,
    response: GetCallGraphResponse,
}

#[derive(Clone)]
struct CachedQuery {
    generations: String,
    response: QueryResponse,
}

#[derive(Default)]
pub(crate) struct AnalysisCache {
    graphs: HashMap<String, CachedGraph>,
    queries: HashMap<String, CachedQuery>,
}

impl AnalysisCache {
    pub(crate) fn graph(&self, key: &str, generation: u64) -> Option<GetCallGraphResponse> {
        self.graphs
            .get(key)
            .filter(|entry| entry.generation == generation)
            .map(|entry| entry.response.clone())
    }

    pub(crate) fn insert_graph(
        &mut self,
        key: String,
        generation: u64,
        response: GetCallGraphResponse,
    ) {
        self.graphs.insert(
            key,
            CachedGraph {
                generation,
                response,
            },
        );
    }

    pub(crate) fn query(&self, key: &str, generations: &str) -> Option<QueryResponse> {
        self.queries
            .get(key)
            .filter(|entry| entry.generations == generations)
            .map(|entry| entry.response.clone())
    }

    pub(crate) fn insert_query(
        &mut self,
        key: String,
        generations: String,
        response: QueryResponse,
    ) {
        self.queries.insert(
            key,
            CachedQuery {
                generations,
                response,
            },
        );
    }

    pub(crate) fn clear(&mut self) {
        self.graphs.clear();
        self.queries.clear();
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::types::{GetCallGraphResponse, GraphTotals, QueryResponse};

    fn response() -> GetCallGraphResponse {
        GetCallGraphResponse {
            root_symbol: None,
            nodes: Vec::new(),
            edges: Vec::new(),
            requested_depth: 2,
            graph_truncated: false,
            returned_nodes: 0,
            returned_edges: 0,
            estimated_or_known_total: GraphTotals {
                nodes: 0,
                edges: 0,
                exact: true,
            },
            continuation: None,
            applied_edge_kinds: Vec::new(),
            applied_resolution_statuses: Vec::new(),
            index_generation: 7,
            cache_hit: false,
            precision: "syntactic".to_string(),
            duration_ms: 1,
        }
    }

    #[test]
    fn older_generation_is_never_returned_as_current() {
        let mut cache = AnalysisCache::default();
        cache.insert_graph("key".to_string(), 7, response());
        assert!(cache.graph("key", 7).is_some());
        assert!(cache.graph("key", 8).is_none());

        let query = QueryResponse {
            results: Vec::new(),
            duration_ms: 1,
            threshold_used: 0.7,
            threshold_lowered: false,
            total_matches: 0,
            returned_matches: 0,
            results_truncated: false,
            next_cursor: None,
        };
        cache.insert_query("query".to_string(), "root=7".to_string(), query);
        assert!(cache.query("query", "root=7").is_some());
        assert!(cache.query("query", "root=8").is_none());
    }

    #[test]
    #[ignore = "manual M5 generation-cache lookup measurement"]
    fn benchmark_m5_generation_cache_lookup() {
        let mut cache = AnalysisCache::default();
        cache.insert_graph("key".to_string(), 7, response());
        let started = std::time::Instant::now();
        for _ in 0..100_000 {
            assert!(cache.graph("key", 7).is_some());
        }
        println!(
            "m5 graph cache: lookups=100000 elapsed_us={}",
            started.elapsed().as_micros()
        );
    }

    #[test]
    #[ignore = "manual M5 generation-scoped query-cache lookup measurement"]
    fn benchmark_m5_generation_query_cache_lookup() {
        let mut cache = AnalysisCache::default();
        cache.insert_query(
            "query".to_string(),
            "root=7".to_string(),
            QueryResponse {
                results: Vec::new(),
                duration_ms: 1,
                threshold_used: 0.7,
                threshold_lowered: false,
                total_matches: 0,
                returned_matches: 0,
                results_truncated: false,
                next_cursor: None,
            },
        );
        let started = std::time::Instant::now();
        for _ in 0..100_000 {
            assert!(cache.query("query", "root=7").is_some());
        }
        println!(
            "m5 query cache: lookups=100000 elapsed_us={}",
            started.elapsed().as_micros()
        );
    }
}
