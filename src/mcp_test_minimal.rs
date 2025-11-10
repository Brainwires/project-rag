// Minimal test to isolate the issue
use rmcp::{
    handler::server::{ServerHandler, tool::ToolRouter, wrapper::Parameters},
    model::{Implementation, ProtocolVersion, ServerCapabilities, ServerInfo},
    service::ServiceExt,
    tool, tool_handler, tool_router,
};
use schemars::JsonSchema;
use serde::{Deserialize, Serialize};

#[derive(Debug, Serialize, Deserialize, JsonSchema)]
pub struct TestRequest {
    pub value: String,
}

#[derive(Clone)]
pub struct TestServer {
    tool_router: ToolRouter<Self>,
}

#[tool_router(router = tool_router)]
impl TestServer {
    pub fn new() -> Self {
        Self {
            tool_router: Self::tool_router(),
        }
    }

    #[tool(description = "Test tool")]
    async fn test_tool(
        &self,
        Parameters(req): Parameters<TestRequest>,
    ) -> Result<String, String> {
        Ok(req.value)
    }
}

#[tool_handler(router = self.tool_router)]
impl ServerHandler for TestServer {
    fn get_info(&self) -> ServerInfo {
        ServerInfo {
            protocol_version: ProtocolVersion::default(),
            capabilities: ServerCapabilities::builder().enable_tools().build(),
            server_info: Implementation {
                name: "test".into(),
                title: Some("Test".into()),
                version: "0.1.0".into(),
                icons: None,
                website_url: None,
            },
            instructions: None,
        }
    }
}
