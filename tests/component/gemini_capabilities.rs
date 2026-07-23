//! Gemini model_provider capabilities and contract tests.

use zeroclaw::providers::create_model_provider_with_url;
use zeroclaw::providers::traits::ModelProvider;

fn gemini_model_provider() -> Box<dyn ModelProvider> {
    create_model_provider_with_url("gemini", Some("test-key"), None)
        .expect("Gemini model_provider should resolve with test key")
}

// ─────────────────────────────────────────────────────────────────────────────
// Capabilities declaration
// ─────────────────────────────────────────────────────────────────────────────

#[test]
fn gemini_reports_native_tool_calling() {
    let model_provider = gemini_model_provider();
    let caps = model_provider.capabilities();
    assert!(
        caps.native_tool_calling,
        "Gemini should use native function calling"
    );
}

#[test]
fn gemini_reports_vision_support() {
    let model_provider = gemini_model_provider();
    let caps = model_provider.capabilities();
    assert!(caps.vision, "Gemini should report vision support");
}

#[test]
fn gemini_supports_native_tools_returns_true() {
    let model_provider = gemini_model_provider();
    assert!(
        model_provider.supports_native_tools(),
        "supports_native_tools() must be true so chat() sends functionDeclarations"
    );
}

#[test]
fn gemini_supports_vision_returns_true() {
    let model_provider = gemini_model_provider();
    assert!(model_provider.supports_vision());
}

// ─────────────────────────────────────────────────────────────────────────────
// Tool conversion contract
// ─────────────────────────────────────────────────────────────────────────────

#[test]
fn gemini_convert_tools_returns_function_declarations() {
    use zeroclaw::providers::traits::ToolsPayload;
    use zeroclaw::tools::ToolSpec;

    let model_provider = gemini_model_provider();
    let tools = vec![ToolSpec::new(
        "memory_store".to_string(),
        "Store a value in memory".to_string(),
        serde_json::json!({
            "type": "object",
            "properties": {
                "key": {"type": "string"},
                "value": {"type": "string"}
            },
            "required": ["key", "value"]
        }),
    )];

    let payload = model_provider.convert_tools(&tools);
    let ToolsPayload::Gemini {
        function_declarations,
    } = payload
    else {
        panic!("Gemini should return a native functionDeclarations payload");
    };

    assert_eq!(function_declarations.len(), 1);
    assert_eq!(function_declarations[0]["name"], "memory_store");
    assert_eq!(
        function_declarations[0]["description"],
        "Store a value in memory"
    );
    // `convert_tools` has no model to read a generation from, so it declares
    // through the OpenAPI-subset field every Gemini generation accepts.
    // `chat` picks `parametersJsonSchema` for generation 3 and newer.
    assert_eq!(
        function_declarations[0]["parameters"]["required"],
        serde_json::json!(["key", "value"])
    );
    assert!(
        function_declarations[0].get("parametersJsonSchema").is_none(),
        "the two parameter fields are mutually exclusive"
    );
}
