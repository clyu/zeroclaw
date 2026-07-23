//! Google Gemini model_provider with support for:
//! - Direct API key (`GEMINI_API_KEY` env var or config)
//! - Gemini CLI OAuth tokens (reuse existing ~/.gemini/ authentication)
//! - ZeroClaw auth-profiles OAuth tokens

use crate::auth::AuthService;
use crate::traits::{
    ChatMessage, ChatRequest as ProviderChatRequest, ChatResponse as ProviderChatResponse,
    ModelProvider, TokenUsage, ToolCall as ProviderToolCall, ToolsPayload,
};
use async_trait::async_trait;
use base64::Engine;
use directories::UserDirs;
use reqwest::{Client, header::HeaderValue};
use serde::{Deserialize, Serialize};
use std::path::PathBuf;
use std::sync::Arc;
use zeroclaw_api::schema::{CleaningStrategy, SchemaCleanr};
use zeroclaw_api::tool::ToolSpec;

const GOOGLE_API_KEY_HEADER: &str = "x-goog-api-key";

/// Gemini model_provider supporting multiple authentication methods.
pub struct GeminiModelProvider {
    /// `[providers.models.gemini.<alias>]` config-key alias.
    alias: String,
    auth: Option<GeminiAuth>,
    oauth_project: Arc<tokio::sync::Mutex<Option<String>>>,
    oauth_project_seed: Option<String>,
    oauth_cred_paths: Vec<PathBuf>,
    oauth_index: Arc<tokio::sync::Mutex<usize>>,
    /// AuthService for managed profiles (auth-profiles.json).
    auth_service: Option<AuthService>,
    /// Override profile name for managed auth.
    auth_profile_override: Option<String>,
    /// Per-alias OAuth app credentials carried at construction time so
    /// runtime token refreshes via `AuthService::get_valid_gemini_access_token`
    /// can mint new access tokens without dipping back into Config. Set
    /// from `GeminiModelProviderConfig.oauth_client_id` / `oauth_client_secret`.
    oauth_client_id: Option<String>,
    oauth_client_secret: Option<String>,
    /// Memoized cleaned tool schemas: a schema that needs rewriting is cleaned
    /// once per provider instance rather than rebuilt on every request. The
    /// memo only pays off while the instance lives — paths that rebuild the
    /// provider per call (e.g. the per-iteration vision route) start it empty
    /// each time.
    schema_cache: zeroclaw_api::schema::SchemaCleanCache,
    /// Tools whose drop set this instance has already walked and reported,
    /// keyed by `(name, parameter field)` — the two fields filter by different
    /// rules, so each has its own drop set to report. This field creates that
    /// fact; nothing else records what has been logged. Keyed by name, so a
    /// tool that re-registers with a different schema mid-life reports only
    /// its first — the right trade for a diagnostic that describes a static
    /// property of a schema.
    schema_drops_inspected: std::sync::Mutex<std::collections::HashSet<(String, ParametersField)>>,
}

/// Mutable OAuth token state — supports runtime refresh for long-lived processes.
struct OAuthTokenState {
    access_token: String,
    refresh_token: Option<String>,
    client_id: Option<String>,
    client_secret: Option<String>,
    /// Expiry as unix millis. `None` means unknown (treat as potentially expired).
    expiry_millis: Option<i64>,
}

/// Resolved credential — the variant determines both the HTTP auth method
/// and the diagnostic label returned by `auth_source()`.
enum GeminiAuth {
    /// Explicit API key from config: sent as `x-goog-api-key`.
    ExplicitKey(String),
    /// OAuth access token from Gemini CLI: sent as `Authorization: Bearer`.
    /// Wrapped in a Mutex to allow runtime token refresh.
    OAuthToken(Arc<tokio::sync::Mutex<OAuthTokenState>>),
    /// OAuth token managed by AuthService (auth-profiles.json).
    /// Token refresh is handled by AuthService, not here.
    ManagedOAuth,
}

impl GeminiAuth {
    /// Whether this credential is an OAuth token (CLI or managed).
    fn is_oauth(&self) -> bool {
        matches!(self, GeminiAuth::OAuthToken(_) | GeminiAuth::ManagedOAuth)
    }
}

// ══════════════════════════════════════════════════════════════════════════════
// API REQUEST/RESPONSE TYPES
// ══════════════════════════════════════════════════════════════════════════════

#[derive(Debug, Serialize, Clone)]
struct GenerateContentRequest {
    contents: Vec<Content>,
    #[serde(rename = "systemInstruction", skip_serializing_if = "Option::is_none")]
    system_instruction: Option<Content>,
    #[serde(skip_serializing_if = "Option::is_none")]
    tools: Option<Vec<ToolDeclarations>>,
    #[serde(rename = "generationConfig")]
    generation_config: GenerationConfig,
}

/// One entry of Gemini's `tools` array. Every ZeroClaw tool is a function, so
/// a single entry carrying all declarations is enough.
#[derive(Debug, Serialize, Clone)]
struct ToolDeclarations {
    #[serde(rename = "functionDeclarations")]
    function_declarations: Vec<serde_json::Value>,
}

#[derive(Debug, Serialize)]
struct InternalGenerateContentEnvelope {
    model: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    project: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    user_prompt_id: Option<String>,
    request: InternalGenerateContentRequest,
}

/// Nested request payload for cloudcode-pa's code assist APIs.
#[derive(Debug, Serialize)]
struct InternalGenerateContentRequest {
    contents: Vec<Content>,
    #[serde(rename = "systemInstruction", skip_serializing_if = "Option::is_none")]
    system_instruction: Option<Content>,
    #[serde(skip_serializing_if = "Option::is_none")]
    tools: Option<Vec<ToolDeclarations>>,
    #[serde(rename = "generationConfig", skip_serializing_if = "Option::is_none")]
    generation_config: Option<GenerationConfig>,
}

#[derive(Debug, Serialize, Clone)]
struct Content {
    #[serde(skip_serializing_if = "Option::is_none")]
    role: Option<String>,
    parts: Vec<Part>,
}

#[derive(Debug, Serialize, Clone)]
#[serde(untagged)]
enum Part {
    Text {
        text: String,
    },
    Inline {
        inline_data: InlineData,
    },
    /// A function call the model emitted on an earlier turn, replayed as part
    /// of the conversation history.
    FunctionCall {
        #[serde(rename = "functionCall")]
        function_call: FunctionCall,
        /// Gemini 3's opaque reasoning token. It arrives on the same part as
        /// the `functionCall` and must be echoed back there verbatim, or the
        /// API rejects the follow-up turn with `400 INVALID_ARGUMENT`.
        #[serde(rename = "thoughtSignature", skip_serializing_if = "Option::is_none")]
        thought_signature: Option<String>,
    },
    /// The result of executing a function call, fed back to the model.
    FunctionResponse {
        #[serde(rename = "functionResponse")]
        function_response: FunctionResponse,
    },
}

impl Part {
    fn text(s: impl Into<String>) -> Self {
        Part::Text { text: s.into() }
    }
}

#[derive(Debug, Serialize, Clone)]
struct FunctionCall {
    /// Only sent when Gemini itself assigned one; ZeroClaw-minted call ids are
    /// local bookkeeping and must not leak onto the wire.
    #[serde(skip_serializing_if = "Option::is_none")]
    id: Option<String>,
    name: String,
    args: serde_json::Value,
}

#[derive(Debug, Serialize, Clone)]
struct FunctionResponse {
    #[serde(skip_serializing_if = "Option::is_none")]
    id: Option<String>,
    name: String,
    response: serde_json::Value,
}

#[derive(Debug, Serialize, Clone)]
struct InlineData {
    mime_type: String,
    data: String,
}

/// Build Gemini Parts from a message content string.
/// If the content contains [IMAGE:data:...] markers (already normalized by the
/// multimodal pipeline), they are extracted as inline_data parts. The remaining
/// text becomes a text part. Falls back to a single text part if no markers.
fn build_parts(content: &str) -> Vec<Part> {
    let (text, image_refs) = crate::multimodal::parse_image_markers(content);
    let mut parts = Vec::new();
    let trimmed = text.trim();
    if !trimmed.is_empty() {
        parts.push(Part::text(trimmed));
    }
    parts.extend(inline_parts(&image_refs));
    if parts.is_empty() {
        parts.push(Part::text(content));
    }
    parts
}

/// Convert the `data:` URIs among image references into inline parts. Markers
/// pointing at anything else (file paths, remote URLs) carry no bytes Gemini
/// can read and are dropped.
fn inline_parts(image_refs: &[String]) -> Vec<Part> {
    image_refs
        .iter()
        .filter_map(|uri| {
            let rest = uri.strip_prefix("data:")?;
            let semi_pos = rest.find(';')?;
            let b64 = rest[semi_pos + 1..].strip_prefix("base64,")?;
            Some(Part::Inline {
                inline_data: InlineData {
                    mime_type: rest[..semi_pos].to_string(),
                    data: b64.to_string(),
                },
            })
        })
        .collect()
}

#[derive(Debug, Serialize, Clone)]
struct GenerationConfig {
    #[serde(skip_serializing_if = "Option::is_none")]
    temperature: Option<f64>,
    #[serde(rename = "maxOutputTokens")]
    max_output_tokens: u32,
}

#[derive(Debug, Deserialize)]
struct GenerateContentResponse {
    candidates: Option<Vec<Candidate>>,
    error: Option<ApiError>,
    #[serde(default)]
    response: Option<Box<GenerateContentResponse>>,
    #[serde(default, rename = "usageMetadata")]
    usage_metadata: Option<GeminiUsageMetadata>,
}

#[derive(Debug, Deserialize)]
struct GeminiUsageMetadata {
    #[serde(default, rename = "promptTokenCount")]
    prompt_token_count: Option<u64>,
    #[serde(default, rename = "candidatesTokenCount")]
    candidates_token_count: Option<u64>,
}

#[derive(Debug, Deserialize)]
struct Candidate {
    #[serde(default)]
    content: Option<CandidateContent>,
}

#[derive(Debug, Deserialize)]
struct CandidateContent {
    parts: Vec<ResponsePart>,
}

#[derive(Debug, Deserialize)]
struct ResponsePart {
    #[serde(default)]
    text: Option<String>,
    /// Thinking models (e.g. gemini-3-pro-preview) mark reasoning parts with `thought: true`.
    #[serde(default)]
    thought: bool,
    #[serde(default, rename = "functionCall")]
    function_call: Option<ResponseFunctionCall>,
    /// Gemini 3's opaque reasoning token, captured here so the next turn can
    /// hand it straight back on the part it arrived on.
    #[serde(default, rename = "thoughtSignature")]
    thought_signature: Option<String>,
}

#[derive(Debug, Deserialize)]
struct ResponseFunctionCall {
    #[serde(default)]
    id: Option<String>,
    #[serde(default)]
    name: Option<String>,
    #[serde(default)]
    args: Option<serde_json::Value>,
}

/// The answer text, reasoning, and function calls carried by one candidate's
/// parts.
#[derive(Default)]
struct CandidateParts {
    text: Option<String>,
    /// The model's thinking, when it is not already the answer. Gemini is the
    /// only native provider that computes this, so without carrying it here
    /// a tool-calling turn reaches history with no reasoning at all.
    reasoning: Option<String>,
    tool_calls: Vec<ProviderToolCall>,
}

impl CandidateParts {
    fn is_empty(&self) -> bool {
        self.text.is_none() && self.tool_calls.is_empty()
    }
}

impl CandidateContent {
    fn into_parts(self) -> CandidateParts {
        let mut answer_parts: Vec<String> = Vec::new();
        let mut first_thinking: Option<String> = None;
        let mut tool_calls: Vec<ProviderToolCall> = Vec::new();

        for part in self.parts {
            if let Some(call) = part.function_call {
                let Some(name) = call.name.filter(|name| !name.trim().is_empty()) else {
                    // Unnamed, so unpairable and uncallable. Dropping it
                    // silently leaves a candidate that may now hold nothing,
                    // which `send_generate_content` reports as an empty
                    // response — an operator would be told the API returned
                    // nothing when it returned a malformed call.
                    ::zeroclaw_log::record!(
                        WARN,
                        ::zeroclaw_log::Event::new(module_path!(), ::zeroclaw_log::Action::Note)
                            .with_category(::zeroclaw_log::EventCategory::Tool)
                            .with_outcome(::zeroclaw_log::EventOutcome::Unknown)
                            .with_attrs(::serde_json::json!({"id": call.id.as_deref()})),
                        "gemini: dropping a function call whose name is missing or blank"
                    );
                    continue;
                };
                let args = call
                    .args
                    .unwrap_or_else(|| serde_json::Value::Object(serde_json::Map::new()));
                tool_calls.push(ProviderToolCall {
                    // Gemini only assigns call ids on some surfaces; mint one
                    // locally otherwise so the agent loop can pair results.
                    id: call
                        .id
                        .clone()
                        .filter(|id| !id.is_empty())
                        .unwrap_or_else(|| format!("call_{}", uuid::Uuid::new_v4())),
                    name,
                    arguments: args.to_string(),
                    extra_content: google_extra_content(call.id, part.thought_signature),
                });
                continue;
            }
            if let Some(text) = part.text {
                if text.is_empty() {
                    continue;
                }
                if !part.thought {
                    answer_parts.push(text);
                } else if first_thinking.is_none() {
                    first_thinking = Some(text);
                }
            }
        }

        let (text, reasoning) = if !answer_parts.is_empty() {
            (Some(answer_parts.join("")), first_thinking)
        } else if tool_calls.is_empty() {
            // A thinking-only turn still carries the model's intent; surface
            // it as the answer rather than nothing. Reporting it as reasoning
            // as well would hand the same text to the caller twice.
            (first_thinking, None)
        } else {
            // The calls are the turn's substance, so the reasoning is not the
            // answer — but it is still the reasoning, and both history and the
            // ACP reasoning stream have a place for it.
            (None, first_thinking)
        };

        CandidateParts {
            text,
            reasoning,
            tool_calls,
        }
    }
}

/// Pack the Gemini-specific fields that must survive a round trip into
/// [`ProviderToolCall::extra_content`], under the documented `google` key.
fn google_extra_content(
    id: Option<String>,
    thought_signature: Option<String>,
) -> Option<serde_json::Value> {
    let mut google = serde_json::Map::new();
    if let Some(signature) = thought_signature.filter(|value| !value.is_empty()) {
        google.insert(
            "thought_signature".to_string(),
            serde_json::Value::String(signature),
        );
    }
    if let Some(id) = id.filter(|value| !value.is_empty()) {
        google.insert("id".to_string(), serde_json::Value::String(id));
    }
    if google.is_empty() {
        return None;
    }
    Some(serde_json::json!({ "google": serde_json::Value::Object(google) }))
}

/// Which `FunctionDeclaration` field carries a tool's arguments.
///
/// The two are mutually exclusive; a declaration may send only one.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
enum ParametersField {
    /// `parametersJsonSchema` — JSON Schema as written, keeping the `$ref`,
    /// `$defs`, and `additionalProperties` that [`Self::OpenApi`] has to
    /// strip. Gemini 3 and newer only.
    JsonSchema,
    /// `parameters` — the OpenAPI 3.0 subset every Gemini generation accepts.
    OpenApi,
}

impl ParametersField {
    /// The declaration key this field serializes under.
    fn wire_key(self) -> &'static str {
        match self {
            Self::JsonSchema => "parametersJsonSchema",
            Self::OpenApi => "parameters",
        }
    }

    /// The cleaning strategy whose output this field accepts.
    fn cleaning_strategy(self) -> CleaningStrategy {
        match self {
            Self::JsonSchema => CleaningStrategy::GeminiJsonSchema,
            Self::OpenApi => CleaningStrategy::Gemini,
        }
    }
}

/// The first Gemini generation whose `FunctionDeclaration` accepts
/// `parametersJsonSchema`.
const FIRST_JSON_SCHEMA_GENERATION: (u32, u32) = (3, 0);

/// Which parameter field a model's `functionDeclarations` may use.
///
/// `parametersJsonSchema` is a Gemini 3 field. Sending it to an earlier model
/// is rejected with `400 INVALID_ARGUMENT`, which fails every tool-enabled
/// turn — so those generations fall back to `parameters` rather than losing
/// native tool calling, which they do support. Nothing here gates *whether*
/// tools are native; every Gemini generation calls them natively.
///
/// A name this cannot read a generation from resolves to the modern field on
/// purpose. The generations that need the fallback are a closed set — no new
/// 1.x or 2.x model will ship — while unreadable names skew new
/// (`gemini-flash-latest`, whatever follows generation 3). An allowlist would
/// invert that and break on every future release.
fn parameters_field_for_model(model: &str) -> ParametersField {
    match gemini_generation(model) {
        Some(generation) if generation < FIRST_JSON_SCHEMA_GENERATION => ParametersField::OpenApi,
        _ => ParametersField::JsonSchema,
    }
}

/// The `(major, minor)` generation a Gemini model name carries, when it
/// carries one at all. `None` for an alias (`gemini-flash-latest`), a dated
/// experimental build (`gemini-exp-1206`), or a non-Gemini model served by
/// the same endpoint.
fn gemini_generation(model: &str) -> Option<(u32, u32)> {
    // Tolerate the fully qualified `models/gemini-…` form.
    let name = model
        .rsplit('/')
        .next()
        .unwrap_or(model)
        .to_ascii_lowercase();
    let version = name.strip_prefix("gemini-")?.split('-').next()?;
    let (major, minor) = version.split_once('.').unwrap_or((version, "0"));
    Some((major.parse().ok()?, minor.parse().ok()?))
}

/// Whether a cleaned parameter schema actually describes arguments.
///
/// Gemini rejects a parameter schema whose `properties` map is empty, so a
/// no-argument tool must declare no parameters at all. But a schema can
/// describe its arguments without a top-level `properties` map, and getting
/// that wrong is silent and permanent: the tool is declared as taking
/// nothing, so the model calls it with `{}` on every turn and the tool
/// rejects it on every turn.
///
/// Recognised instead of `properties`:
///
/// - `$ref`, `anyOf`, `oneOf` — structure to resolve or pick from rather
///   than declare inline.
/// - `additionalProperties` other than `false` — a free-form map, where the
///   keys are the caller's to choose. `false` is the opposite: it closes the
///   object, and on its own describes nothing.
/// - a non-empty `required` — an underspecified schema, but one that still
///   names arguments the tool will demand. `Schema` carries `required` as a
///   field in its own right, with no documented dependency on `properties`,
///   so sending it is no worse than the guaranteed failure of not.
///
/// Takes the *cleaned* schema: keywords the target strips (`$ref` and
/// `additionalProperties` on the OpenAPI-subset path) are already gone, and
/// a tool that expresses its arguments only that way genuinely cannot be
/// declared there.
fn declares_arguments(schema: &serde_json::Value) -> bool {
    let Some(object) = schema.as_object() else {
        return false;
    };
    let has_properties = object
        .get("properties")
        .and_then(serde_json::Value::as_object)
        .is_some_and(|properties| !properties.is_empty());
    let names_required_arguments = object
        .get("required")
        .and_then(serde_json::Value::as_array)
        .is_some_and(|required| !required.is_empty());
    let takes_free_form_entries = object
        .get("additionalProperties")
        .is_some_and(|value| !matches!(value, serde_json::Value::Bool(false)));

    has_properties
        || names_required_arguments
        || takes_free_form_entries
        || ["$ref", "anyOf", "oneOf"].iter().any(|key| object.contains_key(*key))
}

/// Read one field back out of the `google` namespace of a tool call's
/// `extra_content`.
fn google_extra(call: &ProviderToolCall, key: &str) -> Option<String> {
    call.extra_content
        .as_ref()?
        .get("google")?
        .get(key)?
        .as_str()
        .filter(|value| !value.is_empty())
        .map(ToString::to_string)
}

#[derive(Debug, Deserialize)]
struct ApiError {
    message: String,
}

impl GenerateContentResponse {
    /// cloudcode-pa wraps the actual response under `response`.
    fn into_effective_response(self) -> Self {
        match self {
            Self {
                response: Some(mut inner),
                usage_metadata,
                ..
            } => {
                if inner.usage_metadata.is_none() {
                    inner.usage_metadata = usage_metadata;
                }
                *inner
            }
            other => other,
        }
    }
}

// ══════════════════════════════════════════════════════════════════════════════
// GEMINI CLI TOKEN STRUCTURES
// ══════════════════════════════════════════════════════════════════════════════

/// OAuth token stored by Gemini CLI in `~/.gemini/oauth_creds.json`
#[derive(Debug, Deserialize)]
struct GeminiCliOAuthCreds {
    access_token: Option<String>,
    #[serde(alias = "idToken")]
    id_token: Option<String>,
    refresh_token: Option<String>,
    #[serde(alias = "clientId")]
    client_id: Option<String>,
    #[serde(alias = "clientSecret")]
    client_secret: Option<String>,
    /// Unix milliseconds expiry (used by newer Gemini CLI versions).
    #[serde(alias = "expiryDate")]
    expiry_date: Option<i64>,
    /// RFC 3339 expiry string (used by older Gemini CLI versions).
    expiry: Option<String>,
}

// ══════════════════════════════════════════════════════════════════════════════
// GEMINI CLI OAUTH CONSTANTS
// ══════════════════════════════════════════════════════════════════════════════

/// Google OAuth token endpoint.
const GOOGLE_TOKEN_ENDPOINT: &str = "https://oauth2.googleapis.com/token";

/// Internal API endpoint used by Gemini CLI for OAuth users.
/// See: <https://github.com/google-gemini/gemini-cli/issues/19200>
const CLOUDCODE_PA_ENDPOINT: &str = "https://cloudcode-pa.googleapis.com/v1internal";

/// loadCodeAssist endpoint for resolving the project ID.
const LOAD_CODE_ASSIST_ENDPOINT: &str =
    "https://cloudcode-pa.googleapis.com/v1internal:loadCodeAssist";

/// A `loadCodeAssist` project reference: a bare id string, or an object
/// carrying `id` / `projectId`. Google returns either shape depending on
/// whether the account already has an onboarded project.
#[derive(Deserialize)]
#[serde(untagged)]
enum ProjectRef {
    Id(String),
    Object {
        #[serde(alias = "projectId")]
        id: Option<String>,
    },
}

impl ProjectRef {
    fn into_id(self) -> Option<String> {
        match self {
            Self::Id(id) => Some(id),
            Self::Object { id } => id,
        }
        .filter(|p| !p.trim().is_empty())
    }
}

#[derive(Deserialize)]
struct LoadCodeAssistResponse {
    #[serde(rename = "cloudaicompanionProject")]
    cloudaicompanion_project: Option<ProjectRef>,
    #[serde(rename = "currentCloudaicompanionProject")]
    current_cloudaicompanion_project: Option<ProjectRef>,
}

impl LoadCodeAssistResponse {
    fn resolve_project_id(self) -> Option<String> {
        self.cloudaicompanion_project
            .and_then(ProjectRef::into_id)
            .or_else(|| {
                self.current_cloudaicompanion_project
                    .and_then(ProjectRef::into_id)
            })
    }
}

/// Google AI Studio's Gemini endpoint.
pub(crate) const BASE_URL: &str = "https://generativelanguage.googleapis.com/v1beta";

// ══════════════════════════════════════════════════════════════════════════════
// TOKEN REFRESH
// ══════════════════════════════════════════════════════════════════════════════

/// Result of a successful token refresh.
struct RefreshedToken {
    access_token: String,
    /// Expiry as unix millis (computed from `expires_in` seconds in the response).
    expiry_millis: Option<i64>,
}

fn refresh_gemini_cli_token(
    refresh_token: &str,
    client_id: Option<&str>,
    client_secret: Option<&str>,
) -> anyhow::Result<RefreshedToken> {
    let client = reqwest::blocking::Client::builder()
        .timeout(std::time::Duration::from_secs(15))
        .connect_timeout(std::time::Duration::from_secs(5))
        .build()
        .unwrap_or_else(|_| reqwest::blocking::Client::new());

    let form = build_oauth_refresh_form(refresh_token, client_id, client_secret);

    let response = client
        .post(GOOGLE_TOKEN_ENDPOINT)
        .header("Content-Type", "application/x-www-form-urlencoded")
        .header("Accept", "application/json")
        .form(&form)
        .send()
        .map_err(|error| {
            ::zeroclaw_log::record!(
                ERROR,
                ::zeroclaw_log::Event::new(module_path!(), ::zeroclaw_log::Action::Fail)
                    .with_outcome(::zeroclaw_log::EventOutcome::Failure)
                    .with_attrs(::serde_json::json!({
                        "oauth_provider": "gemini_cli",
                        "phase": "refresh_request",
                        "error": format!("{}", error),
                    })),
                "gemini: CLI OAuth refresh request failed"
            );
            anyhow::Error::msg(format!("Gemini CLI OAuth refresh request failed: {error}"))
        })?;

    let status = response.status();
    let body = response
        .text()
        .unwrap_or_else(|_| "<failed to read response body>".to_string());

    if !status.is_success() {
        anyhow::bail!("Gemini CLI OAuth refresh failed (HTTP {status}): {body}");
    }

    #[derive(Deserialize)]
    struct TokenResponse {
        access_token: Option<String>,
        expires_in: Option<i64>,
    }

    let parsed: TokenResponse = serde_json::from_str(&body).map_err(|_| {
        ::zeroclaw_log::record!(
            ERROR,
            ::zeroclaw_log::Event::new(module_path!(), ::zeroclaw_log::Action::Fail)
                .with_outcome(::zeroclaw_log::EventOutcome::Failure)
                .with_attrs(::serde_json::json!({"oauth_provider": "gemini_cli"})),
            "gemini: CLI OAuth refresh response is not valid JSON"
        );
        anyhow::Error::msg("Gemini CLI OAuth refresh response is not valid JSON")
    })?;

    let access_token = parsed
        .access_token
        .filter(|t| !t.trim().is_empty())
        .ok_or_else(|| {
            ::zeroclaw_log::record!(
                ERROR,
                ::zeroclaw_log::Event::new(module_path!(), ::zeroclaw_log::Action::Reject)
                    .with_outcome(::zeroclaw_log::EventOutcome::Failure)
                    .with_attrs(::serde_json::json!({
                        "oauth_provider": "gemini_cli",
                        "missing": "access_token",
                    })),
                "gemini: CLI OAuth refresh missing access_token"
            );
            anyhow::Error::msg("Gemini CLI OAuth refresh response missing access_token")
        })?;

    let expiry_millis = parsed.expires_in.and_then(|secs| {
        let now_millis = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .ok()
            .and_then(|d| i64::try_from(d.as_millis()).ok())?;
        now_millis.checked_add(secs.checked_mul(1000)?)
    });

    Ok(RefreshedToken {
        access_token,
        expiry_millis,
    })
}

fn build_oauth_refresh_form(
    refresh_token: &str,
    client_id: Option<&str>,
    client_secret: Option<&str>,
) -> Vec<(&'static str, String)> {
    let mut form = vec![
        ("grant_type", "refresh_token".to_string()),
        ("refresh_token", refresh_token.to_string()),
    ];
    if let Some(id) = client_id.and_then(GeminiModelProvider::normalize_non_empty) {
        form.push(("client_id", id));
    }
    if let Some(secret) = client_secret.and_then(GeminiModelProvider::normalize_non_empty) {
        form.push(("client_secret", secret));
    }
    form
}

fn extract_client_id_from_id_token(id_token: &str) -> Option<String> {
    let payload = id_token.split('.').nth(1)?;
    let decoded = base64::engine::general_purpose::URL_SAFE_NO_PAD
        .decode(payload)
        .or_else(|_| base64::engine::general_purpose::URL_SAFE.decode(payload))
        .ok()?;

    #[derive(Deserialize)]
    struct IdTokenClaims {
        aud: Option<String>,
        azp: Option<String>,
    }

    let claims: IdTokenClaims = serde_json::from_slice(&decoded).ok()?;
    claims
        .aud
        .as_deref()
        .and_then(GeminiModelProvider::normalize_non_empty)
        .or_else(|| {
            claims
                .azp
                .as_deref()
                .and_then(GeminiModelProvider::normalize_non_empty)
        })
}

/// Async version of token refresh for use during runtime (inside tokio context).
async fn refresh_gemini_cli_token_async(
    refresh_token: &str,
    client_id: Option<&str>,
    client_secret: Option<&str>,
) -> anyhow::Result<RefreshedToken> {
    let refresh_token = refresh_token.to_string();
    let client_id = client_id.map(str::to_string);
    let client_secret = client_secret.map(str::to_string);
    tokio::task::spawn_blocking(move || {
        refresh_gemini_cli_token(
            &refresh_token,
            client_id.as_deref(),
            client_secret.as_deref(),
        )
    })
    .await
    .map_err(|e| {
        ::zeroclaw_log::record!(
            ERROR,
            ::zeroclaw_log::Event::new(module_path!(), ::zeroclaw_log::Action::Fail)
                .with_outcome(::zeroclaw_log::EventOutcome::Failure)
                .with_attrs(::serde_json::json!({
                    "oauth_provider": "gemini_cli",
                    "phase": "task_join",
                    "error": format!("{}", e),
                })),
            "gemini: token refresh task panicked"
        );
        anyhow::Error::msg(format!("Token refresh task panicked: {e}"))
    })?
}

/// Typed builder for [`GeminiModelProvider`].
///
/// Only `alias` is required; every credential input is optional and layered
/// at [`Self::build`] time in this order:
///   1. Explicit API key from [`Self::api_key`]
///   2. Managed OAuth via [`Self::managed_auth`] (when wired and the
///      profile exists)
///   3. CLI OAuth tokens from `~/.gemini/oauth_creds.json`
#[must_use]
pub struct GeminiBuilder {
    alias: String,
    api_key: Option<String>,
    auth_service: Option<AuthService>,
    profile_override: Option<String>,
    oauth_project_seed: Option<String>,
    oauth_client_id: Option<String>,
    oauth_client_secret: Option<String>,
}

impl GeminiBuilder {
    /// Explicit API key (from `[providers.models.gemini.<alias>] api_key`).
    /// When set (and non-empty after trimming), takes precedence over every
    /// OAuth path.
    pub fn api_key(mut self, key: Option<&str>) -> Self {
        self.api_key = key.map(str::to_string);
        self
    }

    /// Wire up managed OAuth via the shared [`AuthService`], with an
    /// optional profile override. When set and no explicit API key is
    /// provided, the builder probes the service at build time; if a
    /// managed profile exists, the resulting provider uses managed OAuth
    /// instead of falling through to the CLI creds.
    pub fn managed_auth(
        mut self,
        auth_service: AuthService,
        profile_override: Option<String>,
    ) -> Self {
        self.auth_service = Some(auth_service);
        self.profile_override = profile_override;
        self
    }

    /// Seed value for the OAuth project resolution cache.
    pub fn oauth_project_seed(mut self, seed: Option<String>) -> Self {
        self.oauth_project_seed = seed;
        self
    }

    /// Override the OAuth client credentials (defaults to the Gemini CLI
    /// public client when unset).
    pub fn oauth_client(
        mut self,
        client_id: Option<String>,
        client_secret: Option<String>,
    ) -> Self {
        self.oauth_client_id = client_id;
        self.oauth_client_secret = client_secret;
        self
    }

    /// Resolve the credential layers and finalize the provider.
    pub fn build(self) -> GeminiModelProvider {
        let oauth_cred_paths = GeminiModelProvider::discover_oauth_cred_paths();

        // Layer 1 — explicit API key.
        let explicit_key = self
            .api_key
            .as_deref()
            .and_then(GeminiModelProvider::normalize_non_empty)
            .map(GeminiAuth::ExplicitKey);

        // Layer 3 — CLI OAuth fallback. Used both when no managed service is
        // wired at all, and when a managed service was wired but no profile
        // was found on disk.
        let load_cli_oauth = || {
            GeminiModelProvider::try_load_gemini_cli_token(oauth_cred_paths.first())
                .map(|state| GeminiAuth::OAuthToken(Arc::new(tokio::sync::Mutex::new(state))))
        };

        // Layer 2 — managed OAuth (only probed when an AuthService is wired
        // and no explicit key beat it).
        let (auth, use_managed) = match (explicit_key, self.auth_service.as_ref()) {
            (Some(a), _) => (Some(a), false),
            (None, Some(service)) => {
                let profile = self.profile_override.clone();
                let has_managed = std::thread::scope(|s| {
                    let service = service.clone();
                    s.spawn(move || {
                        let rt = tokio::runtime::Builder::new_current_thread()
                            .enable_all()
                            .build()
                            .ok()?;
                        rt.block_on(async {
                            service
                                .get_gemini_profile(profile.as_deref())
                                .await
                                .ok()
                                .flatten()
                        })
                    })
                    .join()
                    .ok()
                    .flatten()
                    .is_some()
                });
                if has_managed {
                    (Some(GeminiAuth::ManagedOAuth), true)
                } else {
                    (load_cli_oauth(), false)
                }
            }
            (None, None) => (load_cli_oauth(), false),
        };

        GeminiModelProvider {
            alias: self.alias,
            auth,
            oauth_project: Arc::new(tokio::sync::Mutex::new(None)),
            oauth_project_seed: self.oauth_project_seed,
            oauth_cred_paths,
            oauth_index: Arc::new(tokio::sync::Mutex::new(0)),
            auth_service: if use_managed { self.auth_service } else { None },
            auth_profile_override: self.profile_override,
            oauth_client_id: self.oauth_client_id,
            oauth_client_secret: self.oauth_client_secret,
            schema_cache: zeroclaw_api::schema::SchemaCleanCache::new(),
            schema_drops_inspected: std::sync::Mutex::new(std::collections::HashSet::new()),
        }
    }
}

impl GeminiModelProvider {
    /// Entry point for constructing a Gemini provider. Only `alias` is
    /// required; layer optional credential inputs onto the returned
    /// [`GeminiBuilder`].
    ///
    /// Authentication priority (evaluated at build time):
    /// 1. Explicit API key ([`GeminiBuilder::api_key`], from
    ///    `[providers.models.gemini.<alias>] api_key`)
    /// 2. Managed OAuth via [`GeminiBuilder::managed_auth`], when wired
    ///    and a profile exists
    /// 3. Gemini CLI OAuth tokens (`~/.gemini/oauth_creds.json`)
    pub fn builder(alias: &str) -> GeminiBuilder {
        GeminiBuilder {
            alias: alias.to_string(),
            api_key: None,
            auth_service: None,
            profile_override: None,
            oauth_project_seed: None,
            oauth_client_id: None,
            oauth_client_secret: None,
        }
    }

    fn normalize_non_empty(value: &str) -> Option<String> {
        let trimmed = value.trim();
        if trimmed.is_empty() {
            None
        } else {
            Some(trimmed.to_string())
        }
    }

    fn load_gemini_cli_creds(creds_path: &PathBuf) -> Option<GeminiCliOAuthCreds> {
        if !creds_path.exists() {
            return None;
        }
        let content = std::fs::read_to_string(creds_path).ok()?;
        serde_json::from_str(&content).ok()
    }

    fn discover_oauth_cred_paths() -> Vec<PathBuf> {
        let home = match UserDirs::new() {
            Some(u) => u.home_dir().to_path_buf(),
            None => return Vec::new(),
        };

        let mut paths = Vec::new();

        let primary = home.join(".gemini").join("oauth_creds.json");
        if primary.exists() {
            paths.push(primary);
        }

        if let Ok(entries) = std::fs::read_dir(&home) {
            let mut extras: Vec<PathBuf> = entries
                .filter_map(|e| e.ok())
                .filter_map(|e| {
                    let name = e.file_name().to_string_lossy().to_string();
                    if name.starts_with(".gemini-") && name.ends_with("-home") {
                        let path = e.path().join(".gemini").join("oauth_creds.json");
                        if path.exists() {
                            return Some(path);
                        }
                    }
                    None
                })
                .collect();
            extras.sort();
            paths.extend(extras);
        }

        paths
    }

    /// Try to load OAuth credentials from Gemini CLI's cached credentials.
    /// Location: `~/.gemini/oauth_creds.json`
    /// Returns the full `OAuthTokenState` so the model_provider can refresh at runtime.
    fn try_load_gemini_cli_token(path: Option<&PathBuf>) -> Option<OAuthTokenState> {
        let creds = Self::load_gemini_cli_creds(path?)?;

        // Determine expiry in millis: prefer expiry_date over expiry (RFC 3339)
        let expiry_millis = creds.expiry_date.or_else(|| {
            creds.expiry.as_deref().and_then(|expiry| {
                chrono::DateTime::parse_from_rfc3339(expiry)
                    .ok()
                    .map(|dt| dt.timestamp_millis())
            })
        });

        let access_token = creds
            .access_token
            .and_then(|token| Self::normalize_non_empty(&token))?;

        let id_token_client_id = creds
            .id_token
            .as_deref()
            .and_then(extract_client_id_from_id_token);

        let client_id = creds
            .client_id
            .as_deref()
            .and_then(Self::normalize_non_empty)
            .or(id_token_client_id);
        let client_secret = creds
            .client_secret
            .as_deref()
            .and_then(Self::normalize_non_empty);

        Some(OAuthTokenState {
            access_token,
            refresh_token: creds.refresh_token,
            client_id,
            client_secret,
            expiry_millis,
        })
    }

    /// Check if Gemini CLI is configured and has valid credentials
    pub fn has_cli_credentials() -> bool {
        Self::discover_oauth_cred_paths().iter().any(|path| {
            Self::load_gemini_cli_creds(path)
                .and_then(|creds| {
                    creds
                        .access_token
                        .as_deref()
                        .and_then(Self::normalize_non_empty)
                })
                .is_some()
        })
    }

    /// Check if any Gemini authentication is available via the Gemini CLI
    /// OAuth credential cache. Per-alias config-supplied keys are tracked
    /// separately on the constructed provider, so this helper no longer
    /// reads process env.
    pub fn has_any_auth() -> bool {
        Self::has_cli_credentials()
    }

    /// Get authentication source description for diagnostics.
    /// Uses the stored enum variant — no env var re-reading at call time.
    pub fn auth_source(&self) -> &'static str {
        match self.auth.as_ref() {
            Some(GeminiAuth::ExplicitKey(_)) => "config",
            Some(GeminiAuth::OAuthToken(_)) => "Gemini CLI OAuth",
            Some(GeminiAuth::ManagedOAuth) => "auth-profiles",
            None => "none",
        }
    }

    /// Get a valid OAuth access token, refreshing if expired.
    /// Adds a 60-second buffer before actual expiry to avoid edge-case failures.
    async fn get_valid_oauth_token(
        state: &Arc<tokio::sync::Mutex<OAuthTokenState>>,
    ) -> anyhow::Result<String> {
        let mut guard = state.lock().await;

        let now_millis = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .ok()
            .and_then(|d| i64::try_from(d.as_millis()).ok())
            .unwrap_or(i64::MAX);

        // Refresh if expiry is unknown, already expired, or within 60s of expiry.
        let needs_refresh = guard
            .expiry_millis
            .is_none_or(|exp| exp <= now_millis.saturating_add(60_000));

        if needs_refresh {
            if let Some(ref refresh_token) = guard.refresh_token {
                let refreshed = refresh_gemini_cli_token_async(
                    refresh_token,
                    guard.client_id.as_deref(),
                    guard.client_secret.as_deref(),
                )
                .await?;
                ::zeroclaw_log::record!(
                    INFO,
                    ::zeroclaw_log::Event::new(module_path!(), ::zeroclaw_log::Action::Note),
                    "Gemini CLI OAuth token refreshed successfully (runtime)"
                );
                guard.access_token = refreshed.access_token;
                guard.expiry_millis = refreshed.expiry_millis;
            } else {
                anyhow::bail!(
                    "Gemini CLI OAuth token expired and no refresh_token available — re-run `gemini` to authenticate"
                );
            }
        }

        Ok(guard.access_token.clone())
    }

    /// Rotate to the next available OAuth credentials file and swap state.
    /// Returns `true` when rotation succeeded.
    async fn rotate_oauth_credential(
        &self,
        state: &Arc<tokio::sync::Mutex<OAuthTokenState>>,
    ) -> bool {
        if self.oauth_cred_paths.len() <= 1 {
            return false;
        }

        let mut idx = self.oauth_index.lock().await;
        let start = *idx;

        loop {
            let next = (*idx + 1) % self.oauth_cred_paths.len();
            *idx = next;

            if next == start {
                return false;
            }

            if let Some(next_state) =
                Self::try_load_gemini_cli_token(self.oauth_cred_paths.get(next))
            {
                {
                    let mut guard = state.lock().await;
                    *guard = next_state;
                }
                {
                    let mut cached_project = self.oauth_project.lock().await;
                    *cached_project = None;
                }
                ::zeroclaw_log::record!(
                    WARN,
                    ::zeroclaw_log::Event::new(module_path!(), ::zeroclaw_log::Action::Note)
                        .with_outcome(::zeroclaw_log::EventOutcome::Unknown),
                    &format!(
                        "Gemini OAuth: rotated credential to {}",
                        self.oauth_cred_paths[next].display().to_string()
                    )
                );
                return true;
            }
        }
    }

    fn format_model_name(model: &str) -> String {
        if model.starts_with("models/") {
            model.to_string()
        } else {
            format!("models/{model}")
        }
    }

    fn format_internal_model_name(model: &str) -> String {
        model.strip_prefix("models/").unwrap_or(model).to_string()
    }

    fn build_generate_content_url(model: &str, auth: &GeminiAuth) -> String {
        match auth {
            GeminiAuth::OAuthToken(_) | GeminiAuth::ManagedOAuth => {
                // OAuth tokens are scoped for the internal Code Assist API.
                // The model is passed in the request body, not the URL path.
                format!("{CLOUDCODE_PA_ENDPOINT}:generateContent")
            }
            _ => {
                let model_name = Self::format_model_name(model);
                format!("{BASE_URL}/{model_name}:generateContent")
            }
        }
    }

    fn http_client(&self) -> Client {
        zeroclaw_config::schema::build_runtime_proxy_client_with_timeouts(
            "model_provider.gemini",
            120,
            10,
        )
    }

    /// Resolve the GCP project ID for OAuth by calling the loadCodeAssist endpoint.
    /// Caches the result for subsequent calls.
    async fn resolve_oauth_project(&self, token: &str) -> anyhow::Result<String> {
        let project_seed = self.oauth_project_seed.clone();
        let project_seed_for_request = project_seed.clone();
        let duet_project_for_request = project_seed.clone();

        // Check cache first
        {
            let cached = self.oauth_project.lock().await;
            if let Some(ref project) = *cached {
                return Ok(project.clone());
            }
        }

        // Call loadCodeAssist
        let client = self.http_client();
        let response = client
            .post(LOAD_CODE_ASSIST_ENDPOINT)
            .bearer_auth(token)
            .json(&serde_json::json!({
                "cloudaicompanionProject": project_seed_for_request,
                "metadata": {
                    "ideType": "GEMINI_CLI",
                    "platform": "PLATFORM_UNSPECIFIED",
                    "pluginType": "GEMINI",
                    "duetProject": duet_project_for_request,
                }
            }))
            .send()
            .await?;

        if !response.status().is_success() {
            let status = response.status();
            let body = response.text().await.unwrap_or_default();
            if let Some(seed) = project_seed {
                ::zeroclaw_log::record!(
                    WARN,
                    ::zeroclaw_log::Event::new(module_path!(), ::zeroclaw_log::Action::Note)
                        .with_outcome(::zeroclaw_log::EventOutcome::Unknown)
                        .with_attrs(::serde_json::json!({"status": status.to_string()})),
                    "loadCodeAssist failed (HTTP ); using oauth_project seed fallback"
                );
                return Ok(seed);
            }
            anyhow::bail!("loadCodeAssist failed (HTTP {status}): {body}");
        }

        let result: LoadCodeAssistResponse = response.json().await?;
        let project = result
            .resolve_project_id()
            .or(project_seed)
            .ok_or_else(|| {
                ::zeroclaw_log::record!(
                    ERROR,
                    ::zeroclaw_log::Event::new(module_path!(), ::zeroclaw_log::Action::Fail)
                        .with_outcome(::zeroclaw_log::EventOutcome::Failure)
                        .with_attrs(::serde_json::json!({
                            "missing": "cloudaicompanionProject",
                        })),
                    "gemini: loadCodeAssist missing project context"
                );
                anyhow::Error::msg("loadCodeAssist response missing project context")
            })?;

        // Cache for future calls
        {
            let mut cached = self.oauth_project.lock().await;
            *cached = Some(project.clone());
        }

        Ok(project)
    }

    /// Build the HTTP request for generateContent.
    /// For OAuth, pass the resolved `oauth_token` and `project`.
    /// For API key, both are `None`.
    fn build_generate_content_request(
        &self,
        auth: &GeminiAuth,
        url: &str,
        request: &GenerateContentRequest,
        model: &str,
        include_generation_config: bool,
        project: Option<&str>,
        oauth_token: Option<&str>,
    ) -> anyhow::Result<reqwest::RequestBuilder> {
        let req = self.http_client().post(url).json(request);
        Ok(match auth {
            GeminiAuth::ExplicitKey(key) => {
                req.header(GOOGLE_API_KEY_HEADER, Self::sensitive_api_key_header(key)?)
            }
            GeminiAuth::OAuthToken(_) | GeminiAuth::ManagedOAuth => {
                let token = oauth_token.unwrap_or_default();
                // Internal Code Assist API uses a wrapped payload shape:
                // { model, project?, user_prompt_id?, request: { contents, systemInstruction?, generationConfig } }
                let internal_request = InternalGenerateContentEnvelope {
                    model: Self::format_internal_model_name(model),
                    project: project.map(|value| value.to_string()),
                    user_prompt_id: Some(uuid::Uuid::new_v4().to_string()),
                    request: InternalGenerateContentRequest {
                        contents: request.contents.clone(),
                        system_instruction: request.system_instruction.clone(),
                        tools: request.tools.clone(),
                        generation_config: if include_generation_config {
                            Some(request.generation_config.clone())
                        } else {
                            None
                        },
                    },
                };
                self.http_client()
                    .post(url)
                    .json(&internal_request)
                    .bearer_auth(token)
            }
        })
    }

    fn sensitive_api_key_header(key: &str) -> anyhow::Result<HeaderValue> {
        let mut value = HeaderValue::from_str(key)
            .map_err(|_| anyhow::Error::msg("Gemini API key contains invalid header characters"))?;
        value.set_sensitive(true);
        Ok(value)
    }

    fn build_api_key_probe_request(&self, key: &str) -> anyhow::Result<reqwest::RequestBuilder> {
        Ok(self
            .http_client()
            .get(format!("{BASE_URL}/models"))
            .header(GOOGLE_API_KEY_HEADER, Self::sensitive_api_key_header(key)?))
    }

    fn should_retry_oauth_without_generation_config(
        status: reqwest::StatusCode,
        error_text: &str,
    ) -> bool {
        if status != reqwest::StatusCode::BAD_REQUEST {
            return false;
        }

        error_text.contains("Unknown name \"generationConfig\"")
            || error_text.contains("Unknown name 'generationConfig'")
            || error_text.contains(r#"Unknown name \"generationConfig\""#)
    }

    fn should_rotate_oauth_on_error(status: reqwest::StatusCode, error_text: &str) -> bool {
        status == reqwest::StatusCode::TOO_MANY_REQUESTS
            || status == reqwest::StatusCode::SERVICE_UNAVAILABLE
            || status.is_server_error()
            || error_text.contains("RESOURCE_EXHAUSTED")
    }
}

/// Marker for a tool result that has to travel as plain user text because no
/// declared function can be named for it.
///
/// Deliberately *not* the prompt-guided dispatcher's `[Tool results]` header,
/// for the reason spelled out on [`undeclared_call_text`]: this provider now
/// calls tools natively on every request, so that header is one half of a
/// protocol the request never declares — read on the same turn where the
/// tools that *are* declared answer as `functionResponse` parts.
const TOOL_RESULT_PREFIX: &str = "(tool output)";

/// The function names one request declares.
///
/// Gemini validates every `functionCall` and `functionResponse` in `contents`
/// against this set and rejects the whole turn with `400 INVALID_ARGUMENT`
/// when a name is missing from it. History outlives a tool declaration —
/// an MCP server that does not come back up, a tool dropped from config, a
/// restored ACP session — and the rejection would then repeat on every later
/// turn, so those parts have to degrade to text instead.
///
/// Read back off the wire payload rather than the [`ToolSpec`]s it was built
/// from, so the set stays exactly what the request declares even if
/// [`GeminiModelProvider::function_declaration`] ever rewrites a name.
fn declared_function_names(
    tools: Option<&[ToolDeclarations]>,
) -> std::collections::HashSet<String> {
    tools
        .into_iter()
        .flatten()
        .flat_map(|group| group.function_declarations.iter())
        .filter_map(|declaration| declaration.get("name")?.as_str().map(ToString::to_string))
        .collect()
}

/// Narrate a function call this request cannot declare.
///
/// Deliberately *not* the prompt-guided dispatcher's `<tool_call>` shape.
/// This text lands in a **model** turn while native function calling is
/// active, where that shape is a protocol the request never declared:
/// `NativeToolDispatcher::prompt_instructions` is empty in native mode, so
/// the model was told nothing about `<tool_call>` — but a model that reads it
/// in its own prior turn imitates it on the next one. The native dispatcher
/// parses only `ChatResponse::tool_calls`, so the imitation never executes,
/// and the orchestrator's leaked-envelope guard then suppresses the reply
/// outright: the turn reaches the user as nothing at all.
///
/// Plain prose records the same fact — this call happened, that tool is gone
/// — without teaching a form the model can call back.
fn undeclared_call_text(name: &str, args: &serde_json::Value) -> String {
    format!(
        "(earlier turn called the tool `{name}` with arguments {args}; \
         that tool is no longer available)"
    )
}

/// Emit the user turns that trail a batch of tool results: first the outputs
/// that had to degrade to text, then the images no `functionResponse` can
/// carry.
///
/// Both are deferred to the end of the batch so the `functionResponse` parts
/// stay one contiguous turn answering the model's call turn. A batch where
/// only some tools are still declared would otherwise interleave text between
/// them and split that answer across three turns.
fn flush_pending_tool_turns(
    contents: &mut Vec<Content>,
    pending_text: &mut Vec<String>,
    pending_images: &mut Vec<Part>,
) {
    if !pending_text.is_empty() {
        // One prefixed turn for the whole batch rather than one per result,
        // so the degraded outputs read as a single block.
        let joined = std::mem::take(pending_text).join("\n");
        contents.push(Content {
            role: Some("user".to_string()),
            parts: vec![Part::text(format!("{TOOL_RESULT_PREFIX}\n{joined}"))],
        });
    }
    if !pending_images.is_empty() {
        contents.push(Content {
            role: Some("user".to_string()),
            parts: std::mem::take(pending_images),
        });
    }
}

/// The call a tool result answers. Gemini pairs a `functionResponse` with its
/// `functionCall` by name, so the name has to be recovered from the assistant
/// turn that requested it.
struct ToolCallRef {
    name: String,
    /// The id Gemini assigned to the call, when it assigned one.
    api_id: Option<String>,
}

/// The id of the call a tool result answers: its own when the transcript
/// still carries one that matches, otherwise — and only while exactly one
/// call is outstanding — the call left by elimination.
///
/// With two outstanding there is nothing to pick from. Naming either would
/// tell the model the wrong tool produced this output *and* still leave the
/// right call unanswered, so a guess buys no pairing and corrupts what the
/// model reads; the caller degrades the result to text instead.
fn answered_call_id(
    tool_call_id: Option<&str>,
    calls_by_id: &std::collections::HashMap<String, ToolCallRef>,
    unanswered_calls: &[String],
) -> Option<String> {
    if let Some(id) = tool_call_id.filter(|id| calls_by_id.contains_key(*id)) {
        if unanswered_calls.iter().any(|outstanding| outstanding.as_str() == id) {
            return Some(id.to_string());
        }
        // A known id whose call already has its answer — a retried tool round,
        // or a session row replayed after a partial commit. Two
        // `functionResponse` parts for one `functionCall` is not a pairing
        // Gemini accepts, and elimination must not hand this output to some
        // *other* outstanding call either, so it degrades to text.
        ::zeroclaw_log::record!(
            WARN,
            ::zeroclaw_log::Event::new(module_path!(), ::zeroclaw_log::Action::Note)
                .with_category(::zeroclaw_log::EventCategory::Tool)
                .with_outcome(::zeroclaw_log::EventOutcome::Unknown)
                .with_attrs(::serde_json::json!({"tool_call_id": id})),
            "gemini: tool result repeats an already-answered call; sending its output as text \
             rather than answering the same call twice"
        );
        return None;
    }
    let [only] = unanswered_calls else {
        if unanswered_calls.len() > 1 {
            ::zeroclaw_log::record!(
                WARN,
                ::zeroclaw_log::Event::new(module_path!(), ::zeroclaw_log::Action::Note)
                    .with_category(::zeroclaw_log::EventCategory::Tool)
                    .with_outcome(::zeroclaw_log::EventOutcome::Unknown)
                    .with_attrs(::serde_json::json!({
                        "tool_call_id": tool_call_id,
                        "outstanding": unanswered_calls.len(),
                    })),
                "gemini: tool result matches no call in the transcript and more than one is \
                 outstanding; sending its output as text rather than guessing which function \
                 produced it"
            );
        }
        return None;
    };
    Some(only.clone())
}

/// Whether a content is a tool-result turn that further results can join.
/// Parallel calls are answered by one turn carrying several
/// `functionResponse` parts.
fn is_tool_result_turn(content: &Content) -> bool {
    content.role.as_deref() == Some("user")
        && !content.parts.is_empty()
        && content
            .parts
            .iter()
            .all(|part| matches!(part, Part::FunctionResponse { .. }))
}

/// Parse an assistant history row written by the native tool dispatcher —
/// a `{"content": ..., "tool_calls": [...]}` envelope — into its narration
/// text and the model's function calls. Returns `None` for plain assistant
/// prose.
fn parse_assistant_tool_calls(content: &str) -> Option<(Option<String>, Vec<ProviderToolCall>)> {
    let value = serde_json::from_str::<serde_json::Value>(content).ok()?;
    let tool_calls =
        serde_json::from_value::<Vec<ProviderToolCall>>(value.get("tool_calls")?.clone()).ok()?;
    let text = value
        .get("content")
        .and_then(serde_json::Value::as_str)
        .map(str::to_string)
        .filter(|text| !text.trim().is_empty());
    Some((text, tool_calls))
}

/// Parse a tool-result history row — a `{"tool_call_id": ..., "content": ...}`
/// envelope — into the answered call id and the tool's output.
fn parse_tool_result(content: &str) -> Option<(Option<String>, String)> {
    let value = serde_json::from_str::<serde_json::Value>(content).ok()?;
    let output = value.get("content")?.as_str()?.to_string();
    let tool_call_id = value
        .get("tool_call_id")
        .and_then(serde_json::Value::as_str)
        .filter(|id| !id.is_empty())
        .map(str::to_string);
    Some((tool_call_id, output))
}

impl GeminiModelProvider {
    /// Render agent history as Gemini `contents`.
    ///
    /// `declared` is the function-name set this request carries — see
    /// [`declared_function_names`]. Function parts naming anything outside it
    /// degrade to text on both sides of the exchange: the call is narrated in
    /// the model turn, its result travels as [`TOOL_RESULT_PREFIX`] user text.
    fn build_chat_contents(
        messages: &[ChatMessage],
        declared: &std::collections::HashSet<String>,
    ) -> anyhow::Result<(Vec<Content>, Option<Content>)> {
        let mut system_parts: Vec<&str> = Vec::new();
        let mut contents: Vec<Content> = Vec::new();
        let mut calls_by_id: std::collections::HashMap<String, ToolCallRef> =
            std::collections::HashMap::new();
        // Ids of calls no result has claimed yet. A result whose own id went
        // missing can only be attributed by elimination, and only while
        // exactly one call is outstanding — see the `tool` arm.
        let mut unanswered_calls: Vec<String> = Vec::new();
        // Outputs that no `functionResponse` can carry — an undeclared
        // function's, or an image's bytes — trail the whole batch of results
        // as extra user turns. See `flush_pending_tool_turns`.
        let mut pending_tool_text: Vec<String> = Vec::new();
        let mut pending_images: Vec<Part> = Vec::new();

        for msg in messages {
            if msg.role != "tool" {
                flush_pending_tool_turns(
                    &mut contents,
                    &mut pending_tool_text,
                    &mut pending_images,
                );
            }

            match msg.role.as_str() {
                "system" => system_parts.push(&msg.content),
                "user" => contents.push(Content {
                    role: Some("user".to_string()),
                    parts: build_parts(&msg.content),
                }),
                "assistant" => {
                    let Some((text, tool_calls)) = parse_assistant_tool_calls(&msg.content) else {
                        contents.push(Content {
                            role: Some("model".to_string()),
                            parts: vec![Part::text(&msg.content)],
                        });
                        continue;
                    };

                    let mut parts = Vec::with_capacity(tool_calls.len() + 1);
                    if let Some(text) = text {
                        parts.push(Part::text(text));
                    }
                    for call in tool_calls {
                        let api_id = google_extra(&call, "id");
                        let thought_signature = google_extra(&call, "thought_signature");
                        let args = serde_json::from_str(&call.arguments).unwrap_or_else(|_| {
                            serde_json::Value::Object(serde_json::Map::new())
                        });
                        unanswered_calls.push(call.id.clone());
                        calls_by_id.insert(
                            call.id,
                            ToolCallRef {
                                name: call.name.clone(),
                                api_id: api_id.clone(),
                            },
                        );
                        // Still recorded above, so the result branch resolves
                        // the same name and reaches the same verdict — a
                        // narrated call is never answered by a
                        // `functionResponse`, and vice versa.
                        if !declared.contains(call.name.as_str()) {
                            parts.push(Part::text(undeclared_call_text(&call.name, &args)));
                            continue;
                        }
                        parts.push(Part::FunctionCall {
                            function_call: FunctionCall {
                                id: api_id,
                                name: call.name,
                                args,
                            },
                            thought_signature,
                        });
                    }
                    if !parts.is_empty() {
                        contents.push(Content {
                            role: Some("model".to_string()),
                            parts,
                        });
                    }
                }
                "tool" => {
                    let (tool_call_id, output) = parse_tool_result(&msg.content)
                        .unwrap_or_else(|| (None, msg.content.clone()));
                    let answered =
                        answered_call_id(tool_call_id.as_deref(), &calls_by_id, &unanswered_calls);
                    let call = answered
                        .as_ref()
                        .and_then(|id| calls_by_id.get(id.as_str()));
                    // Gemini pairs a `functionResponse` with a *declared*
                    // function by name, so take it from the call this result
                    // answers, and keep it only if this request declares it.
                    let name = call
                        .map(|call| call.name.clone())
                        .filter(|name| !name.trim().is_empty())
                        .filter(|name| declared.contains(name.as_str()));
                    let api_id = call.and_then(|call| call.api_id.clone());
                    // Answered either way: a degraded result still settles the
                    // call, so it stops being a candidate for elimination.
                    if let Some(answered) = answered {
                        unanswered_calls.retain(|id| *id != answered);
                    }

                    let (text, image_refs) = crate::multimodal::parse_image_markers(&output);
                    pending_images.extend(inline_parts(&image_refs));
                    let output = if image_refs.is_empty() {
                        output
                    } else {
                        text.trim().to_string()
                    };

                    let Some(name) = name else {
                        // Either nothing in this transcript names a function
                        // for the result to answer — the assistant turn that
                        // asked for it is gone — or the name it does have is
                        // not declared on this request. A `functionResponse`
                        // naming a function the request never declared is
                        // rejected outright, and would keep being rejected on
                        // every later turn, so hand the output over as
                        // ordinary user text rather than losing it, poisoning
                        // the turn, or wedging the session.
                        pending_tool_text.push(output);
                        continue;
                    };

                    let part = Part::FunctionResponse {
                        function_response: FunctionResponse {
                            id: api_id,
                            name,
                            response: serde_json::json!({ "output": output }),
                        },
                    };
                    match contents.last_mut() {
                        Some(last) if is_tool_result_turn(last) => last.parts.push(part),
                        _ => contents.push(Content {
                            role: Some("user".to_string()),
                            parts: vec![part],
                        }),
                    }
                }
                _ => {}
            }
        }

        flush_pending_tool_turns(&mut contents, &mut pending_tool_text, &mut pending_images);

        // Gemini rejects a request whose last turn is a model turn. History
        // trims, session restores, and steering continuations can all leave
        // the history ending on the model's own output. Those model turns are
        // the context a continuation must see, so they are kept and a final
        // user continuation turn is appended; a request with no turns at all
        // falls back to a lone user placeholder, and model-only history has
        // nothing to anchor a request on and fails explicitly instead of
        // being silently replaced with a context-free one.
        match contents.last().map(|c| c.role.as_deref()) {
            Some(Some("user")) => {}
            Some(Some("model")) => {
                if contents.iter().any(|c| c.role.as_deref() == Some("user")) {
                    contents.push(Content {
                        role: Some("user".to_string()),
                        parts: vec![Part::text("[continue]")],
                    });
                } else {
                    ::zeroclaw_log::record!(
                        WARN,
                        ::zeroclaw_log::Event::new(module_path!(), ::zeroclaw_log::Action::Reject)
                            .with_outcome(::zeroclaw_log::EventOutcome::Failure)
                            .with_attrs(::serde_json::json!({ "turns": contents.len() })),
                        "gemini: model-only history cannot anchor a generateContent request"
                    );
                    return Err(anyhow::Error::msg(
                        "gemini: history ends on model turns but contains no user turn; \
                         refusing to drop the model context or fabricate a context-free request",
                    ));
                }
            }
            // no turns at all (empty or system-only): a request still needs
            // one user turn to anchor on
            _ => {
                contents.push(Content {
                    role: Some("user".to_string()),
                    parts: vec![Part::text("[continue]")],
                });
            }
        }
        let system_instruction = if system_parts.is_empty() {
            None
        } else {
            Some(Content {
                role: None,
                parts: vec![Part::text(system_parts.join("\n\n"))],
            })
        };
        Ok((contents, system_instruction))
    }

    /// One Gemini `functionDeclaration` for a ZeroClaw tool.
    ///
    /// `field` decides how the arguments are declared — see
    /// [`parameters_field_for_model`]. `parametersJsonSchema` takes JSON
    /// Schema directly; `parameters` is limited to an OpenAPI 3.0 subset, so
    /// it costs every `$ref`, `$defs`, and `additionalProperties` the tool
    /// declared. The two are mutually exclusive; only one may be sent.
    fn function_declaration(&self, tool: &ToolSpec, field: ParametersField) -> serde_json::Value {
        let mut declaration = serde_json::json!({
            "name": tool.name,
            "description": tool.description,
        });
        // Memoized: a schema that needs rewriting is cleaned once per provider
        // instance per strategy, not rebuilt on every request.
        let parameters = self
            .schema_cache
            .clean_shared(&tool.parameters, field.cleaning_strategy());
        // Same `Arc` back means the pre-scan proved cleaning is a no-op, so
        // nothing was dropped and there is nothing to walk for.
        if !Arc::ptr_eq(&tool.parameters, &parameters) {
            self.report_dropped_schema_keywords(&tool.name, &tool.parameters, field);
        }
        if declares_arguments(&parameters) {
            // One deep copy per request survives: `ToolsPayload::Gemini`
            // carries owned `Value`s, so the memoized tree cannot ride into
            // the declaration by `Arc` the way Anthropic's `input_schema` does.
            declaration[field.wire_key()] = (*parameters).clone();
        }
        declaration
    }

    /// Every tool declared as one request's `functionDeclarations`.
    fn function_declarations(
        &self,
        tools: &[ToolSpec],
        field: ParametersField,
    ) -> Vec<serde_json::Value> {
        tools
            .iter()
            .map(|tool| self.function_declaration(tool, field))
            .collect()
    }

    /// Everything `chat` sends except the transport: the tool declarations
    /// for `model`, and the history rendered against exactly the names those
    /// declarations carry.
    ///
    /// Declared against the model this request dispatches to, not the
    /// provider: `parametersJsonSchema` exists only from generation 3 on. An
    /// empty tool list declares nothing rather than an empty `tools` block.
    fn prepare_chat(
        &self,
        messages: &[ChatMessage],
        tools: Option<&[ToolSpec]>,
        model: &str,
    ) -> anyhow::Result<PreparedChat> {
        let tools = tools.filter(|tools| !tools.is_empty()).map(|tools| {
            vec![ToolDeclarations {
                function_declarations: self
                    .function_declarations(tools, parameters_field_for_model(model)),
            }]
        });
        let declared = declared_function_names(tools.as_deref());
        let (contents, system_instruction) = Self::build_chat_contents(messages, &declared)?;
        Ok(PreparedChat {
            contents,
            system_instruction,
            tools,
        })
    }

    /// Record which schema keywords the declared parameter field's filter
    /// removed from a tool's parameters.
    ///
    /// A dropped keyword is a constraint the model never sees, so it can
    /// propose an argument the tool itself then rejects. Nothing here is
    /// recoverable at runtime, but a tool that keeps getting called wrong
    /// should not require reading the allowlist to explain.
    ///
    /// `allOf` is not among them: the cleaner folds it into the object that
    /// carries it rather than filtering it out.
    ///
    /// Walked and reported once per tool per field per provider instance. The
    /// drop set is a static property of `(schema, field)`, so repeating it
    /// every request would both re-walk the schema and bury the DEBUG stream
    /// an operator turned on to debug something else.
    fn report_dropped_schema_keywords(
        &self,
        tool: &str,
        parameters: &serde_json::Value,
        field: ParametersField,
    ) {
        if !::zeroclaw_log::debug_enabled() {
            return;
        }
        let first_sight = self
            .schema_drops_inspected
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner)
            .insert((tool.to_string(), field));
        if !first_sight {
            return;
        }
        let dropped = SchemaCleanr::dropped_keywords(parameters, field.cleaning_strategy());
        if dropped.is_empty() {
            return;
        }
        ::zeroclaw_log::record!(
            DEBUG,
            ::zeroclaw_log::Event::new(module_path!(), ::zeroclaw_log::Action::Note)
                .with_category(::zeroclaw_log::EventCategory::Tool)
                .with_attrs(::serde_json::json!({
                    "tool": tool,
                    "field": field.wire_key(),
                    "dropped": dropped.iter().collect::<Vec<_>>(),
                })),
            "gemini: tool schema keywords dropped to fit the declared parameter field"
        );
    }

    fn token_usage_from_metadata(usage: GeminiUsageMetadata) -> Option<TokenUsage> {
        if usage.prompt_token_count.is_none() && usage.candidates_token_count.is_none() {
            return None;
        }
        Some(TokenUsage {
            input_tokens: usage.prompt_token_count,
            output_tokens: usage.candidates_token_count,
            cached_input_tokens: None,
            cache_creation_input_tokens: None,
        })
    }

    async fn send_generate_content(
        &self,
        contents: Vec<Content>,
        system_instruction: Option<Content>,
        tools: Option<Vec<ToolDeclarations>>,
        model: &str,
        temperature: Option<f64>,
    ) -> anyhow::Result<GeminiTurn> {
        let auth = self.auth.as_ref().ok_or_else(|| {
            ::zeroclaw_log::record!(
                ERROR,
                ::zeroclaw_log::Event::new(module_path!(), ::zeroclaw_log::Action::Reject)
                    .with_outcome(::zeroclaw_log::EventOutcome::Failure)
                    .with_attrs(::serde_json::json!({"missing": "auth"})),
                "gemini: no auth configured"
            );
            anyhow::Error::msg(
                "Gemini API key not found. Options:\n\
                 1. Set GEMINI_API_KEY env var\n\
                 2. Run `gemini` CLI to authenticate (tokens will be reused)\n\
                 3. Run `zeroclaw auth login --model-provider gemini`\n\
                 4. Get an API key from https://aistudio.google.com/app/apikey\n\
                 5. Run `zeroclaw quickstart --model-provider gemini --api-key <key>` to configure",
            )
        })?;

        let oauth_state = match auth {
            GeminiAuth::OAuthToken(state) => Some(state.clone()),
            _ => None,
        };

        // For OAuth: get a valid (potentially refreshed) token and resolve project
        let (mut oauth_token, mut project) = match auth {
            GeminiAuth::OAuthToken(state) => {
                let token = Self::get_valid_oauth_token(state).await?;
                let proj = self.resolve_oauth_project(&token).await?;
                (Some(token), Some(proj))
            }
            GeminiAuth::ManagedOAuth => {
                let auth_service = self.auth_service.as_ref().ok_or_else(|| {
                    ::zeroclaw_log::record!(
                        ERROR,
                        ::zeroclaw_log::Event::new(module_path!(), ::zeroclaw_log::Action::Reject)
                            .with_outcome(::zeroclaw_log::EventOutcome::Failure)
                            .with_attrs(::serde_json::json!({"missing": "auth_service"})),
                        "gemini: ManagedOAuth requires auth_service"
                    );
                    anyhow::Error::msg("ManagedOAuth requires auth_service")
                })?;
                let token = auth_service
                    .get_valid_gemini_access_token(
                        self.auth_profile_override.as_deref(),
                        self.oauth_client_id.as_deref().unwrap_or(""),
                        self.oauth_client_secret.as_deref().unwrap_or(""),
                    )
                    .await?
                    .ok_or_else(|| {
                        ::zeroclaw_log::record!(
                            ERROR,
                            ::zeroclaw_log::Event::new(module_path!(), ::zeroclaw_log::Action::Reject)
                                .with_outcome(::zeroclaw_log::EventOutcome::Failure)
                                .with_attrs(::serde_json::json!({"oauth_provider": "gemini"})),
                            "gemini: auth profile not found"
                        );
                        anyhow::Error::msg(
                            "Gemini auth profile not found. Run `zeroclaw auth login --model-provider gemini`.",
                        )
                    })?;
                let proj = self.resolve_oauth_project(&token).await?;
                (Some(token), Some(proj))
            }
            _ => (None, None),
        };

        let request = GenerateContentRequest {
            contents,
            system_instruction,
            tools,
            generation_config: GenerationConfig {
                temperature,
                max_output_tokens: 8192,
            },
        };

        let url = Self::build_generate_content_url(model, auth);

        let mut response = self
            .build_generate_content_request(
                auth,
                &url,
                &request,
                model,
                true,
                project.as_deref(),
                oauth_token.as_deref(),
            )?
            .send()
            .await?;

        if !response.status().is_success() {
            let status = response.status();
            let error_text = response.text().await.unwrap_or_default();

            if auth.is_oauth() && Self::should_rotate_oauth_on_error(status, &error_text) {
                // For CLI OAuth: rotate credentials
                // For ManagedOAuth: AuthService handles refresh, just retry
                let can_retry = match auth {
                    GeminiAuth::OAuthToken(_) => {
                        if let Some(state) = oauth_state.as_ref() {
                            self.rotate_oauth_credential(state).await
                        } else {
                            false
                        }
                    }
                    GeminiAuth::ManagedOAuth => true, // AuthService refreshes automatically
                    _ => false,
                };

                if can_retry {
                    // Re-fetch token (may be refreshed)
                    let (new_token, new_project) = match auth {
                        GeminiAuth::OAuthToken(state) => {
                            let token = Self::get_valid_oauth_token(state).await?;
                            let proj = self.resolve_oauth_project(&token).await?;
                            (token, proj)
                        }
                        GeminiAuth::ManagedOAuth => {
                            let Some(auth_service) = self.auth_service.as_ref() else {
                                return Err(anyhow::Error::msg(
                                    "Gemini managed OAuth requires an auth service",
                                ));
                            };
                            let token = auth_service
                                .get_valid_gemini_access_token(
                                    self.auth_profile_override.as_deref(),
                                    self.oauth_client_id.as_deref().unwrap_or(""),
                                    self.oauth_client_secret.as_deref().unwrap_or(""),
                                )
                                .await?
                                .ok_or_else(|| {
                                    ::zeroclaw_log::record!(
                                        ERROR,
                                        ::zeroclaw_log::Event::new(
                                            module_path!(),
                                            ::zeroclaw_log::Action::Reject
                                        )
                                        .with_outcome(::zeroclaw_log::EventOutcome::Failure)
                                        .with_attrs(
                                            ::serde_json::json!({"oauth_provider": "gemini"})
                                        ),
                                        "gemini: auth profile not found"
                                    );
                                    anyhow::Error::msg("Gemini auth profile not found")
                                })?;
                            let proj = self.resolve_oauth_project(&token).await?;
                            (token, proj)
                        }
                        _ => {
                            return Err(anyhow::Error::msg(
                                "Gemini retry reached a non-refreshable authentication mode",
                            ));
                        }
                    };
                    oauth_token = Some(new_token);
                    project = Some(new_project);
                    response = self
                        .build_generate_content_request(
                            auth,
                            &url,
                            &request,
                            model,
                            true,
                            project.as_deref(),
                            oauth_token.as_deref(),
                        )?
                        .send()
                        .await?;
                } else {
                    anyhow::bail!("Gemini API error ({status}): {error_text}");
                }
            } else if auth.is_oauth()
                && Self::should_retry_oauth_without_generation_config(status, &error_text)
            {
                ::zeroclaw_log::record!(
                    WARN,
                    ::zeroclaw_log::Event::new(module_path!(), ::zeroclaw_log::Action::Note)
                        .with_outcome(::zeroclaw_log::EventOutcome::Unknown),
                    "Gemini OAuth internal endpoint rejected generationConfig; retrying without generationConfig"
                );
                response = self
                    .build_generate_content_request(
                        auth,
                        &url,
                        &request,
                        model,
                        false,
                        project.as_deref(),
                        oauth_token.as_deref(),
                    )?
                    .send()
                    .await?;
            } else {
                anyhow::bail!("Gemini API error ({status}): {error_text}");
            }
        }

        if !response.status().is_success() {
            let status = response.status();
            let error_text = response.text().await.unwrap_or_default();
            if auth.is_oauth()
                && Self::should_retry_oauth_without_generation_config(status, &error_text)
            {
                ::zeroclaw_log::record!(
                    WARN,
                    ::zeroclaw_log::Event::new(module_path!(), ::zeroclaw_log::Action::Note)
                        .with_outcome(::zeroclaw_log::EventOutcome::Unknown),
                    "Gemini OAuth internal endpoint rejected generationConfig; retrying without generationConfig"
                );
                response = self
                    .build_generate_content_request(
                        auth,
                        &url,
                        &request,
                        model,
                        false,
                        project.as_deref(),
                        oauth_token.as_deref(),
                    )?
                    .send()
                    .await?;
            } else {
                anyhow::bail!("Gemini API error ({status}): {error_text}");
            }
        }

        if !response.status().is_success() {
            let status = response.status();
            let error_text = response.text().await.unwrap_or_default();
            anyhow::bail!("Gemini API error ({status}): {error_text}");
        }

        let result: GenerateContentResponse = response.json().await?;
        GeminiTurn::from_response(result)
    }
}

/// One `chat` request's body before transport: the rendered history and the
/// tool declarations it was rendered against.
struct PreparedChat {
    contents: Vec<Content>,
    system_instruction: Option<Content>,
    tools: Option<Vec<ToolDeclarations>>,
}

/// One completed Gemini turn: the answer text, any function calls the model
/// wants executed, and the reported token usage.
struct GeminiTurn {
    text: Option<String>,
    /// The model's thinking, when the turn's answer is something else. See
    /// [`CandidateParts::reasoning`].
    reasoning: Option<String>,
    tool_calls: Vec<ProviderToolCall>,
    usage: Option<TokenUsage>,
}

impl GeminiTurn {
    /// Read one turn out of a parsed `generateContent` body, unwrapping the
    /// cloudcode-pa envelope first. A candidate that yields neither text nor a
    /// function call is no response at all; one that only asks for a tool is
    /// a complete turn.
    fn from_response(result: GenerateContentResponse) -> anyhow::Result<Self> {
        if let Some(err) = &result.error {
            anyhow::bail!("Gemini API error: {}", err.message);
        }
        let result = result.into_effective_response();
        if let Some(err) = result.error {
            anyhow::bail!("Gemini API error: {}", err.message);
        }

        let usage = result
            .usage_metadata
            .and_then(GeminiModelProvider::token_usage_from_metadata);

        let parts = result
            .candidates
            .and_then(|c| c.into_iter().next())
            .and_then(|c| c.content)
            .map(CandidateContent::into_parts)
            .unwrap_or_default();

        if parts.is_empty() {
            ::zeroclaw_log::record!(
                ERROR,
                ::zeroclaw_log::Event::new(module_path!(), ::zeroclaw_log::Action::Fail)
                    .with_outcome(::zeroclaw_log::EventOutcome::Failure),
                "gemini: empty response text"
            );
            anyhow::bail!("No response from Gemini");
        }

        Ok(Self {
            text: parts.text,
            reasoning: parts.reasoning,
            tool_calls: parts.tool_calls,
            usage,
        })
    }

    /// The turn as the agent loop reads it. Calls and reasoning pass through
    /// beside the answer, since `chat` declared the tools the calls name.
    fn into_chat_response(self) -> ProviderChatResponse {
        ProviderChatResponse {
            text: self.text,
            tool_calls: self.tool_calls,
            usage: self.usage,
            reasoning_content: self.reasoning,
        }
    }

    /// The answer text, for the entry points that declare no tools and have no
    /// way to act on a function call. A turn carrying only calls holds no
    /// answer for them, and an empty string would pass that off as a real
    /// reply — the caller must see the failure instead.
    fn into_text(self) -> anyhow::Result<String> {
        let requested_calls = self.tool_calls.len();
        let Some(text) = self.text else {
            ::zeroclaw_log::record!(
                ERROR,
                ::zeroclaw_log::Event::new(module_path!(), ::zeroclaw_log::Action::Fail)
                    .with_outcome(::zeroclaw_log::EventOutcome::Failure)
                    .with_attrs(::serde_json::json!({
                        "tool_calls": requested_calls,
                    })),
                "gemini: response carried function calls but no answer text"
            );
            anyhow::bail!("No response from Gemini");
        };
        Ok(text)
    }
}

#[async_trait]
impl ModelProvider for GeminiModelProvider {
    // ── ModelProvider-family defaults ──
    fn default_base_url(&self) -> Option<&str> {
        Some(BASE_URL)
    }

    fn capabilities(&self) -> zeroclaw_api::model_provider::ProviderCapabilities {
        zeroclaw_api::model_provider::ProviderCapabilities {
            vision: true,
            native_tool_calling: true,
            prompt_caching: false,
            extended_thinking: false,
        }
    }

    /// Declare tools without knowing which model will receive them.
    ///
    /// The model-aware path is [`Self::chat`], the only in-tree consumer of a
    /// [`ToolsPayload::Gemini`]. With no model to read a generation from, this
    /// falls back to the OpenAPI-subset `parameters` field: lossy, but every
    /// Gemini generation accepts it, where `parametersJsonSchema` fails
    /// outright below generation 3.
    fn convert_tools(&self, tools: &[ToolSpec]) -> ToolsPayload {
        ToolsPayload::Gemini {
            function_declarations: self.function_declarations(tools, ParametersField::OpenApi),
        }
    }

    async fn chat_with_system(
        &self,
        system_prompt: Option<&str>,
        message: &str,
        model: &str,
        temperature: Option<f64>,
    ) -> anyhow::Result<String> {
        let system_instruction = system_prompt.map(|sys| Content {
            role: None,
            parts: vec![Part::text(sys)],
        });

        let contents = vec![Content {
            role: Some("user".to_string()),
            parts: build_parts(message),
        }];

        self.send_generate_content(contents, system_instruction, None, model, temperature)
            .await?
            .into_text()
    }

    async fn chat_with_history(
        &self,
        messages: &[ChatMessage],
        model: &str,
        temperature: Option<f64>,
    ) -> anyhow::Result<String> {
        // This entry point declares no tools, so nothing in the history can
        // legally travel as a function part: Gemini rejects a `functionCall`
        // naming a function the request never declared. It all degrades to
        // text — which is safe to alternate with the native parts `chat`
        // sends for the same history only because the degraded shape is prose
        // rather than a callable protocol. See `undeclared_call_text`.
        let (contents, system_instruction) =
            Self::build_chat_contents(messages, &std::collections::HashSet::new())?;
        self.send_generate_content(contents, system_instruction, None, model, temperature)
            .await?
            .into_text()
    }

    async fn chat(
        &self,
        request: ProviderChatRequest<'_>,
        model: &str,
        temperature: Option<f64>,
    ) -> anyhow::Result<ProviderChatResponse> {
        let PreparedChat {
            contents,
            system_instruction,
            tools,
        } = self.prepare_chat(request.messages, request.tools, model)?;
        let turn = self
            .send_generate_content(contents, system_instruction, tools, model, temperature)
            .await?;
        Ok(turn.into_chat_response())
    }

    async fn warmup(&self) -> anyhow::Result<()> {
        if let Some(auth) = self.auth.as_ref() {
            match auth {
                GeminiAuth::ManagedOAuth => {
                    // For ManagedOAuth, verify and refresh the token if needed.
                    // This ensures fallback works even if tokens expired during daemon uptime.
                    let auth_service = self.auth_service.as_ref().ok_or_else(|| {
                        ::zeroclaw_log::record!(
                            ERROR,
                            ::zeroclaw_log::Event::new(
                                module_path!(),
                                ::zeroclaw_log::Action::Reject
                            )
                            .with_outcome(::zeroclaw_log::EventOutcome::Failure)
                            .with_attrs(::serde_json::json!({"missing": "auth_service"})),
                            "gemini: ManagedOAuth requires auth_service"
                        );
                        anyhow::Error::msg("ManagedOAuth requires auth_service")
                    })?;

                    let _token = auth_service
                        .get_valid_gemini_access_token(
                            self.auth_profile_override.as_deref(),
                            self.oauth_client_id.as_deref().unwrap_or(""),
                            self.oauth_client_secret.as_deref().unwrap_or(""),
                        )
                        .await?
                        .ok_or_else(|| {
                            ::zeroclaw_log::record!(
                                ERROR,
                                ::zeroclaw_log::Event::new(module_path!(), ::zeroclaw_log::Action::Reject)
                                    .with_outcome(::zeroclaw_log::EventOutcome::Failure)
                                    .with_attrs(::serde_json::json!({"oauth_provider": "gemini"})),
                                "gemini: auth profile not found or expired"
                            );
                            anyhow::Error::msg(
                                "Gemini auth profile not found or expired. Run: zeroclaw auth login --model-provider gemini",
                            )
                        })?;

                    // Token refresh happens in get_valid_gemini_access_token().
                    // We don't call resolve_oauth_project() here to keep warmup fast.
                    // OAuth project will be resolved lazily on first real request.
                }
                GeminiAuth::OAuthToken(_) => {
                    // CLI OAuth — cloudcode-pa does not expose a lightweight model-list probe.
                    // Token will be validated on first real request.
                }
                GeminiAuth::ExplicitKey(key) => {
                    // API key path — verify with public API models endpoint.
                    self.build_api_key_probe_request(key)?
                        .send()
                        .await?
                        .error_for_status()?;
                }
            }
        }
        Ok(())
    }

    async fn list_models(&self) -> anyhow::Result<Vec<String>> {
        // Gemini's /v1beta/models requires authentication. Onboard pulls the
        // catalog from models.dev before the user has entered a key.
        crate::models_dev::list_models_for("google").await
    }
}

impl ::zeroclaw_api::attribution::Attributable for GeminiModelProvider {
    fn role(&self) -> ::zeroclaw_api::attribution::Role {
        ::zeroclaw_api::attribution::Role::Provider(
            ::zeroclaw_api::attribution::ProviderKind::Model(
                ::zeroclaw_api::attribution::ModelProviderKind::Gemini,
            ),
        )
    }
    fn alias(&self) -> &str {
        &self.alias
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use reqwest::{StatusCode, header::AUTHORIZATION};

    fn test_oauth_auth(token: &str) -> GeminiAuth {
        GeminiAuth::OAuthToken(Arc::new(tokio::sync::Mutex::new(OAuthTokenState {
            access_token: token.to_string(),
            refresh_token: None,
            client_id: None,
            client_secret: None,
            expiry_millis: None,
        })))
    }

    fn test_model_provider(auth: Option<GeminiAuth>) -> GeminiModelProvider {
        GeminiModelProvider {
            alias: "test".to_string(),
            auth,
            oauth_project: Arc::new(tokio::sync::Mutex::new(None)),
            oauth_project_seed: None,
            oauth_cred_paths: Vec::new(),
            oauth_index: Arc::new(tokio::sync::Mutex::new(0)),
            auth_service: None,
            auth_profile_override: None,
            oauth_client_id: None,
            oauth_client_secret: None,
            schema_cache: zeroclaw_api::schema::SchemaCleanCache::new(),
            schema_drops_inspected: std::sync::Mutex::new(std::collections::HashSet::new()),
        }
    }

    #[test]
    fn normalize_non_empty_trims_and_filters() {
        assert_eq!(
            GeminiModelProvider::normalize_non_empty(" value "),
            Some("value".into())
        );
        assert_eq!(GeminiModelProvider::normalize_non_empty(""), None);
        assert_eq!(GeminiModelProvider::normalize_non_empty(" \t\n"), None);
    }

    fn resolve(body: &str) -> Option<String> {
        serde_json::from_str::<LoadCodeAssistResponse>(body)
            .unwrap()
            .resolve_project_id()
    }

    #[test]
    fn load_code_assist_reads_bare_string_project() {
        assert_eq!(
            resolve(r#"{"cloudaicompanionProject":"my-proj"}"#),
            Some("my-proj".into())
        );
    }

    #[test]
    fn load_code_assist_reads_object_id_and_project_id() {
        assert_eq!(
            resolve(r#"{"cloudaicompanionProject":{"id":"obj-proj"}}"#),
            Some("obj-proj".into())
        );
        assert_eq!(
            resolve(r#"{"cloudaicompanionProject":{"projectId":"alias-proj"}}"#),
            Some("alias-proj".into())
        );
    }

    #[test]
    fn load_code_assist_falls_back_to_current_project() {
        assert_eq!(
            resolve(r#"{"currentCloudaicompanionProject":"current-proj"}"#),
            Some("current-proj".into())
        );
        assert_eq!(
            resolve(r#"{"currentCloudaicompanionProject":{"id":"current-obj"}}"#),
            Some("current-obj".into())
        );
        assert_eq!(
            resolve(r#"{"currentCloudaicompanionProject":{"projectId":"current-alias"}}"#),
            Some("current-alias".into())
        );
    }

    #[test]
    fn load_code_assist_prefers_primary_over_current() {
        assert_eq!(
            resolve(
                r#"{"cloudaicompanionProject":"primary","currentCloudaicompanionProject":"current"}"#
            ),
            Some("primary".into())
        );
    }

    #[test]
    fn load_code_assist_skips_blank_primary_for_current() {
        assert_eq!(
            resolve(
                r#"{"cloudaicompanionProject":"  ","currentCloudaicompanionProject":"current"}"#
            ),
            Some("current".into())
        );
    }

    #[test]
    fn load_code_assist_none_when_no_project_context() {
        assert_eq!(resolve(r#"{}"#), None);
        assert_eq!(resolve(r#"{"cloudaicompanionProject":{"id":null}}"#), None);
        assert_eq!(resolve(r#"{"cloudaicompanionProject":""}"#), None);
    }

    #[test]
    fn oauth_refresh_form_uses_provided_client_credentials() {
        let form = build_oauth_refresh_form("refresh-token", Some("client-id"), Some("secret"));
        let map: std::collections::HashMap<_, _> = form.into_iter().collect();
        assert_eq!(map.get("grant_type"), Some(&"refresh_token".to_string()));
        assert_eq!(map.get("refresh_token"), Some(&"refresh-token".to_string()));
        assert_eq!(map.get("client_id"), Some(&"client-id".to_string()));
        assert_eq!(map.get("client_secret"), Some(&"secret".to_string()));
    }

    #[test]
    fn oauth_refresh_form_omits_client_credentials_when_missing() {
        let form = build_oauth_refresh_form("refresh-token", None, None);
        let map: std::collections::HashMap<_, _> = form.into_iter().collect();
        assert!(!map.contains_key("client_id"));
        assert!(!map.contains_key("client_secret"));
    }

    #[test]
    fn extract_client_id_from_id_token_prefers_aud_claim() {
        let payload = serde_json::json!({
            "aud": "aud-client-id",
            "azp": "azp-client-id"
        });
        let payload_b64 = base64::engine::general_purpose::URL_SAFE_NO_PAD
            .encode(serde_json::to_vec(&payload).unwrap());
        let token = format!("header.{payload_b64}.sig");

        assert_eq!(
            extract_client_id_from_id_token(&token),
            Some("aud-client-id".to_string())
        );
    }

    #[test]
    fn extract_client_id_from_id_token_uses_azp_when_aud_missing() {
        let payload = serde_json::json!({
            "azp": "azp-client-id"
        });
        let payload_b64 = base64::engine::general_purpose::URL_SAFE_NO_PAD
            .encode(serde_json::to_vec(&payload).unwrap());
        let token = format!("header.{payload_b64}.sig");

        assert_eq!(
            extract_client_id_from_id_token(&token),
            Some("azp-client-id".to_string())
        );
    }

    #[test]
    fn extract_client_id_from_id_token_returns_none_for_invalid_tokens() {
        assert_eq!(extract_client_id_from_id_token("invalid"), None);
        assert_eq!(extract_client_id_from_id_token("a.b.c"), None);
    }

    #[test]
    fn try_load_cli_token_derives_client_id_from_id_token_when_missing() {
        let payload = serde_json::json!({ "aud": "derived-client-id" });
        let payload_b64 = base64::engine::general_purpose::URL_SAFE_NO_PAD
            .encode(serde_json::to_vec(&payload).unwrap());
        let id_token = format!("header.{payload_b64}.sig");

        let file = tempfile::NamedTempFile::new().unwrap();
        let json = format!(
            r#"{{
                "access_token": "ya29.test-access",
                "refresh_token": "1//test-refresh",
                "id_token": "{id_token}"
            }}"#
        );
        std::fs::write(file.path(), json).unwrap();

        let path = file.path().to_path_buf();
        let state = GeminiModelProvider::try_load_gemini_cli_token(Some(&path)).unwrap();
        assert_eq!(state.client_id.as_deref(), Some("derived-client-id"));
        assert_eq!(state.client_secret, None);
    }

    #[test]
    fn provider_creates_without_key() {
        let model_provider = GeminiModelProvider::builder("test").build();
        // May pick up env vars; just verify it doesn't panic
        let _ = model_provider.auth_source();
    }

    #[test]
    fn provider_creates_with_key() {
        let model_provider = GeminiModelProvider::builder("test")
            .api_key(Some("test-api-key"))
            .build();
        assert!(matches!(
            model_provider.auth,
            Some(GeminiAuth::ExplicitKey(ref key)) if key == "test-api-key"
        ));
    }

    #[test]
    fn provider_rejects_empty_key() {
        let model_provider = GeminiModelProvider::builder("test")
            .api_key(Some(""))
            .build();
        assert!(!matches!(
            model_provider.auth,
            Some(GeminiAuth::ExplicitKey(_))
        ));
    }

    #[test]
    fn auth_source_explicit_key() {
        let model_provider = test_model_provider(Some(GeminiAuth::ExplicitKey("key".into())));
        assert_eq!(model_provider.auth_source(), "config");
    }

    #[test]
    fn auth_source_none_without_credentials() {
        let model_provider = test_model_provider(None);
        assert_eq!(model_provider.auth_source(), "none");
    }

    #[test]
    fn auth_source_oauth() {
        let model_provider = test_model_provider(Some(test_oauth_auth("ya29.mock")));
        assert_eq!(model_provider.auth_source(), "Gemini CLI OAuth");
    }

    #[test]
    fn model_name_formatting() {
        assert_eq!(
            GeminiModelProvider::format_model_name("gemini-2.0-flash"),
            "models/gemini-2.0-flash"
        );
        assert_eq!(
            GeminiModelProvider::format_model_name("models/gemini-1.5-pro"),
            "models/gemini-1.5-pro"
        );
        assert_eq!(
            GeminiModelProvider::format_internal_model_name("models/gemini-2.5-flash"),
            "gemini-2.5-flash"
        );
        assert_eq!(
            GeminiModelProvider::format_internal_model_name("gemini-2.5-flash"),
            "gemini-2.5-flash"
        );
    }

    #[test]
    fn api_key_url_excludes_credential() {
        let auth = GeminiAuth::ExplicitKey("api-key-123".into());
        let url = GeminiModelProvider::build_generate_content_url("gemini-2.0-flash", &auth);
        assert!(url.ends_with(":generateContent"));
        assert!(!url.contains("api-key-123"));
        assert!(!url.contains("?key="));
    }

    #[test]
    fn oauth_url_uses_internal_endpoint() {
        let auth = test_oauth_auth("ya29.test-token");
        let url = GeminiModelProvider::build_generate_content_url("gemini-2.0-flash", &auth);
        assert!(url.starts_with("https://cloudcode-pa.googleapis.com/v1internal"));
        assert!(url.ends_with(":generateContent"));
        assert!(!url.contains("generativelanguage.googleapis.com"));
        assert!(!url.contains("?key="));
    }

    #[test]
    fn api_key_url_uses_public_endpoint() {
        let auth = GeminiAuth::ExplicitKey("api-key-123".into());
        let url = GeminiModelProvider::build_generate_content_url("gemini-2.0-flash", &auth);
        assert!(url.contains("generativelanguage.googleapis.com/v1beta"));
        assert!(url.contains("models/gemini-2.0-flash"));
    }

    #[test]
    fn oauth_request_uses_bearer_auth_header() {
        let model_provider = test_model_provider(Some(test_oauth_auth("ya29.mock-token")));
        let auth = test_oauth_auth("ya29.mock-token");
        let url = GeminiModelProvider::build_generate_content_url("gemini-2.0-flash", &auth);
        let body = GenerateContentRequest {
            contents: vec![Content {
                role: Some("user".into()),
                parts: vec![Part::text("hello")],
            }],
            system_instruction: None,
            tools: None,
            generation_config: GenerationConfig {
                temperature: Some(0.7),
                max_output_tokens: 8192,
            },
        };

        let request = model_provider
            .build_generate_content_request(
                &auth,
                &url,
                &body,
                "gemini-2.0-flash",
                true,
                Some("test-project"),
                Some("ya29.mock-token"),
            )
            .unwrap()
            .build()
            .unwrap();

        assert_eq!(
            request
                .headers()
                .get(AUTHORIZATION)
                .and_then(|h| h.to_str().ok()),
            Some("Bearer ya29.mock-token")
        );
    }

    #[test]
    fn oauth_request_wraps_payload_in_request_envelope() {
        let model_provider = test_model_provider(Some(test_oauth_auth("ya29.mock-token")));
        let auth = test_oauth_auth("ya29.mock-token");
        let url = GeminiModelProvider::build_generate_content_url("gemini-2.0-flash", &auth);
        let body = GenerateContentRequest {
            contents: vec![Content {
                role: Some("user".into()),
                parts: vec![Part::text("hello")],
            }],
            system_instruction: None,
            tools: None,
            generation_config: GenerationConfig {
                temperature: Some(0.7),
                max_output_tokens: 8192,
            },
        };

        let request = model_provider
            .build_generate_content_request(
                &auth,
                &url,
                &body,
                "models/gemini-2.0-flash",
                true,
                Some("test-project"),
                Some("ya29.mock-token"),
            )
            .unwrap()
            .build()
            .unwrap();

        let payload = request
            .body()
            .and_then(|b| b.as_bytes())
            .expect("json request body should be bytes");
        let json: serde_json::Value = serde_json::from_slice(payload).unwrap();

        assert_eq!(json["model"], "gemini-2.0-flash");
        assert!(json.get("generationConfig").is_none());
        assert!(json.get("request").is_some());
        assert!(json["request"].get("generationConfig").is_some());
    }

    #[test]
    fn api_key_request_uses_header_without_url_credential() {
        let model_provider =
            test_model_provider(Some(GeminiAuth::ExplicitKey("api-key-123".into())));
        let auth = GeminiAuth::ExplicitKey("api-key-123".into());
        let url = GeminiModelProvider::build_generate_content_url("gemini-2.0-flash", &auth);
        let body = GenerateContentRequest {
            contents: vec![Content {
                role: Some("user".into()),
                parts: vec![Part::text("hello")],
            }],
            system_instruction: None,
            tools: None,
            generation_config: GenerationConfig {
                temperature: Some(0.7),
                max_output_tokens: 8192,
            },
        };

        let request = model_provider
            .build_generate_content_request(
                &auth,
                &url,
                &body,
                "gemini-2.0-flash",
                true,
                None,
                None,
            )
            .unwrap()
            .build()
            .unwrap();

        assert!(request.headers().get(AUTHORIZATION).is_none());
        assert_eq!(
            request
                .headers()
                .get(GOOGLE_API_KEY_HEADER)
                .and_then(|value| value.to_str().ok()),
            Some("api-key-123")
        );
        assert!(
            request
                .headers()
                .get(GOOGLE_API_KEY_HEADER)
                .expect("API key header")
                .is_sensitive()
        );
        assert!(!request.url().as_str().contains("api-key-123"));
        assert!(!request.url().query_pairs().any(|(name, _)| name == "key"));
    }

    #[test]
    fn api_key_probe_uses_header_without_url_credential() {
        let model_provider = test_model_provider(None);
        let request = model_provider
            .build_api_key_probe_request("api-key-123")
            .unwrap()
            .build()
            .unwrap();

        assert_eq!(
            request
                .headers()
                .get(GOOGLE_API_KEY_HEADER)
                .and_then(|value| value.to_str().ok()),
            Some("api-key-123")
        );
        assert!(
            request
                .headers()
                .get(GOOGLE_API_KEY_HEADER)
                .expect("API key header")
                .is_sensitive()
        );
        assert_eq!(
            request.url().as_str(),
            "https://generativelanguage.googleapis.com/v1beta/models"
        );
    }

    #[test]
    fn api_key_header_rejects_control_characters_without_echoing_credential() {
        let credential = "api-key-123\r\ninjected: value";
        let error = GeminiModelProvider::sensitive_api_key_header(credential)
            .expect_err("control characters must be rejected");

        assert_eq!(
            error.to_string(),
            "Gemini API key contains invalid header characters"
        );
        assert!(!error.to_string().contains(credential));
    }

    #[test]
    fn request_serialization() {
        let request = GenerateContentRequest {
            contents: vec![Content {
                role: Some("user".to_string()),
                parts: vec![Part::text("Hello")],
            }],
            system_instruction: Some(Content {
                role: None,
                parts: vec![Part::text("You are helpful")],
            }),
            tools: None,
            generation_config: GenerationConfig {
                temperature: Some(0.7),
                max_output_tokens: 8192,
            },
        };

        let json = serde_json::to_string(&request).unwrap();
        assert!(json.contains("\"role\":\"user\""));
        assert!(json.contains("\"text\":\"Hello\""));
        assert!(json.contains("\"systemInstruction\""));
        assert!(!json.contains("\"system_instruction\""));
        assert!(json.contains("\"temperature\":0.7"));
        assert!(json.contains("\"maxOutputTokens\":8192"));
    }

    #[test]
    fn internal_request_includes_model() {
        let request = InternalGenerateContentEnvelope {
            model: "gemini-3-pro-preview".to_string(),
            project: Some("test-project".to_string()),
            user_prompt_id: Some("prompt-123".to_string()),
            request: InternalGenerateContentRequest {
                contents: vec![Content {
                    role: Some("user".to_string()),
                    parts: vec![Part::text("Hello")],
                }],
                system_instruction: None,
                tools: None,
                generation_config: Some(GenerationConfig {
                    temperature: Some(0.7),
                    max_output_tokens: 8192,
                }),
            },
        };

        let json = serde_json::to_string(&request).unwrap();
        assert!(json.contains("\"model\":\"gemini-3-pro-preview\""));
        assert!(json.contains("\"request\""));
        assert!(json.contains("\"generationConfig\""));
        assert!(json.contains("\"maxOutputTokens\":8192"));
        assert!(json.contains("\"user_prompt_id\":\"prompt-123\""));
        assert!(json.contains("\"project\":\"test-project\""));
        assert!(json.contains("\"role\":\"user\""));
        assert!(json.contains("\"temperature\":0.7"));
    }

    #[test]
    fn internal_request_omits_generation_config_when_none() {
        let request = InternalGenerateContentEnvelope {
            model: "gemini-3-pro-preview".to_string(),
            project: Some("test-project".to_string()),
            user_prompt_id: None,
            request: InternalGenerateContentRequest {
                contents: vec![Content {
                    role: Some("user".to_string()),
                    parts: vec![Part::text("Hello")],
                }],
                system_instruction: None,
                tools: None,
                generation_config: None,
            },
        };

        let json = serde_json::to_string(&request).unwrap();
        assert!(!json.contains("generationConfig"));
        assert!(json.contains("\"model\":\"gemini-3-pro-preview\""));
    }

    #[test]
    fn internal_request_includes_project() {
        let request = InternalGenerateContentEnvelope {
            model: "gemini-2.5-flash".to_string(),
            project: Some("my-gcp-project-id".to_string()),
            user_prompt_id: None,
            request: InternalGenerateContentRequest {
                contents: vec![Content {
                    role: Some("user".to_string()),
                    parts: vec![Part::text("Hello")],
                }],
                system_instruction: None,
                tools: None,
                generation_config: None,
            },
        };

        let json = serde_json::to_string(&request).unwrap();
        assert!(json.contains("\"project\":\"my-gcp-project-id\""));
    }

    #[test]
    fn creds_deserialize_with_expiry_date() {
        let json = r#"{
            "access_token": "ya29.test-token",
            "refresh_token": "1//test-refresh",
            "expiry_date": 4102444800000
        }"#;

        let creds: GeminiCliOAuthCreds = serde_json::from_str(json).unwrap();
        assert_eq!(creds.access_token.as_deref(), Some("ya29.test-token"));
        assert_eq!(creds.refresh_token.as_deref(), Some("1//test-refresh"));
        assert_eq!(creds.expiry_date, Some(4_102_444_800_000));
        assert!(creds.expiry.is_none());
    }

    #[test]
    fn creds_deserialize_accepts_camel_case_fields() {
        let json = r#"{
            "access_token": "ya29.test-token",
            "idToken": "header.payload.sig",
            "refresh_token": "1//test-refresh",
            "clientId": "test-client-id",
            "clientSecret": "test-client-secret",
            "expiryDate": 4102444800000
        }"#;

        let creds: GeminiCliOAuthCreds = serde_json::from_str(json).unwrap();
        assert_eq!(creds.id_token.as_deref(), Some("header.payload.sig"));
        assert_eq!(creds.client_id.as_deref(), Some("test-client-id"));
        assert_eq!(creds.client_secret.as_deref(), Some("test-client-secret"));
        assert_eq!(creds.expiry_date, Some(4_102_444_800_000));
    }

    #[test]
    fn oauth_retry_detection_for_generation_config_rejection() {
        // Bare quotes (e.g. pre-parsed error string)
        let err =
            "Invalid JSON payload received. Unknown name \"generationConfig\": Cannot find field.";
        assert!(
            GeminiModelProvider::should_retry_oauth_without_generation_config(
                StatusCode::BAD_REQUEST,
                err
            )
        );
        // JSON-escaped quotes (raw response body from Google API)
        let err_json = r#"Invalid JSON payload received. Unknown name \"generationConfig\": Cannot find field."#;
        assert!(
            GeminiModelProvider::should_retry_oauth_without_generation_config(
                StatusCode::BAD_REQUEST,
                err_json
            )
        );
        assert!(
            !GeminiModelProvider::should_retry_oauth_without_generation_config(
                StatusCode::UNAUTHORIZED,
                err
            )
        );
        assert!(
            !GeminiModelProvider::should_retry_oauth_without_generation_config(
                StatusCode::BAD_REQUEST,
                "something else"
            )
        );
    }

    #[test]
    fn response_deserialization() {
        let json = r#"{
            "candidates": [{
                "content": {
                    "parts": [{"text": "Hello there!"}]
                }
            }]
        }"#;

        let response: GenerateContentResponse = serde_json::from_str(json).unwrap();
        assert!(response.candidates.is_some());
        let text = response
            .candidates
            .unwrap()
            .into_iter()
            .next()
            .unwrap()
            .content
            .unwrap()
            .parts
            .into_iter()
            .next()
            .unwrap()
            .text;
        assert_eq!(text, Some("Hello there!".to_string()));
    }

    #[test]
    fn error_response_deserialization() {
        let json = r#"{
            "error": {
                "message": "Invalid API key"
            }
        }"#;

        let response: GenerateContentResponse = serde_json::from_str(json).unwrap();
        assert!(response.error.is_some());
        assert_eq!(response.error.unwrap().message, "Invalid API key");
    }

    #[test]
    fn internal_response_deserialization() {
        let json = r#"{
            "response": {
                "candidates": [{
                    "content": {
                        "parts": [{"text": "Hello from internal"}]
                    }
                }]
            }
        }"#;

        let response: GenerateContentResponse = serde_json::from_str(json).unwrap();
        let text = response
            .into_effective_response()
            .candidates
            .unwrap()
            .into_iter()
            .next()
            .unwrap()
            .content
            .unwrap()
            .parts
            .into_iter()
            .next()
            .unwrap()
            .text;
        assert_eq!(text, Some("Hello from internal".to_string()));
    }

    // ── Thinking model response tests ──────────────────────────────────────

    #[test]
    fn thinking_response_extracts_non_thinking_text() {
        let json = r#"{
            "candidates": [{
                "content": {
                    "parts": [
                        {"thought": true, "text": "Let me think about this..."},
                        {"text": "The answer is 42."},
                        {"thoughtSignature": "c2lnbmF0dXJl"}
                    ]
                }
            }]
        }"#;

        let response: GenerateContentResponse = serde_json::from_str(json).unwrap();
        let candidate = response.candidates.unwrap().into_iter().next().unwrap();
        let text = candidate.content.unwrap().into_parts().text;
        assert_eq!(text, Some("The answer is 42.".to_string()));
    }

    #[test]
    fn non_thinking_response_unaffected() {
        let json = r#"{
            "candidates": [{
                "content": {
                    "parts": [{"text": "Hello there!"}]
                }
            }]
        }"#;

        let response: GenerateContentResponse = serde_json::from_str(json).unwrap();
        let candidate = response.candidates.unwrap().into_iter().next().unwrap();
        let text = candidate.content.unwrap().into_parts().text;
        assert_eq!(text, Some("Hello there!".to_string()));
    }

    #[test]
    fn thinking_only_response_falls_back_to_thinking_text() {
        let json = r#"{
            "candidates": [{
                "content": {
                    "parts": [
                        {"thought": true, "text": "I need more context..."},
                        {"thoughtSignature": "c2lnbmF0dXJl"}
                    ]
                }
            }]
        }"#;

        let response: GenerateContentResponse = serde_json::from_str(json).unwrap();
        let candidate = response.candidates.unwrap().into_iter().next().unwrap();
        let text = candidate.content.unwrap().into_parts().text;
        assert_eq!(text, Some("I need more context...".to_string()));
    }

    #[test]
    fn empty_parts_returns_none() {
        let json = r#"{
            "candidates": [{
                "content": {
                    "parts": []
                }
            }]
        }"#;

        let response: GenerateContentResponse = serde_json::from_str(json).unwrap();
        let candidate = response.candidates.unwrap().into_iter().next().unwrap();
        let text = candidate.content.unwrap().into_parts().text;
        assert_eq!(text, None);
    }

    #[test]
    fn multiple_text_parts_concatenated() {
        let json = r#"{
            "candidates": [{
                "content": {
                    "parts": [
                        {"text": "Part one. "},
                        {"text": "Part two."}
                    ]
                }
            }]
        }"#;

        let response: GenerateContentResponse = serde_json::from_str(json).unwrap();
        let candidate = response.candidates.unwrap().into_iter().next().unwrap();
        let text = candidate.content.unwrap().into_parts().text;
        assert_eq!(text, Some("Part one. Part two.".to_string()));
    }

    #[test]
    fn thought_signature_only_parts_skipped() {
        let json = r#"{
            "candidates": [{
                "content": {
                    "parts": [
                        {"thoughtSignature": "c2lnbmF0dXJl"}
                    ]
                }
            }]
        }"#;

        let response: GenerateContentResponse = serde_json::from_str(json).unwrap();
        let candidate = response.candidates.unwrap().into_iter().next().unwrap();
        let text = candidate.content.unwrap().into_parts().text;
        assert_eq!(text, None);
    }

    #[test]
    fn internal_response_thinking_model() {
        let json = r#"{
            "response": {
                "candidates": [{
                    "content": {
                        "parts": [
                            {"thought": true, "text": "reasoning..."},
                            {"text": "final answer"}
                        ]
                    }
                }]
            }
        }"#;

        let response: GenerateContentResponse = serde_json::from_str(json).unwrap();
        let effective = response.into_effective_response();
        let candidate = effective.candidates.unwrap().into_iter().next().unwrap();
        let text = candidate.content.unwrap().into_parts().text;
        assert_eq!(text, Some("final answer".to_string()));
    }

    #[tokio::test]
    async fn warmup_without_key_is_noop() {
        let model_provider = test_model_provider(None);
        let result = model_provider.warmup().await;
        assert!(result.is_ok());
    }

    #[tokio::test]
    async fn warmup_oauth_is_noop() {
        let model_provider = test_model_provider(Some(test_oauth_auth("ya29.mock-token")));
        let result = model_provider.warmup().await;
        assert!(result.is_ok());
    }

    #[test]
    fn discover_oauth_cred_paths_does_not_panic() {
        let _paths = GeminiModelProvider::discover_oauth_cred_paths();
    }

    #[tokio::test]
    async fn rotate_oauth_without_alternatives_returns_false() {
        let state = Arc::new(tokio::sync::Mutex::new(OAuthTokenState {
            access_token: "ya29.mock".to_string(),
            refresh_token: None,
            client_id: None,
            client_secret: None,
            expiry_millis: None,
        }));
        let model_provider = test_model_provider(Some(GeminiAuth::OAuthToken(state.clone())));
        assert!(!model_provider.rotate_oauth_credential(&state).await);
    }

    #[test]
    fn response_parses_usage_metadata() {
        let json = r#"{
            "candidates": [{"content": {"parts": [{"text": "Hello"}]}}],
            "usageMetadata": {"promptTokenCount": 120, "candidatesTokenCount": 40}
        }"#;
        let resp: GenerateContentResponse = serde_json::from_str(json).unwrap();
        let usage = resp.usage_metadata.unwrap();
        assert_eq!(usage.prompt_token_count, Some(120));
        assert_eq!(usage.candidates_token_count, Some(40));
    }

    #[test]
    fn response_usage_metadata_maps_to_token_usage() {
        let usage = GeminiUsageMetadata {
            prompt_token_count: Some(120),
            candidates_token_count: Some(40),
        };

        let token_usage =
            GeminiModelProvider::token_usage_from_metadata(usage).expect("usage counts should map");

        assert_eq!(token_usage.input_tokens, Some(120));
        assert_eq!(token_usage.output_tokens, Some(40));
        assert_eq!(token_usage.cached_input_tokens, None);
    }

    #[test]
    fn empty_usage_metadata_maps_to_none() {
        let usage = GeminiUsageMetadata {
            prompt_token_count: None,
            candidates_token_count: None,
        };

        assert!(GeminiModelProvider::token_usage_from_metadata(usage).is_none());
    }

    #[test]
    fn wrapped_response_preserves_outer_usage_metadata() {
        let json = r#"{
            "usageMetadata": {"promptTokenCount": 120, "candidatesTokenCount": 40},
            "response": {
                "candidates": [{"content": {"parts": [{"text": "Hello"}]}}]
            }
        }"#;

        let resp: GenerateContentResponse = serde_json::from_str(json).unwrap();
        let effective = resp.into_effective_response();
        let usage = effective.usage_metadata.unwrap();

        assert_eq!(usage.prompt_token_count, Some(120));
        assert_eq!(usage.candidates_token_count, Some(40));
    }

    #[test]
    fn wrapped_response_prefers_inner_usage_metadata() {
        let json = r#"{
            "usageMetadata": {"promptTokenCount": 120, "candidatesTokenCount": 40},
            "response": {
                "candidates": [{"content": {"parts": [{"text": "Hello"}]}}],
                "usageMetadata": {"promptTokenCount": 5, "candidatesTokenCount": 2}
            }
        }"#;

        let resp: GenerateContentResponse = serde_json::from_str(json).unwrap();
        let effective = resp.into_effective_response();
        let usage = effective.usage_metadata.unwrap();

        assert_eq!(usage.prompt_token_count, Some(5));
        assert_eq!(usage.candidates_token_count, Some(2));
    }

    #[test]
    fn response_parses_without_usage_metadata() {
        let json = r#"{"candidates": [{"content": {"parts": [{"text": "Hello"}]}}]}"#;
        let resp: GenerateContentResponse = serde_json::from_str(json).unwrap();
        assert!(resp.usage_metadata.is_none());
    }

    #[tokio::test]
    async fn warmup_managed_oauth_requires_auth_service() {
        let model_provider = GeminiModelProvider {
            alias: "test".to_string(),
            auth: Some(GeminiAuth::ManagedOAuth),
            oauth_project: Arc::new(tokio::sync::Mutex::new(None)),
            oauth_project_seed: None,
            oauth_cred_paths: Vec::new(),
            oauth_index: Arc::new(tokio::sync::Mutex::new(0)),
            auth_service: None, // Missing auth_service
            auth_profile_override: None,
            oauth_client_id: None,
            oauth_client_secret: None,
            schema_cache: zeroclaw_api::schema::SchemaCleanCache::new(),
            schema_drops_inspected: std::sync::Mutex::new(std::collections::HashSet::new()),
        };

        let result = model_provider.warmup().await;
        assert!(result.is_err());
        assert!(
            result
                .unwrap_err()
                .to_string()
                .contains("ManagedOAuth requires auth_service")
        );
    }

    #[tokio::test]
    async fn warmup_cli_oauth_skips_validation() {
        let model_provider = test_model_provider(Some(test_oauth_auth("fake_token")));
        let result = model_provider.warmup().await;
        // Should succeed without making HTTP requests
        assert!(result.is_ok());
    }

    // ── Part enum serialization tests ────────────────────────────────────

    #[test]
    fn part_text_serializes_as_text_object() {
        let part = Part::text("hello");
        let json = serde_json::to_value(&part).unwrap();
        assert_eq!(json, serde_json::json!({"text": "hello"}));
    }

    #[test]
    fn part_inline_serializes_as_inline_data_object() {
        let part = Part::Inline {
            inline_data: InlineData {
                mime_type: "image/png".to_string(),
                data: "iVBOR...".to_string(),
            },
        };
        let json = serde_json::to_value(&part).unwrap();
        assert_eq!(
            json,
            serde_json::json!({"inline_data": {"mime_type": "image/png", "data": "iVBOR..."}})
        );
    }

    #[test]
    fn part_text_constructor_accepts_string_and_str() {
        let from_str = Part::text("hello");
        let from_string = Part::text(String::from("hello"));
        // Both should serialize identically
        assert_eq!(
            serde_json::to_value(&from_str).unwrap(),
            serde_json::to_value(&from_string).unwrap(),
        );
    }

    #[test]
    fn content_with_mixed_parts_serializes_correctly() {
        let content = Content {
            role: Some("user".to_string()),
            parts: vec![
                Part::text("Describe this image:"),
                Part::Inline {
                    inline_data: InlineData {
                        mime_type: "image/jpeg".to_string(),
                        data: "/9j/4AAQ...".to_string(),
                    },
                },
            ],
        };
        let json = serde_json::to_value(&content).unwrap();
        let parts = json["parts"].as_array().unwrap();
        assert_eq!(parts.len(), 2);
        assert!(parts[0].get("text").is_some());
        assert!(parts[1].get("inline_data").is_some());
    }

    // ── build_parts tests ────────────────────────────────────────────────

    #[test]
    fn build_parts_plain_text_returns_single_text_part() {
        let parts = build_parts("Hello, world!");
        assert_eq!(parts.len(), 1);
        assert_eq!(
            serde_json::to_value(&parts[0]).unwrap(),
            serde_json::json!({"text": "Hello, world!"})
        );
    }

    #[test]
    fn build_parts_empty_string_returns_single_text_part() {
        let parts = build_parts("");
        assert_eq!(parts.len(), 1);
        // Falls back to original content when no markers and trimmed is empty
        assert_eq!(
            serde_json::to_value(&parts[0]).unwrap(),
            serde_json::json!({"text": ""})
        );
    }

    #[test]
    fn build_parts_extracts_data_uri_as_inline_part() {
        let content = "Check this [IMAGE:data:image/png;base64,iVBORw0KGgo=]";
        let parts = build_parts(content);
        assert_eq!(parts.len(), 2);
        // First part is text
        assert_eq!(
            serde_json::to_value(&parts[0]).unwrap(),
            serde_json::json!({"text": "Check this"})
        );
        // Second part is inline image
        assert_eq!(
            serde_json::to_value(&parts[1]).unwrap(),
            serde_json::json!({"inline_data": {"mime_type": "image/png", "data": "iVBORw0KGgo="}})
        );
    }

    #[test]
    fn build_parts_multiple_images() {
        let content = "Image A: [IMAGE:data:image/png;base64,AAAA] Image B: [IMAGE:data:image/jpeg;base64,BBBB]";
        let parts = build_parts(content);
        assert_eq!(parts.len(), 3); // text + 2 images
        // Verify both inline parts
        let inline_parts: Vec<_> = parts
            .iter()
            .filter(|p| matches!(p, Part::Inline { .. }))
            .collect();
        assert_eq!(inline_parts.len(), 2);
    }

    #[test]
    fn build_parts_ignores_non_data_uri_markers() {
        // File paths and URLs are not data URIs — build_parts should only
        // extract data: URIs, leaving non-data markers as stripped text.
        let content = "Look [IMAGE:/tmp/photo.png]";
        let parts = build_parts(content);
        // parse_image_markers extracts the marker, but build_parts only
        // converts data: URIs to inline parts. The text remains.
        for part in &parts {
            assert!(matches!(part, Part::Text { .. }));
        }
    }

    #[test]
    fn build_parts_image_only_still_produces_inline_part() {
        let content = "[IMAGE:data:image/gif;base64,R0lGODlh]";
        let parts = build_parts(content);
        // Should have just the inline part (text is empty after marker removal)
        assert_eq!(parts.len(), 1);
        assert!(matches!(&parts[0], Part::Inline { .. }));
    }

    // ── chat_with_history uses build_parts for user messages ─────────────

    #[test]
    fn chat_with_history_maps_roles_correctly() {
        let messages = vec![
            ChatMessage::system("You are helpful"),
            ChatMessage::user("Hello [IMAGE:data:image/png;base64,AA==]"),
        ];

        let (contents, system_instruction) =
            GeminiModelProvider::build_chat_contents(&messages, &declared(&[]))
                .expect("user-anchored history must build");

        let system_instruction = system_instruction.expect("system prompt should be separated");
        assert_eq!(system_instruction.role, None);
        assert!(
            matches!(&system_instruction.parts[0], Part::Text { text } if text == "You are helpful")
        );

        assert_eq!(contents.len(), 1);
        assert_eq!(contents[0].role.as_deref(), Some("user"));
        assert!(
            contents[0]
                .parts
                .iter()
                .any(|p| matches!(p, Part::Inline { .. }))
        );
    }

    #[test]
    fn chat_contents_preserves_trailing_model_turn_and_appends_user_continuation() {
        let messages = vec![
            ChatMessage::system("You are helpful"),
            ChatMessage::user("Hello"),
            ChatMessage::assistant("I see the image"),
        ];

        let (contents, _system_instruction) =
            GeminiModelProvider::build_chat_contents(&messages, &declared(&[]))
                .expect("history with a prior user turn must build");

        // the model turn is the context a continuation needs, so it is kept
        // and the request is anchored with a final user continuation turn
        assert_eq!(contents.len(), 3, "model turn must be preserved");
        assert_eq!(contents[0].role.as_deref(), Some("user"));
        assert_eq!(contents[1].role.as_deref(), Some("model"));
        assert!(
            matches!(&contents[1].parts[0], Part::Text { text } if text == "I see the image"),
            "the assistant context must survive the request build"
        );
        assert_eq!(contents[2].role.as_deref(), Some("user"));
        assert!(matches!(&contents[2].parts[0], Part::Text { text } if text == "[continue]"));
    }

    #[test]
    fn chat_contents_model_only_history_fails_explicitly() {
        let messages = vec![ChatMessage::assistant("I see the image")];

        let result = GeminiModelProvider::build_chat_contents(&messages, &declared(&[]));

        let err = result.expect_err("model-only history must fail instead of losing context");
        assert!(
            err.to_string()
                .contains("history ends on model turns but contains no user turn"),
            "error should name the anchoring problem, got: {err}"
        );
    }

    #[test]
    fn chat_contents_empty_history_falls_back_to_bare_continue() {
        let messages = vec![ChatMessage::system("You are helpful")];

        let (contents, _system_instruction) =
            GeminiModelProvider::build_chat_contents(&messages, &declared(&[]))
                .expect("system-only history must build with a bare anchor turn");

        assert_eq!(contents.len(), 1);
        assert_eq!(contents[0].role.as_deref(), Some("user"));
        assert!(matches!(&contents[0].parts[0], Part::Text { text } if text == "[continue]"));
    }

    #[test]
    fn chat_contents_ending_on_user_turn_is_untouched() {
        let messages = vec![
            ChatMessage::user("Hello"),
            ChatMessage::assistant("Hi"),
            ChatMessage::user("Now continue"),
        ];

        let (contents, _system_instruction) =
            GeminiModelProvider::build_chat_contents(&messages, &declared(&[]))
                .expect("history already ending on a user turn must build");

        assert_eq!(contents.len(), 3, "no synthetic turn may be appended");
        assert!(
            !contents.iter().any(|c| {
                c.parts
                    .iter()
                    .any(|p| matches!(p, Part::Text { text } if text == "[continue]"))
            }),
            "no [continue] placeholder may appear"
        );
    }

    // ── Native tool calling ──────────────────────────────────────────────

    #[test]
    fn capabilities_report_native_tool_calling() {
        let model_provider = test_model_provider(Some(GeminiAuth::ExplicitKey("key".into())));
        assert!(model_provider.capabilities().native_tool_calling);
        assert!(model_provider.supports_native_tools());
    }

    #[test]
    fn parameters_field_follows_the_model_generation() {
        for model in [
            "gemini-3-pro-preview",
            "gemini-3-flash",
            "models/gemini-3-pro",
            "GEMINI-3-PRO",
        ] {
            assert_eq!(
                parameters_field_for_model(model),
                ParametersField::JsonSchema,
                "{model} reads parametersJsonSchema"
            );
        }
        for model in [
            "gemini-2.5-pro",
            "gemini-2.5-flash-lite",
            "gemini-2.0-flash",
            "gemini-1.5-pro",
            "models/gemini-1.5-flash-8b",
        ] {
            assert_eq!(
                parameters_field_for_model(model),
                ParametersField::OpenApi,
                "{model} predates parametersJsonSchema and must fall back"
            );
        }
    }

    #[test]
    fn an_unreadable_model_name_gets_the_modern_parameters_field() {
        // The generations that need the fallback are a closed set, so a name
        // no generation can be read from skews new. Defaulting to the legacy
        // field instead would silently strip constraints from every model
        // Google ships after generation 3.
        for model in ["gemini-flash-latest", "gemini-exp-1206", "tuned-model-abc123", ""] {
            assert_eq!(
                parameters_field_for_model(model),
                ParametersField::JsonSchema,
                "{model:?}"
            );
        }
    }

    #[test]
    fn convert_tools_returns_gemini_function_declarations() {
        let model_provider = test_model_provider(Some(GeminiAuth::ExplicitKey("key".into())));
        let tools = vec![ToolSpec::new(
            "memory_store",
            "Store a value in memory",
            serde_json::json!({
                "type": "object",
                "properties": {"key": {"type": "string"}},
                "required": ["key"]
            }),
        )];

        let ToolsPayload::Gemini {
            function_declarations,
        } = model_provider.convert_tools(&tools)
        else {
            panic!("Gemini must return native functionDeclarations");
        };

        assert_eq!(function_declarations.len(), 1);
        assert_eq!(function_declarations[0]["name"], "memory_store");
        // No model to read a generation from, so the field every generation
        // accepts. `chat` overrides this with the model-aware choice.
        assert_eq!(
            function_declarations[0]["parameters"]["properties"]["key"]["type"],
            "string"
        );
        assert!(
            function_declarations[0].get("parametersJsonSchema").is_none(),
            "parameters and parametersJsonSchema are mutually exclusive"
        );
    }

    /// One declaration, built through a throwaway provider so the memoized
    /// schema cache is exercised exactly as a real request would exercise it.
    fn function_declaration_for(tool: &ToolSpec, field: ParametersField) -> serde_json::Value {
        test_model_provider(Some(GeminiAuth::ExplicitKey("key".into())))
            .function_declaration(tool, field)
    }

    /// The generation-3 declaration, which is what most of these assert on.
    fn json_schema_declaration_for(tool: &ToolSpec) -> serde_json::Value {
        function_declaration_for(tool, ParametersField::JsonSchema)
    }

    #[test]
    fn function_declaration_omits_parameters_without_properties() {
        let tool = ToolSpec::new(
            "ping",
            "No arguments",
            serde_json::json!({"type": "object", "properties": {}}),
        );

        let declaration = json_schema_declaration_for(&tool);

        assert_eq!(declaration["name"], "ping");
        assert!(
            declaration.get("parametersJsonSchema").is_none(),
            "Gemini rejects a parameters schema with no properties"
        );
    }

    #[test]
    fn function_declaration_sends_a_free_form_map_schema() {
        // `HashMap<String, T>` as the whole argument object: the keys are the
        // caller's, so there is no `properties` map to find.
        let tool = ToolSpec::new(
            "set_labels",
            "Free-form map",
            serde_json::json!({
                "type": "object",
                "additionalProperties": {"type": "string"}
            }),
        );

        let declaration = json_schema_declaration_for(&tool);

        assert_eq!(
            declaration["parametersJsonSchema"]["additionalProperties"]["type"],
            "string"
        );

        // The OpenAPI subset has no `additionalProperties` to carry it, so
        // the cleaned schema really does describe nothing and is omitted.
        let open_api = function_declaration_for(&tool, ParametersField::OpenApi);
        assert!(open_api.get("parameters").is_none());
    }

    #[test]
    fn a_closed_object_with_no_properties_still_declares_nothing() {
        let tool = ToolSpec::new(
            "ping",
            "No arguments",
            serde_json::json!({"type": "object", "additionalProperties": false}),
        );

        let declaration = json_schema_declaration_for(&tool);

        assert!(declaration.get("parametersJsonSchema").is_none());
    }

    #[test]
    fn function_declaration_sends_a_schema_that_only_names_required_arguments() {
        let tool = ToolSpec::new(
            "query",
            "Underspecified",
            serde_json::json!({"type": "object", "required": ["q"]}),
        );

        for field in [ParametersField::JsonSchema, ParametersField::OpenApi] {
            let declaration = function_declaration_for(&tool, field);
            assert_eq!(
                declaration[field.wire_key()]["required"],
                serde_json::json!(["q"]),
                "{field:?} dropped the only argument the tool names"
            );
        }
    }

    #[test]
    fn function_declaration_sends_a_root_ref_schema() {
        // No top-level `properties`, but the `$ref` still describes arguments.
        let tool = ToolSpec::new(
            "query",
            "Root-level ref",
            serde_json::json!({
                "$ref": "#/$defs/Args",
                "$defs": {"Args": {"type": "object", "properties": {"q": {"type": "string"}}}}
            }),
        );

        let declaration = json_schema_declaration_for(&tool);

        assert_eq!(declaration["parametersJsonSchema"]["$ref"], "#/$defs/Args");
    }

    #[test]
    fn function_declaration_keeps_refs_unresolved() {
        // `parametersJsonSchema` resolves `$ref` itself, so the cleaner must
        // leave the pointer and its `$defs` intact instead of inlining them.
        let tool = ToolSpec::new(
            "query",
            "Search with a ref",
            serde_json::json!({
                "type": "object",
                "properties": {"filter": {"$ref": "#/$defs/FilterSpec"}},
                "$defs": {
                    "FilterSpec": {
                        "type": "object",
                        "properties": {"field": {"type": "string"}}
                    }
                }
            }),
        );

        let declaration = json_schema_declaration_for(&tool);
        let parameters = &declaration["parametersJsonSchema"];

        assert_eq!(parameters["properties"]["filter"]["$ref"], "#/$defs/FilterSpec");
        assert_eq!(
            parameters["$defs"]["FilterSpec"]["properties"]["field"]["type"],
            "string"
        );
    }

    #[test]
    fn function_declaration_keeps_the_structure_the_openapi_subset_would_strip() {
        let tool = ToolSpec::new(
            "cron_add",
            "Schedule a job",
            serde_json::json!({
                "type": "object",
                "additionalProperties": false,
                "properties": {
                    "retries": {"type": "integer", "minimum": 1, "maximum": 5},
                    "tags": {"type": "array", "items": {"type": "string"}, "minItems": 1},
                    "schedule": {
                        "oneOf": [
                            {
                                "type": "object",
                                "properties": {"kind": {"type": "string", "enum": ["cron"]}},
                                "required": ["kind"]
                            },
                            {
                                "type": "object",
                                "properties": {"kind": {"type": "string", "enum": ["after"]}},
                                "required": ["kind"]
                            }
                        ]
                    }
                }
            }),
        );

        let declaration = json_schema_declaration_for(&tool);
        let parameters = &declaration["parametersJsonSchema"];

        assert_eq!(parameters["additionalProperties"], false);
        assert_eq!(parameters["properties"]["retries"]["minimum"], 1);
        assert_eq!(parameters["properties"]["retries"]["maximum"], 5);
        assert_eq!(parameters["properties"]["tags"]["minItems"], 1);
        // A discriminated union survives as a union rather than collapsing.
        assert_eq!(
            parameters["properties"]["schedule"]["oneOf"]
                .as_array()
                .expect("schedule should stay a union")
                .len(),
            2
        );
    }

    #[test]
    fn function_declaration_drops_keywords_outside_the_allowlist() {
        let tool = ToolSpec::new(
            "query",
            "Search",
            serde_json::json!({
                "$schema": "https://json-schema.org/draft/2020-12/schema",
                "type": "object",
                "properties": {"q": {"type": "string", "minLength": 1, "multipleOf": 2}},
                "patternProperties": {"^x-": {"type": "string"}}
            }),
        );

        let declaration = json_schema_declaration_for(&tool);
        let parameters = &declaration["parametersJsonSchema"];

        assert!(parameters.get("$schema").is_none());
        assert!(parameters.get("patternProperties").is_none());
        assert!(parameters["properties"]["q"].get("multipleOf").is_none());
        // On the list, so it is not one of the keywords outside it.
        assert_eq!(parameters["properties"]["q"]["minLength"], 1);
        assert_eq!(parameters["properties"]["q"]["type"], "string");
    }

    #[test]
    fn the_openapi_fallback_field_carries_a_schema_that_field_accepts() {
        // A pre-generation-3 model does not take `parametersJsonSchema`, so
        // the declaration travels in `parameters` — which has no `$ref` field
        // at all and answers one with `Unknown name`, failing the request. The
        // cleaner has to inline the pointer rather than send it.
        let tool = ToolSpec::new(
            "query",
            "Search with a ref",
            serde_json::json!({
                "type": "object",
                "properties": {"filter": {"$ref": "#/$defs/FilterSpec"}},
                "required": ["filter"],
                "$defs": {
                    "FilterSpec": {
                        "type": "object",
                        "properties": {"field": {"type": "string"}}
                    }
                }
            }),
        );

        let declaration = function_declaration_for(&tool, ParametersField::OpenApi);

        assert!(
            declaration.get("parametersJsonSchema").is_none(),
            "the two parameter fields are mutually exclusive"
        );
        let parameters = &declaration["parameters"];
        assert_eq!(parameters["required"], serde_json::json!(["filter"]));
        assert!(
            parameters.get("$defs").is_none(),
            "the OpenAPI subset has no `$defs`"
        );
        let filter = &parameters["properties"]["filter"];
        assert!(
            filter.get("$ref").is_none(),
            "a pointer the OpenAPI subset cannot resolve must be inlined, got {filter}"
        );
        assert_eq!(filter["properties"]["field"]["type"], "string");
    }

    #[test]
    fn a_dirty_tool_schema_is_cleaned_once_per_provider_instance() {
        // Every request declares the whole tool set, so a schema that needs
        // rewriting must not be rebuilt per request the way the unmemoized
        // `SchemaCleanr::clean_shared` would.
        let model_provider = test_model_provider(Some(GeminiAuth::ExplicitKey("key".into())));
        let dirty = Arc::new(serde_json::json!({
            "type": "object",
            "properties": {"q": {"type": "string", "multipleOf": 2}}
        }));

        let first = model_provider
            .schema_cache
            .clean_shared(&dirty, CleaningStrategy::GeminiJsonSchema);
        let second = model_provider
            .schema_cache
            .clean_shared(&dirty, CleaningStrategy::GeminiJsonSchema);

        assert!(
            !Arc::ptr_eq(&dirty, &first),
            "multipleOf is outside the allowlist, so this schema is rewritten"
        );
        assert!(
            Arc::ptr_eq(&first, &second),
            "the second request must reuse the memoized tree, not rebuild it"
        );
    }

    #[test]
    fn a_clean_tool_schema_is_shared_rather_than_copied() {
        let model_provider = test_model_provider(Some(GeminiAuth::ExplicitKey("key".into())));
        let clean = Arc::new(serde_json::json!({
            "type": "object",
            "properties": {"q": {"type": "string"}},
            "required": ["q"]
        }));

        let shared = model_provider
            .schema_cache
            .clean_shared(&clean, CleaningStrategy::GeminiJsonSchema);

        assert!(
            Arc::ptr_eq(&clean, &shared),
            "nothing to rewrite: the registry's own Arc comes straight back"
        );
    }

    #[test]
    fn function_call_response_captures_thought_signature() {
        let json = r#"{
            "candidates": [{
                "content": {
                    "parts": [
                        {"thought": true, "text": "planning..."},
                        {
                            "functionCall": {"id": "gcall-1", "name": "search", "args": {"q": "rust"}},
                            "thoughtSignature": "c2lnbmF0dXJl"
                        }
                    ]
                }
            }]
        }"#;

        let response: GenerateContentResponse = serde_json::from_str(json).unwrap();
        let parts = response
            .candidates
            .unwrap()
            .into_iter()
            .next()
            .unwrap()
            .content
            .unwrap()
            .into_parts();

        assert_eq!(parts.tool_calls.len(), 1);
        let call = &parts.tool_calls[0];
        assert_eq!(call.id, "gcall-1");
        assert_eq!(call.name, "search");
        assert_eq!(
            serde_json::from_str::<serde_json::Value>(&call.arguments).unwrap(),
            serde_json::json!({"q": "rust"})
        );
        assert_eq!(
            call.extra_content,
            Some(serde_json::json!({
                "google": {"thought_signature": "c2lnbmF0dXJl", "id": "gcall-1"}
            }))
        );
        // Reasoning is not an answer when the model asked for a tool — but it
        // is still the reasoning, and history has a field for it.
        assert_eq!(parts.text, None);
        assert_eq!(parts.reasoning.as_deref(), Some("planning..."));
    }

    #[test]
    fn a_thinking_only_turn_reports_its_thinking_once() {
        // With nothing else in the turn the thinking *is* the answer.
        // Reporting it as reasoning as well would hand the caller the same
        // text twice.
        let json = r#"{
            "candidates": [{
                "content": {
                    "parts": [{"thought": true, "text": "planning..."}]
                }
            }]
        }"#;

        let response: GenerateContentResponse = serde_json::from_str(json).unwrap();
        let parts = response
            .candidates
            .unwrap()
            .into_iter()
            .next()
            .unwrap()
            .content
            .unwrap()
            .into_parts();

        assert_eq!(parts.text.as_deref(), Some("planning..."));
        assert_eq!(parts.reasoning, None);
    }

    #[test]
    fn a_nameless_function_call_is_skipped() {
        // Unnamed is unpairable and uncallable. The candidate then holds
        // nothing, which `send_generate_content` reports as an empty
        // response — so the skip is logged at WARN to keep the two apart.
        let json = r#"{
            "candidates": [{
                "content": {
                    "parts": [{"functionCall": {"id": "gcall-1", "args": {}}}]
                }
            }]
        }"#;

        let response: GenerateContentResponse = serde_json::from_str(json).unwrap();
        let parts = response
            .candidates
            .unwrap()
            .into_iter()
            .next()
            .unwrap()
            .content
            .unwrap()
            .into_parts();

        assert!(parts.tool_calls.is_empty());
        assert!(parts.is_empty(), "nothing usable survives the candidate");
    }

    #[test]
    fn function_call_without_id_gets_a_local_call_id() {
        let json = r#"{
            "candidates": [{
                "content": {
                    "parts": [{"functionCall": {"name": "search", "args": {}}}]
                }
            }]
        }"#;

        let response: GenerateContentResponse = serde_json::from_str(json).unwrap();
        let parts = response
            .candidates
            .unwrap()
            .into_iter()
            .next()
            .unwrap()
            .content
            .unwrap()
            .into_parts();

        assert_eq!(parts.tool_calls.len(), 1);
        assert!(parts.tool_calls[0].id.starts_with("call_"));
        assert_eq!(parts.tool_calls[0].extra_content, None);
    }

    #[test]
    fn text_alongside_function_call_is_preserved() {
        let json = r#"{
            "candidates": [{
                "content": {
                    "parts": [
                        {"text": "Looking that up."},
                        {"functionCall": {"name": "search", "args": {}}}
                    ]
                }
            }]
        }"#;

        let response: GenerateContentResponse = serde_json::from_str(json).unwrap();
        let parts = response
            .candidates
            .unwrap()
            .into_iter()
            .next()
            .unwrap()
            .content
            .unwrap()
            .into_parts();

        assert_eq!(parts.text.as_deref(), Some("Looking that up."));
        assert_eq!(parts.tool_calls.len(), 1);
    }

    #[test]
    fn into_text_returns_the_answer_when_there_is_one() {
        let turn = GeminiTurn {
            text: Some("the answer".to_string()),
            reasoning: None,
            tool_calls: Vec::new(),
            usage: None,
        };

        assert_eq!(turn.into_text().unwrap(), "the answer");
    }

    #[test]
    fn into_text_fails_on_a_turn_that_only_asked_for_a_tool() {
        // `chat_with_system` / `chat_with_history` declare no tools, so a turn
        // carrying only a function call holds no answer for them. Reporting an
        // empty string would hand the caller a fabricated reply.
        let turn = GeminiTurn {
            text: None,
            reasoning: None,
            tool_calls: vec![ProviderToolCall {
                id: "call_1".to_string(),
                name: "search".to_string(),
                arguments: "{}".to_string(),
                extra_content: None,
            }],
            usage: None,
        };

        assert!(turn.into_text().is_err());
    }

    /// The function-name set a request built from `names` would declare.
    fn declared(names: &[&str]) -> std::collections::HashSet<String> {
        names.iter().map(ToString::to_string).collect()
    }

    fn assistant_tool_call_history(extra_content: serde_json::Value) -> ChatMessage {
        ChatMessage::assistant(
            serde_json::json!({
                "content": serde_json::Value::Null,
                "tool_calls": [{
                    "id": "call_1",
                    "name": "search",
                    "arguments": "{\"q\":\"rust\"}",
                    "extra_content": extra_content,
                }],
            })
            .to_string(),
        )
    }

    #[test]
    fn assistant_tool_call_history_replays_thought_signature() {
        let messages = vec![
            ChatMessage::user("find something"),
            assistant_tool_call_history(
                serde_json::json!({"google": {"thought_signature": "sig-1", "id": "gcall-1"}}),
            ),
        ];

        let (contents, _) =
            GeminiModelProvider::build_chat_contents(&messages, &declared(&["search"]))
                .expect("user-anchored history must build");

        // The replayed call is the last model turn, so `build_chat_contents`
        // anchors the request with a trailing `[continue]` user turn.
        assert_eq!(contents.len(), 3);
        assert_eq!(contents[1].role.as_deref(), Some("model"));
        assert_eq!(
            serde_json::to_value(&contents[1].parts[0]).unwrap(),
            serde_json::json!({
                "functionCall": {"id": "gcall-1", "name": "search", "args": {"q": "rust"}},
                "thoughtSignature": "sig-1"
            })
        );
    }

    #[test]
    fn replayed_function_call_omits_locally_minted_id() {
        let messages = vec![
            ChatMessage::user("find something"),
            assistant_tool_call_history(
                serde_json::json!({"google": {"thought_signature": "sig-1"}}),
            ),
        ];

        let (contents, _) =
            GeminiModelProvider::build_chat_contents(&messages, &declared(&["search"]))
                .expect("user-anchored history must build");

        let part = serde_json::to_value(&contents[1].parts[0]).unwrap();
        assert!(
            part["functionCall"].get("id").is_none(),
            "ZeroClaw-minted call ids must not reach the Gemini API"
        );
    }

    #[test]
    fn tool_result_becomes_a_function_response_named_after_its_call() {
        let messages = vec![
            ChatMessage::user("find something"),
            assistant_tool_call_history(
                serde_json::json!({"google": {"thought_signature": "sig-1", "id": "gcall-1"}}),
            ),
            ChatMessage::tool(
                serde_json::json!({"tool_call_id": "call_1", "content": "42"}).to_string(),
            ),
        ];

        let (contents, _) =
            GeminiModelProvider::build_chat_contents(&messages, &declared(&["search"]))
                .expect("user-anchored history must build");

        assert_eq!(contents.len(), 3);
        assert_eq!(contents[2].role.as_deref(), Some("user"));
        assert_eq!(
            serde_json::to_value(&contents[2].parts[0]).unwrap(),
            serde_json::json!({
                "functionResponse": {
                    "id": "gcall-1",
                    "name": "search",
                    "response": {"output": "42"}
                }
            })
        );
    }

    #[test]
    fn parallel_tool_results_share_one_turn() {
        let assistant = ChatMessage::assistant(
            serde_json::json!({
                "content": serde_json::Value::Null,
                "tool_calls": [
                    {"id": "call_1", "name": "search", "arguments": "{}"},
                    {"id": "call_2", "name": "fetch", "arguments": "{}"},
                ],
            })
            .to_string(),
        );
        let messages = vec![
            ChatMessage::user("go"),
            assistant,
            ChatMessage::tool(
                serde_json::json!({"tool_call_id": "call_1", "content": "a"}).to_string(),
            ),
            ChatMessage::tool(
                serde_json::json!({"tool_call_id": "call_2", "content": "b"}).to_string(),
            ),
        ];

        let (contents, _) =
            GeminiModelProvider::build_chat_contents(&messages, &declared(&["search", "fetch"]))
                .expect("user-anchored history must build");

        assert_eq!(contents.len(), 3);
        assert_eq!(contents[2].parts.len(), 2);
        assert_eq!(
            serde_json::to_value(&contents[2].parts[1]).unwrap()["functionResponse"]["name"],
            "fetch"
        );
    }

    #[test]
    fn tool_result_images_trail_as_a_separate_user_turn() {
        let messages = vec![
            ChatMessage::user("screenshot it"),
            assistant_tool_call_history(serde_json::Value::Null),
            ChatMessage::tool(
                serde_json::json!({
                    "tool_call_id": "call_1",
                    "content": "captured [IMAGE:data:image/png;base64,AA==]"
                })
                .to_string(),
            ),
        ];

        let (contents, _) =
            GeminiModelProvider::build_chat_contents(&messages, &declared(&["search"]))
                .expect("user-anchored history must build");

        assert_eq!(contents.len(), 4);
        assert_eq!(
            serde_json::to_value(&contents[2].parts[0]).unwrap()["functionResponse"]["response"]
                ["output"],
            "captured"
        );
        assert_eq!(contents[3].role.as_deref(), Some("user"));
        assert!(matches!(&contents[3].parts[0], Part::Inline { .. }));
    }

    #[test]
    fn tool_result_without_a_matching_call_answers_the_only_outstanding_one() {
        // One call outstanding, so elimination is not a guess.
        let messages = vec![
            assistant_tool_call_history(serde_json::Value::Null),
            ChatMessage::tool(
                serde_json::json!({"tool_call_id": "unknown", "content": "42"}).to_string(),
            ),
        ];

        let (contents, _) =
            GeminiModelProvider::build_chat_contents(&messages, &declared(&["search"]))
                .expect("user-anchored history must build");

        assert_eq!(
            serde_json::to_value(&contents[1].parts[0]).unwrap()["functionResponse"]["name"],
            "search"
        );
    }

    /// A user turn plus one assistant turn calling `search` and `fetch` in
    /// parallel.
    fn parallel_calls_prefix() -> Vec<ChatMessage> {
        vec![
            ChatMessage::user("go"),
            ChatMessage::assistant(
                serde_json::json!({
                    "content": serde_json::Value::Null,
                    "tool_calls": [
                        {"id": "call_1", "name": "search", "arguments": "{}"},
                        {"id": "call_2", "name": "fetch", "arguments": "{}"},
                    ],
                })
                .to_string(),
            ),
        ]
    }

    /// A tool result whose `tool_call_id` matches no call in the transcript.
    fn unmatched_tool_result(output: &str) -> ChatMessage {
        ChatMessage::tool(
            serde_json::json!({"tool_call_id": "lost", "content": output}).to_string(),
        )
    }

    #[test]
    fn an_unattributable_result_degrades_rather_than_naming_the_wrong_function() {
        // `search`'s result lost its id. The old fallback pinned it on the
        // most recent call — the model would read `fetch` as having produced
        // search's output, while `search` went unanswered anyway.
        let mut messages = parallel_calls_prefix();
        messages.push(unmatched_tool_result("a"));

        let (contents, _) =
            GeminiModelProvider::build_chat_contents(&messages, &declared(&["search", "fetch"]))
                .expect("user-anchored history must build");

        assert!(
            !contents
                .iter()
                .flat_map(|content| &content.parts)
                .any(|part| matches!(part, Part::FunctionResponse { .. })),
            "with two calls outstanding there is nothing to attribute the result to"
        );
        assert!(
            matches!(&contents[2].parts[0], Part::Text { text } if text == "(tool output)\na"),
            "got {:?}",
            contents[2].parts[0]
        );
    }

    #[test]
    fn elimination_narrows_as_matched_results_arrive() {
        // `search` answers itself by id, which leaves `fetch` as the only
        // outstanding call — so the second result, id and all lost, is no
        // longer ambiguous.
        let mut messages = parallel_calls_prefix();
        messages.push(ChatMessage::tool(
            serde_json::json!({"tool_call_id": "call_1", "content": "a"}).to_string(),
        ));
        messages.push(unmatched_tool_result("b"));

        let (contents, _) =
            GeminiModelProvider::build_chat_contents(&messages, &declared(&["search", "fetch"]))
                .expect("user-anchored history must build");

        assert_eq!(contents.len(), 3, "both results answer in one turn");
        assert_eq!(contents[2].parts.len(), 2);
        let responses = serde_json::to_value(&contents[2].parts).unwrap();
        assert_eq!(responses[0]["functionResponse"]["name"], "search");
        assert_eq!(responses[1]["functionResponse"]["name"], "fetch");
        assert_eq!(responses[1]["functionResponse"]["response"]["output"], "b");
    }

    #[test]
    fn a_repeated_result_does_not_answer_the_same_call_twice() {
        // A retried tool round, or a session row replayed after a partial
        // commit: the same `tool_call_id` arrives twice. The second must not
        // pair with the call again — Gemini takes one `functionResponse` per
        // `functionCall` — nor be handed to a different call by elimination.
        let messages = vec![
            assistant_tool_call_history(serde_json::Value::Null),
            ChatMessage::tool(
                serde_json::json!({"tool_call_id": "call_1", "content": "a"}).to_string(),
            ),
            ChatMessage::tool(
                serde_json::json!({"tool_call_id": "call_1", "content": "b"}).to_string(),
            ),
        ];

        let (contents, _) =
            GeminiModelProvider::build_chat_contents(&messages, &declared(&["search"]))
                .expect("user-anchored history must build");

        let responses = contents
            .iter()
            .flat_map(|content| &content.parts)
            .filter(|part| matches!(part, Part::FunctionResponse { .. }))
            .count();
        assert_eq!(responses, 1, "one call, one response");
        assert_eq!(contents.len(), 3);
        assert!(
            matches!(&contents[2].parts[0], Part::Text { text } if text == "(tool output)\nb"),
            "the repeat degrades to text, got {:?}",
            contents[2].parts[0]
        );
    }

    #[test]
    fn an_answered_call_stops_being_a_candidate_for_elimination() {
        // One call, two results. The first takes it by elimination; the
        // second has nothing left to answer and must not re-use it.
        let messages = vec![
            assistant_tool_call_history(serde_json::Value::Null),
            unmatched_tool_result("a"),
            unmatched_tool_result("b"),
        ];

        let (contents, _) =
            GeminiModelProvider::build_chat_contents(&messages, &declared(&["search"]))
                .expect("user-anchored history must build");

        assert_eq!(contents.len(), 3);
        assert_eq!(
            serde_json::to_value(&contents[1].parts[0]).unwrap()["functionResponse"]["response"]
                ["output"],
            "a"
        );
        assert!(
            matches!(&contents[2].parts[0], Part::Text { text } if text == "(tool output)\nb"),
            "got {:?}",
            contents[2].parts[0]
        );
    }

    #[test]
    fn orphan_tool_result_becomes_user_text_rather_than_an_undeclared_function() {
        // No assistant turn to recover a function name from. A
        // `functionResponse` would have to name something the request never
        // declared, which Gemini rejects — so the output travels as text.
        let messages = vec![ChatMessage::tool(
            serde_json::json!({"tool_call_id": "unknown", "content": "42"}).to_string(),
        )];

        let (contents, _) =
            GeminiModelProvider::build_chat_contents(&messages, &declared(&["search"]))
                .expect("user-anchored history must build");

        assert_eq!(contents.len(), 1);
        assert_eq!(contents[0].role.as_deref(), Some("user"));
        assert!(
            matches!(&contents[0].parts[0], Part::Text { text } if text == "(tool output)\n42"),
            "orphan result should carry its output as text, got {:?}",
            contents[0].parts[0]
        );
    }

    #[test]
    fn orphan_tool_result_images_still_trail_as_their_own_turn() {
        let messages = vec![
            ChatMessage::tool(
                serde_json::json!({
                    "tool_call_id": "unknown",
                    "content": "captured [IMAGE:data:image/png;base64,AA==]"
                })
                .to_string(),
            ),
            ChatMessage::user("what is it?"),
        ];

        let (contents, _) =
            GeminiModelProvider::build_chat_contents(&messages, &declared(&["search"]))
                .expect("user-anchored history must build");

        assert_eq!(contents.len(), 3);
        assert!(
            matches!(&contents[0].parts[0], Part::Text { text } if text == "(tool output)\ncaptured")
        );
        assert!(matches!(&contents[1].parts[0], Part::Inline { .. }));
        assert!(matches!(&contents[2].parts[0], Part::Text { text } if text == "what is it?"));
    }

    #[test]
    fn plain_assistant_prose_is_not_read_as_a_tool_call_envelope() {
        let messages = vec![
            ChatMessage::user("say something"),
            ChatMessage::assistant(r#"{"content": "just JSON"}"#),
        ];

        let (contents, _) = GeminiModelProvider::build_chat_contents(&messages, &declared(&[]))
            .expect("user-anchored history must build");

        assert_eq!(contents[1].role.as_deref(), Some("model"));
        assert!(
            matches!(&contents[1].parts[0], Part::Text { text } if text == r#"{"content": "just JSON"}"#)
        );
    }

    // ── Tools the request no longer declares ─────────────────────────────

    /// Whether anything in `contents` names a function on the wire.
    fn has_function_part(contents: &[Content]) -> bool {
        contents
            .iter()
            .flat_map(|content| &content.parts)
            .any(|part| matches!(part, Part::FunctionCall { .. } | Part::FunctionResponse { .. }))
    }

    #[test]
    fn declared_function_names_reads_the_wire_payload() {
        let tools = vec![ToolDeclarations {
            function_declarations: vec![
                serde_json::json!({"name": "search"}),
                serde_json::json!({"description": "nameless"}),
            ],
        }];

        assert_eq!(
            declared_function_names(Some(tools.as_slice())),
            declared(&["search"])
        );
        assert!(declared_function_names(None).is_empty());
    }

    #[test]
    fn undeclared_call_replays_as_narration_instead_of_a_function_call() {
        // History outlives a tool declaration — an MCP server that did not
        // come back up, a tool dropped from config, a restored session. A
        // `functionCall` naming it is rejected outright, and would be rejected
        // again on every later turn.
        let messages = vec![
            ChatMessage::user("find something"),
            assistant_tool_call_history(
                serde_json::json!({"google": {"thought_signature": "sig-1", "id": "gcall-1"}}),
            ),
        ];

        let (contents, _) = GeminiModelProvider::build_chat_contents(&messages, &declared(&[]))
            .expect("user-anchored history must build");

        assert_eq!(contents.len(), 3, "the narrated call gains a `[continue]` anchor");
        assert_eq!(contents[1].role.as_deref(), Some("model"));
        let Part::Text { text } = &contents[1].parts[0] else {
            panic!("undeclared call should narrate, got {:?}", contents[1].parts[0]);
        };
        assert!(text.contains("search"), "got {text}");
        assert!(text.contains(r#"{"q":"rust"}"#), "got {text}");
        // The narration must not hand the model a protocol this request never
        // declared: it would imitate the shape, the native dispatcher would
        // not execute it, and the orchestrator's leak guard would swallow the
        // whole reply. That guard flags a tagged envelope or a bare JSON
        // object, so the narration must read as neither.
        assert!(!text.contains("<tool_call>"), "got {text}");
        assert!(
            serde_json::from_str::<serde_json::Value>(text.trim()).is_err(),
            "narration must not read as a tool-call envelope, got {text}"
        );
        assert!(!has_function_part(&contents));
    }

    #[test]
    fn undeclared_tool_result_degrades_alongside_its_call() {
        let messages = vec![
            ChatMessage::user("find something"),
            assistant_tool_call_history(serde_json::Value::Null),
            ChatMessage::tool(
                serde_json::json!({"tool_call_id": "call_1", "content": "42"}).to_string(),
            ),
        ];

        let (contents, _) = GeminiModelProvider::build_chat_contents(&messages, &declared(&[]))
            .expect("user-anchored history must build");

        // Both sides of the exchange always degrade together: a narrated call
        // is never answered by a `functionResponse`, and vice versa.
        assert_eq!(contents.len(), 3);
        assert_eq!(contents[1].role.as_deref(), Some("model"));
        assert_eq!(contents[2].role.as_deref(), Some("user"));
        assert!(
            matches!(&contents[2].parts[0], Part::Text { text } if text == "(tool output)\n42"),
            "got {:?}",
            contents[2].parts[0]
        );
        assert!(!has_function_part(&contents));
    }

    #[test]
    fn partially_declared_batch_keeps_function_responses_in_one_turn() {
        // `fetch` is gone, `search` survives. The degraded output must not
        // land between the `functionResponse` parts and split the answer to
        // one model turn across three user turns.
        let assistant = ChatMessage::assistant(
            serde_json::json!({
                "content": serde_json::Value::Null,
                "tool_calls": [
                    {"id": "call_1", "name": "search", "arguments": "{}"},
                    {"id": "call_2", "name": "fetch", "arguments": "{}"},
                    {"id": "call_3", "name": "search", "arguments": "{}"},
                ],
            })
            .to_string(),
        );
        let messages = vec![
            ChatMessage::user("go"),
            assistant,
            ChatMessage::tool(
                serde_json::json!({"tool_call_id": "call_1", "content": "a"}).to_string(),
            ),
            ChatMessage::tool(
                serde_json::json!({"tool_call_id": "call_2", "content": "b"}).to_string(),
            ),
            ChatMessage::tool(
                serde_json::json!({"tool_call_id": "call_3", "content": "c"}).to_string(),
            ),
        ];

        let (contents, _) =
            GeminiModelProvider::build_chat_contents(&messages, &declared(&["search"]))
                .expect("user-anchored history must build");

        // user, model, the two surviving results, then the degraded one.
        assert_eq!(contents.len(), 4);
        assert_eq!(contents[1].parts.len(), 3, "the dropped call is narrated in place");
        assert_eq!(contents[2].parts.len(), 2);
        assert!(
            contents[2]
                .parts
                .iter()
                .all(|part| matches!(part, Part::FunctionResponse { .. })),
            "both surviving results answer in one contiguous turn"
        );
        assert!(
            matches!(&contents[3].parts[0], Part::Text { text } if text == "(tool output)\nb"),
            "got {:?}",
            contents[3].parts[0]
        );
    }

    // ── `chat` assembly, short of the transport ──────────────────────────

    /// A one-argument tool, so every declaration has parameters to send.
    fn tool_named(name: &str) -> ToolSpec {
        ToolSpec::new(
            name,
            "Look something up",
            serde_json::json!({
                "type": "object",
                "properties": {"q": {"type": "string"}},
                "required": ["q"]
            }),
        )
    }

    #[test]
    fn chat_declares_tools_through_the_field_its_model_accepts() {
        // `convert_tools` has no model and always falls back to `parameters`;
        // `chat` knows the model, and generation 3 must get the JSON Schema
        // field while older generations keep the one they accept.
        let model_provider = test_model_provider(Some(GeminiAuth::ExplicitKey("key".into())));
        let messages = vec![ChatMessage::user("find something")];
        let tools = [tool_named("search")];

        for (model, sent, withheld) in [
            ("gemini-3-pro-preview", "parametersJsonSchema", "parameters"),
            ("gemini-2.5-flash", "parameters", "parametersJsonSchema"),
        ] {
            let prepared = model_provider
                .prepare_chat(&messages, Some(&tools[..]), model)
                .expect("user-anchored history must build");

            let groups = prepared.tools.expect("a non-empty tool list is declared");
            assert_eq!(groups.len(), 1, "{model}: every function rides one entry");
            let declaration = &groups[0].function_declarations[0];
            assert_eq!(declaration["name"], "search", "{model}");
            assert_eq!(
                declaration[sent]["required"],
                serde_json::json!(["q"]),
                "{model} must declare its arguments through {sent}"
            );
            assert!(
                declaration.get(withheld).is_none(),
                "{model} must not send {withheld}"
            );
        }
    }

    #[test]
    fn chat_sends_no_tools_block_without_tools() {
        let model_provider = test_model_provider(Some(GeminiAuth::ExplicitKey("key".into())));
        let messages = vec![ChatMessage::user("hello")];
        let no_tools: [ToolSpec; 0] = [];

        for (label, tools) in [("absent", None), ("empty", Some(&no_tools[..]))] {
            let prepared = model_provider
                .prepare_chat(&messages, tools, "gemini-3-pro-preview")
                .expect("user-anchored history must build");
            assert!(
                prepared.tools.is_none(),
                "{label} tool list declared a block"
            );
        }
    }

    #[test]
    fn chat_renders_history_against_the_tools_it_declares() {
        // The declared set is read off the payload `chat` built. A call to a
        // tool the request still declares replays natively; one it no longer
        // declares degrades to text rather than failing the whole request.
        let model_provider = test_model_provider(Some(GeminiAuth::ExplicitKey("key".into())));
        let messages = vec![
            ChatMessage::user("find something"),
            assistant_tool_call_history(
                serde_json::json!({"google": {"thought_signature": "sig-1"}}),
            ),
            ChatMessage::tool(
                serde_json::json!({"tool_call_id": "call_1", "content": "42"}).to_string(),
            ),
        ];

        let search = [tool_named("search")];
        let prepared = model_provider
            .prepare_chat(&messages, Some(&search[..]), "gemini-3-pro-preview")
            .expect("user-anchored history must build");
        assert_eq!(
            serde_json::to_value(&prepared.contents[1].parts[0]).unwrap()["thoughtSignature"],
            "sig-1",
            "a still-declared call must replay natively with its signature"
        );
        assert!(has_function_part(&prepared.contents));

        let only_fetch = [tool_named("fetch")];
        for (label, tools) in [("fetch only", Some(&only_fetch[..])), ("none", None)] {
            let prepared = model_provider
                .prepare_chat(&messages, tools, "gemini-3-pro-preview")
                .expect("user-anchored history must build");
            assert!(
                !has_function_part(&prepared.contents),
                "{label}: an undeclared call must not travel as a function part"
            );
        }
    }

    fn turn_from(json: &str) -> anyhow::Result<GeminiTurn> {
        GeminiTurn::from_response(serde_json::from_str(json).expect("response JSON parses"))
    }

    #[test]
    fn a_tool_call_only_response_reaches_the_agent_loop_intact() {
        // Before native tool calling a candidate with no text was "No response
        // from Gemini". A turn that only asks for a tool is now complete, and
        // `chat` has to hand on its calls, reasoning, and usage.
        let turn = turn_from(
            r#"{
                "candidates": [{
                    "content": {
                        "parts": [
                            {"thought": true, "text": "planning..."},
                            {
                                "functionCall": {"name": "search", "args": {"q": "rust"}},
                                "thoughtSignature": "c2lnbmF0dXJl"
                            }
                        ]
                    }
                }],
                "usageMetadata": {"promptTokenCount": 12, "candidatesTokenCount": 3}
            }"#,
        )
        .expect("a function call is a complete turn");

        let response = turn.into_chat_response();

        assert_eq!(response.text, None);
        assert_eq!(response.reasoning_content.as_deref(), Some("planning..."));
        assert_eq!(response.tool_calls.len(), 1);
        assert_eq!(response.tool_calls[0].name, "search");
        assert_eq!(
            response.tool_calls[0].extra_content,
            Some(serde_json::json!({"google": {"thought_signature": "c2lnbmF0dXJl"}})),
            "the signature must reach the agent loop to be replayed next turn"
        );
        let usage = response.usage.expect("usage is reported");
        assert_eq!(usage.input_tokens, Some(12));
        assert_eq!(usage.output_tokens, Some(3));
    }

    #[test]
    fn a_wrapped_internal_response_is_unwrapped_before_its_parts_are_read() {
        let turn = turn_from(
            r#"{
                "response": {
                    "candidates": [{
                        "content": {"parts": [{"functionCall": {"name": "search", "args": {}}}]}
                    }]
                },
                "usageMetadata": {"promptTokenCount": 5}
            }"#,
        )
        .expect("the cloudcode-pa envelope carries a real turn");

        assert_eq!(turn.tool_calls.len(), 1);
        assert_eq!(turn.usage.and_then(|usage| usage.input_tokens), Some(5));
    }

    #[test]
    fn a_candidate_with_nothing_usable_is_still_no_response() {
        for json in [
            r#"{"candidates": []}"#,
            r#"{"candidates": [{"content": {"parts": []}}]}"#,
            // A nameless call is dropped, which leaves nothing behind.
            r#"{"candidates": [{"content": {"parts": [{"functionCall": {"args": {}}}]}}]}"#,
        ] {
            let Err(err) = turn_from(json) else {
                panic!("{json} must not pass for a turn");
            };
            assert!(err.to_string().contains("No response from Gemini"), "{err}");
        }
    }

    #[test]
    fn an_api_error_body_fails_the_turn_on_either_layer() {
        for json in [
            r#"{"error": {"message": "quota exhausted"}}"#,
            r#"{"response": {"error": {"message": "quota exhausted"}}}"#,
        ] {
            let Err(err) = turn_from(json) else {
                panic!("{json} must fail");
            };
            assert!(err.to_string().contains("quota exhausted"), "{err}");
        }
    }

    #[test]
    fn request_serializes_function_declarations() {
        let request = GenerateContentRequest {
            contents: vec![Content {
                role: Some("user".to_string()),
                parts: vec![Part::text("Hello")],
            }],
            system_instruction: None,
            tools: Some(vec![ToolDeclarations {
                function_declarations: vec![serde_json::json!({"name": "search"})],
            }]),
            generation_config: GenerationConfig {
                temperature: None,
                max_output_tokens: 8192,
            },
        };

        let json = serde_json::to_value(&request).unwrap();
        assert_eq!(json["tools"][0]["functionDeclarations"][0]["name"], "search");
    }

    #[test]
    fn internal_request_forwards_tools() {
        let model_provider = test_model_provider(Some(test_oauth_auth("ya29.mock-token")));
        let auth = test_oauth_auth("ya29.mock-token");
        let url = GeminiModelProvider::build_generate_content_url("gemini-3-pro-preview", &auth);
        let body = GenerateContentRequest {
            contents: vec![Content {
                role: Some("user".into()),
                parts: vec![Part::text("hello")],
            }],
            system_instruction: None,
            tools: Some(vec![ToolDeclarations {
                function_declarations: vec![serde_json::json!({"name": "search"})],
            }]),
            generation_config: GenerationConfig {
                temperature: None,
                max_output_tokens: 8192,
            },
        };

        let request = model_provider
            .build_generate_content_request(
                &auth,
                &url,
                &body,
                "gemini-3-pro-preview",
                true,
                Some("test-project"),
                Some("ya29.mock-token"),
            )
            .unwrap()
            .build()
            .unwrap();

        let payload = request
            .body()
            .and_then(|b| b.as_bytes())
            .expect("json request body should be bytes");
        let json: serde_json::Value = serde_json::from_slice(payload).unwrap();

        assert_eq!(
            json["request"]["tools"][0]["functionDeclarations"][0]["name"],
            "search"
        );
    }
}
