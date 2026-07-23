//! JSON Schema cleaning and validation for LLM tool-calling compatibility.
//!
//! Different model_providers support different subsets of JSON Schema. This module
//! normalizes tool schemas to improve cross-provider compatibility while
//! preserving semantic intent.
//!
//! ## What this module does
//!
//! 1. Removes unsupported keywords per model_provider strategy — by allowlist
//!    for both Gemini parameter fields, whose backends reject or ignore what
//!    they do not declare, and by denylist for the rest
//! 2. Resolves local `$ref` entries from `$defs` and `definitions`, except
//!    where the target API resolves them itself
//! 3. Flattens literal `anyOf` / `oneOf` unions into `enum`
//! 4. Folds `allOf` into the object that carries it, and rewrites `oneOf` to
//!    `anyOf`, for targets whose dialect has no such keyword
//! 5. Strips nullable variants from unions, for the targets that need it,
//!    and `null` from `type` arrays, which no target takes as written
//! 6. Converts `const` to single-value `enum`
//! 7. Detects circular references and stops recursion safely
//!
//! # Example
//!
//! ```rust
//! use serde_json::json;
//! use zeroclaw_api::schema::SchemaCleanr;
//!
//! let dirty_schema = json!({
//!     "type": "object",
//!     "properties": {
//!         "name": {
//!             "type": "string",
//!             "minLength": 1, // Kept: `Schema` declares it
//!             "multipleOf": 2 // Dropped: it does not
//!         },
//!         "age": {
//!             "$ref": "#/$defs/Age" // Needs resolution
//!         }
//!     },
//!     "$defs": {
//!         "Age": {
//!             "type": "integer",
//!             "minimum": 0
//!         }
//!     }
//! });
//!
//! let cleaned = SchemaCleanr::clean_for_gemini(dirty_schema);
//!
//! // Result:
//! // {
//! // "type": "object",
//! // "properties": {
//! // "name": { "type": "string", "minLength": 1 },
//! // "age": { "type": "integer", "minimum": 0 }
//! // }
//! // }
//! ```
//!
use serde_json::{Map, Value, json};
use std::collections::{BTreeSet, HashMap, HashSet};
use std::sync::Arc;

/// Keywords Gemini accepts in `FunctionDeclaration.parameters`, whose
/// `Schema` is a select subset of OpenAPI 3.0.
///
/// An allowlist, because the field is closed and the failure is loud: a
/// keyword `Schema` does not declare comes back as `Invalid JSON payload
/// received. Unknown name "…"`, failing the whole request and taking every
/// other tool declared alongside it down too. JSON Schema's vocabulary is
/// open-ended, so a denylist cannot be complete — it would keep failing that
/// way for whichever keyword nobody had thought of yet.
///
/// The contents are the `Schema` field set from the `generativelanguage`
/// v1beta discovery document, which v1 matches exactly, plus two keywords the
/// cleaner rewrites rather than sends and which therefore have to survive
/// this filter to reach their rewrite:
///
/// - `const` becomes a single-value `enum`, which `Schema` does declare.
/// - `oneOf` becomes `anyOf` — see [`CleaningStrategy::rewrites_one_of`].
///
/// `allOf` needs no entry either way: [`CleaningStrategy::merges_all_of`]
/// folds it into its parent before this filter runs.
///
/// Compare [`GEMINI_JSON_SCHEMA_KEYWORDS`]. Neither list contains the other:
/// that one carries the `$ref`, `$defs` and `additionalProperties` this field
/// has nowhere to put, and leaves out `nullable` and `example`, which are
/// OpenAPI spellings with no place in JSON Schema.
pub const GEMINI_SCHEMA_KEYWORDS: &[&str] = &[
    "type",
    "format",
    "title",
    "description",
    "nullable",
    "default",
    "example",
    "enum",
    "items",
    "properties",
    "required",
    "propertyOrdering",
    "anyOf",
    // Value and size bounds.
    "minLength",
    "maxLength",
    "pattern",
    "minimum",
    "maximum",
    "minItems",
    "maxItems",
    "minProperties",
    "maxProperties",
    // Rewritten by the cleaner rather than passed through, so each has to
    // survive this filter to reach the rewrite that removes it.
    "const",
    "oneOf",
];

/// Keywords Gemini accepts in the `parametersJsonSchema` /
/// `responseJsonSchema` fields, which take JSON Schema rather than the
/// OpenAPI-3.0 `Schema` subset that `parameters` is limited to.
///
/// This is an allowlist rather than a denylist, but not for the reason the
/// other one is. This field takes an unconstrained JSON value on the wire and
/// its backend ignores what it does not recognise, so nothing here fails
/// loudly — an unsupported keyword travels, does nothing, and says nothing.
/// The list is therefore what has been shown to reach the model rather than
/// what survives validation, since validation accepts everything.
///
/// What that costs, for the keywords tool authors actually reach for:
///
/// - `multipleOf` / `uniqueItems` — the model stops seeing the constraint and
///   can propose a value the tool then rejects. Neither has been measured
///   either way; they are out because nothing shows them getting through.
/// - `allOf` — folded into its parent before this filter runs rather than
///   dropped, so a property whose only content is an `allOf` (what schemars
///   and pydantic emit for a `$ref` with sibling keywords) keeps its shape.
///   Only a branch keyword contradicting one already on the parent is lost.
///   See [`CleaningStrategy::merges_all_of`].
/// - `example` / `examples` / `$comment` / `$schema` — documentation only.
///
/// [`SchemaCleanr::dropped_keywords`] reports what a given schema loses, so
/// this never has to be silent at the call site.
pub const GEMINI_JSON_SCHEMA_KEYWORDS: &[&str] = &[
    // Documented as supported by Google.
    "$id",
    "$defs",
    "$ref",
    "$anchor",
    "type",
    "format",
    "title",
    "description",
    "enum",
    "items",
    "prefixItems",
    "minItems",
    "maxItems",
    "minimum",
    "maximum",
    "anyOf",
    "oneOf",
    "properties",
    "additionalProperties",
    "required",
    "propertyOrdering",
    // Not in Google's published list, but each measured to reach the model:
    // pin one so that a single answer satisfies it, force a call with a
    // prompt that names no value, and the argument comes back conforming.
    "pattern",
    "minLength",
    "maxLength",
    "minProperties",
    "maxProperties",
    // Rewritten to `enum` by the cleaner rather than passed through, so it
    // has to survive the keyword filter to reach that rewrite.
    "const",
    // Draft-07 spelling of `$defs`. Undocumented, but dropping it while
    // `$ref` survives would leave every draft-07 pointer dangling.
    "definitions",
    // Not in Google's list, but already carried by the stricter `parameters`
    // path without complaint, and it tells the model real information.
    "default",
];

/// Keywords whose value maps caller-chosen *names* to subschemas. Their keys
/// are never schema keywords, so the cleaner recurses into the values while
/// leaving the names alone — a property named `pattern`, or a definition named
/// `Age`, must survive.
const NAME_MAP_SCHEMA_KEYS: &[&str] = &[
    "properties",
    "$defs",
    "definitions",
    "patternProperties",
    "dependentSchemas",
];

/// Keywords whose value is instance data rather than a subschema. The cleaner
/// copies these verbatim so a value's own keys are never mistaken for schema
/// keywords — `"default": {"format": "json"}` must keep its `format` entry.
const OPAQUE_SCHEMA_KEYS: &[&str] = &[
    "default",
    "enum",
    "example",
    "examples",
    "required",
    "propertyOrdering",
];

/// Keywords that should be preserved during cleaning (metadata).
const SCHEMA_META_KEYS: &[&str] = &["description", "title", "default"];

/// Schema cleaning strategies for different LLM model_providers.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum CleaningStrategy {
    /// Gemini `FunctionDeclaration.parameters`, the OpenAPI 3.0 subset every
    /// generation accepts. Filters by allowlist: this field rejects a keyword
    /// it does not declare rather than ignoring it.
    Gemini,
    /// Gemini `FunctionDeclaration.parametersJsonSchema` - keeps the `$ref`,
    /// `$defs`, and `additionalProperties` that [`Self::Gemini`] has to
    /// strip, and every constraint that one carries.
    GeminiJsonSchema,
    /// Anthropic Claude - Moderately permissive
    Anthropic,
    /// OpenAI GPT - Most permissive
    OpenAI,
    /// Conservative: Remove only universally unsupported keywords
    Conservative,
}

impl CleaningStrategy {
    /// Get the list of unsupported keywords for this strategy.
    ///
    /// Empty for allowlist strategies — see [`Self::supported_keywords`].
    pub fn unsupported_keywords(self) -> &'static [&'static str] {
        match self {
            // Allowlist-driven; see `supported_keywords`.
            Self::Gemini | Self::GeminiJsonSchema => &[],
            Self::Anthropic => &["$ref", "$defs", "definitions"], // Anthropic doesn't resolve refs
            Self::OpenAI => &[], // OpenAI is most permissive
            Self::Conservative => &["$ref", "$defs", "definitions", "additionalProperties"],
        }
    }

    /// The keywords this strategy keeps, when it filters by allowlist.
    /// `None` for strategies that filter by denylist instead.
    pub fn supported_keywords(self) -> Option<&'static [&'static str]> {
        match self {
            Self::Gemini => Some(GEMINI_SCHEMA_KEYWORDS),
            Self::GeminiJsonSchema => Some(GEMINI_JSON_SCHEMA_KEYWORDS),
            _ => None,
        }
    }

    /// Whether `key` survives cleaning under this strategy.
    fn retains_keyword(self, key: &str) -> bool {
        match self.supported_keywords() {
            Some(allowed) => allowed.contains(&key),
            None => !self.unsupported_keywords().contains(&key),
        }
    }

    /// Whether local `$ref` pointers must be inlined from `$defs`.
    ///
    /// Only false where the target API resolves references itself; inlining
    /// there would discard `$defs` for no gain and break recursive schemas.
    ///
    /// The two Gemini fields sit on opposite sides of that line. `parameters`
    /// answers a `$ref` with `Unknown name` and fails the whole request;
    /// `parametersJsonSchema` resolves it, and the model reads what it points
    /// at. Contrast [`Self::merges_all_of`], where the fields agree.
    fn resolves_refs(self) -> bool {
        !matches!(self, Self::GeminiJsonSchema)
    }

    /// Whether `allOf` must be folded into the object that carries it rather
    /// than passed through.
    ///
    /// Both Gemini parameter fields accept the keyword — it passes wire
    /// validation — and the model then never reads what is inside it. Put an
    /// `enum` where only an `allOf` branch leads to it, force a call with a
    /// prompt that names no value, and the argument comes back invented; the
    /// same `enum` reached through `anyOf` comes back exactly. That holds for
    /// `parameters` and `parametersJsonSchema` alike.
    ///
    /// So the choice is not between sending and being rejected. It is between
    /// folding the branch into its parent, where the model reads it, and
    /// sending composition that is inert — which is worse than a rejection,
    /// because nothing reports it.
    ///
    /// It also rules out the obvious-looking alternative of adding `allOf` to
    /// [`GEMINI_JSON_SCHEMA_KEYWORDS`]: passing it through untouched is
    /// precisely the case measured as dead.
    fn merges_all_of(self) -> bool {
        matches!(self, Self::Gemini | Self::GeminiJsonSchema)
    }

    /// Whether `oneOf` must be rewritten to `anyOf`.
    ///
    /// `parameters` accepts `oneOf` and the model does not read it — the same
    /// valid-but-dead shape described on [`Self::merges_all_of`] — while it
    /// does read `anyOf`, so the branch list survives the rename. The two
    /// differ in exclusivity, which a generating model has no way to act on
    /// anyway.
    ///
    /// `parametersJsonSchema` reads `oneOf` as written, which is why this is
    /// not simply always true: rewriting there would give up the exclusivity
    /// for nothing.
    fn rewrites_one_of(self) -> bool {
        matches!(self, Self::Gemini)
    }

    /// Whether a `{"type": "null"}` branch has to come out of a union.
    ///
    /// Off for both Gemini fields, which take the branch and read the union
    /// around it: `Schema`'s `type` enum carries a NULL member, and a union
    /// whose non-null branch holds an `enum` comes back answered on either
    /// field. Dropping the branch would only take away the caller's way of
    /// saying the argument is optional.
    ///
    /// Left on elsewhere, where it is long-standing behaviour that has not
    /// been measured rather than known to be needed. Those targets take full
    /// JSON Schema, so it is probably unnecessary there too, but that is a
    /// separate change needing its own evidence.
    ///
    /// Not gated on this, and not the same question: `null` comes out of a
    /// `type` *array* for every target, because `Schema.type` is a single
    /// string that answers an array with `Unknown name`, and nothing shows
    /// the other field reading one.
    fn drops_null_variants(self) -> bool {
        !matches!(self, Self::Gemini | Self::GeminiJsonSchema)
    }

    /// Whether a `type` sitting beside a union has to give way to it.
    ///
    /// Off for `parametersJsonSchema`, which is read as written: keeping both
    /// costs nothing and the `type` is a constraint the field would carry.
    ///
    /// Left on for the OpenAPI subset, where the two cannot obviously be
    /// combined and the union is the more specific of the pair. Unmeasured —
    /// unlike the null branch above, nothing has been run to show whether
    /// `Schema` minds the sibling.
    fn drops_type_beside_union(self) -> bool {
        !matches!(self, Self::GeminiJsonSchema)
    }
}

/// JSON Schema cleaner optimized for LLM tool calling.
pub struct SchemaCleanr;

impl SchemaCleanr {
    /// Clean schema for Gemini compatibility (strictest).
    ///
    /// This is the most aggressive cleaning strategy, removing all keywords
    /// that Gemini's API rejects.
    pub fn clean_for_gemini(schema: Value) -> Value {
        Self::clean(schema, CleaningStrategy::Gemini)
    }

    /// Clean schema for Anthropic compatibility.
    pub fn clean_for_anthropic(schema: Value) -> Value {
        Self::clean(schema, CleaningStrategy::Anthropic)
    }

    /// Clean schema for OpenAI compatibility (most permissive).
    pub fn clean_for_openai(schema: Value) -> Value {
        Self::clean(schema, CleaningStrategy::OpenAI)
    }

    /// Zero-copy wrapper around [`Self::clean`] for `Arc`-shared tool schemas:
    /// returns the same `Arc` when the pre-scan proves cleaning is a
    /// no-op, deep-copying the tree only when a rewrite is actually needed.
    pub fn clean_shared(schema: &Arc<Value>, strategy: CleaningStrategy) -> Arc<Value> {
        if Self::needs_cleaning(schema, strategy) {
            Arc::new(Self::clean((**schema).clone(), strategy))
        } else {
            Arc::clone(schema)
        }
    }

    /// Conservative read-only pre-scan: `true` when [`Self::clean`] with
    /// `strategy` could change `schema`.
    ///
    /// False positives are allowed (a flagged schema may clean to an equal
    /// value); false negatives are not — `!needs_cleaning(s)` must imply
    /// `clean(s) == s`. The triggers mirror every rewrite path in
    /// `clean_object`: strategy-specific keyword removal, plus the
    /// strategy-independent rewrites (`$ref` resolution, `const` → `enum`,
    /// `anyOf`/`oneOf` simplification and sibling-`type` skipping, and
    /// null-stripping in `type` arrays).
    pub fn needs_cleaning(schema: &Value, strategy: CleaningStrategy) -> bool {
        match schema {
            Value::Object(obj) => {
                for (key, value) in obj {
                    if !strategy.retains_keyword(key.as_str()) {
                        return true;
                    }
                    // Instance data is copied verbatim; nothing below it can
                    // trigger a rewrite.
                    if OPAQUE_SCHEMA_KEYS.contains(&key.as_str()) {
                        continue;
                    }
                    // Caller-chosen names: scan the subschemas, not the names.
                    if NAME_MAP_SCHEMA_KEYS.contains(&key.as_str()) {
                        if value.as_object().is_some_and(|entries| {
                            entries.values().any(|v| Self::needs_cleaning(v, strategy))
                        }) {
                            return true;
                        }
                        continue;
                    }
                    match key.as_str() {
                        "$ref" if strategy.resolves_refs() => return true,
                        "const" | "anyOf" | "oneOf" => return true,
                        // Folded into the parent by `clean_object`. Keyed on
                        // the fold's own predicate, never on keyword-list
                        // membership: a list edit must not be able to turn
                        // this into a false negative.
                        "allOf" if strategy.merges_all_of() => return true,
                        "type" if value.is_array() => return true,
                        _ => {}
                    }
                    if Self::needs_cleaning(value, strategy) {
                        return true;
                    }
                }
                false
            }
            Value::Array(arr) => arr.iter().any(|v| Self::needs_cleaning(v, strategy)),
            _ => false,
        }
    }

    /// The keywords `strategy`'s filter removes from `schema`, sorted.
    ///
    /// For diagnostics: a dropped keyword is a constraint the model never
    /// sees, so it can propose a value the tool itself then rejects. This
    /// reports only the keyword filter — `$ref` inlining and `const` → `enum`
    /// are rewrites that lose no information and are not listed.
    ///
    /// Traversal mirrors the cleaner's, so a caller-chosen name is never
    /// reported: a property called `pattern` is data, not a lost constraint.
    pub fn dropped_keywords(schema: &Value, strategy: CleaningStrategy) -> BTreeSet<String> {
        let mut dropped = BTreeSet::new();
        Self::collect_dropped_keywords(schema, strategy, &mut dropped);
        dropped
    }

    fn collect_dropped_keywords(
        schema: &Value,
        strategy: CleaningStrategy,
        dropped: &mut BTreeSet<String>,
    ) {
        match schema {
            Value::Object(obj) => {
                for (key, value) in obj {
                    // Where refs are inlined, the ref machinery is consumed
                    // rather than lost — the definition still reaches the
                    // output at every use site, so only its contents can lose
                    // anything to the filter.
                    if strategy.resolves_refs()
                        && matches!(key.as_str(), "$ref" | "$defs" | "definitions")
                    {
                        if let Some(entries) = value.as_object() {
                            for entry in entries.values() {
                                Self::collect_dropped_keywords(entry, strategy, dropped);
                            }
                        }
                        continue;
                    }
                    // `allOf` is folded into its parent rather than filtered
                    // out, so the keyword is consumed, not lost. Only its
                    // branches' contents can lose anything here — plus a
                    // branch keyword the fold declines to overwrite, which is
                    // too rare to be worth reporting as a keyword.
                    if key == "allOf" && strategy.merges_all_of() {
                        Self::collect_dropped_keywords(value, strategy, dropped);
                        continue;
                    }
                    if !strategy.retains_keyword(key.as_str()) {
                        // The whole subtree goes with it; naming the keyword
                        // is enough.
                        dropped.insert(key.clone());
                        continue;
                    }
                    if OPAQUE_SCHEMA_KEYS.contains(&key.as_str()) {
                        continue;
                    }
                    if NAME_MAP_SCHEMA_KEYS.contains(&key.as_str()) {
                        if let Some(entries) = value.as_object() {
                            for entry in entries.values() {
                                Self::collect_dropped_keywords(entry, strategy, dropped);
                            }
                        }
                        continue;
                    }
                    Self::collect_dropped_keywords(value, strategy, dropped);
                }
            }
            Value::Array(arr) => {
                for value in arr {
                    Self::collect_dropped_keywords(value, strategy, dropped);
                }
            }
            _ => {}
        }
    }

    /// Clean schema with specified strategy.
    pub fn clean(schema: Value, strategy: CleaningStrategy) -> Value {
        // Extract $defs for reference resolution
        let defs = if let Some(obj) = schema.as_object() {
            Self::extract_defs(obj)
        } else {
            HashMap::new()
        };

        Self::clean_with_defs(schema, &defs, strategy, &mut HashSet::new())
    }

    /// Validate that a schema is suitable for LLM tool calling.
    ///
    /// Returns an error if the schema is invalid or missing required fields.
    pub fn validate(schema: &Value) -> anyhow::Result<()> {
        let obj = schema
            .as_object()
            .ok_or_else(|| anyhow::Error::msg("Schema must be an object"))?;

        // Must have 'type' field
        if !obj.contains_key("type") {
            anyhow::bail!("Schema missing required 'type' field");
        }

        // If type is 'object', should have 'properties'
        if let Some(Value::String(t)) = obj.get("type")
            && t == "object"
            && !obj.contains_key("properties")
        {
            eprintln!("warn: Object schema without 'properties' field may cause issues");
        }

        Ok(())
    }

    // --------------------------------------------------------------------
    // Internal implementation
    // --------------------------------------------------------------------

    /// Extract $defs and definitions into a flat map for reference resolution.
    fn extract_defs(obj: &Map<String, Value>) -> HashMap<String, Value> {
        let mut defs = HashMap::new();

        // Extract from $defs (JSON Schema 2019-09+)
        if let Some(Value::Object(defs_obj)) = obj.get("$defs") {
            for (key, value) in defs_obj {
                defs.insert(key.clone(), value.clone());
            }
        }

        // Extract from definitions (JSON Schema draft-07)
        if let Some(Value::Object(defs_obj)) = obj.get("definitions") {
            for (key, value) in defs_obj {
                defs.insert(key.clone(), value.clone());
            }
        }

        defs
    }

    /// Recursively clean a schema value.
    fn clean_with_defs(
        schema: Value,
        defs: &HashMap<String, Value>,
        strategy: CleaningStrategy,
        ref_stack: &mut HashSet<String>,
    ) -> Value {
        match schema {
            Value::Object(obj) => Self::clean_object(obj, defs, strategy, ref_stack),
            Value::Array(arr) => Value::Array(
                arr.into_iter()
                    .map(|v| Self::clean_with_defs(v, defs, strategy, ref_stack))
                    .collect(),
            ),
            other => other,
        }
    }

    /// Clean an object schema.
    fn clean_object(
        obj: Map<String, Value>,
        defs: &HashMap<String, Value>,
        strategy: CleaningStrategy,
        ref_stack: &mut HashSet<String>,
    ) -> Value {
        // Handle $ref resolution
        if strategy.resolves_refs()
            && let Some(Value::String(ref_value)) = obj.get("$ref")
        {
            return Self::resolve_ref(ref_value, &obj, defs, strategy, ref_stack);
        }

        // Fold `allOf` into the object that carries it. No Gemini field
        // accepts the keyword, and the filter below would take the whole
        // branch list with it, so the composition is merged in while the
        // parent is still intact — and early enough that a branch's own union
        // still reaches the simplification below.
        //
        // After the `$ref` short-circuit deliberately: an object that *is* a
        // reference keeps nothing but its metadata anyway.
        let obj = if strategy.merges_all_of() && obj.contains_key("allOf") {
            Self::merge_all_of(obj, defs, strategy, ref_stack)
        } else {
            obj
        };

        // Handle anyOf/oneOf simplification
        if (obj.contains_key("anyOf") || obj.contains_key("oneOf"))
            && let Some(simplified) = Self::try_simplify_union(&obj, defs, strategy, ref_stack)
        {
            return simplified;
        }

        // Build cleaned object
        let mut cleaned = Map::new();
        // See `CleaningStrategy::drops_type_beside_union`.
        let type_yields_to_union = strategy.drops_type_beside_union()
            && (obj.contains_key("anyOf") || obj.contains_key("oneOf"));
        let has_any_of = obj.contains_key("anyOf");

        for (key, value) in obj {
            // Skip keywords this strategy does not carry
            if !strategy.retains_keyword(key.as_str()) {
                continue;
            }

            // Instance data: copy verbatim so its own keys are never read as
            // schema keywords.
            if OPAQUE_SCHEMA_KEYS.contains(&key.as_str()) {
                cleaned.insert(key, value);
                continue;
            }

            // Caller-chosen names mapping to subschemas: clean the values and
            // leave the names alone.
            if NAME_MAP_SCHEMA_KEYS.contains(&key.as_str()) {
                let cleaned_value = Self::clean_properties(value, defs, strategy, ref_stack);
                cleaned.insert(key, cleaned_value);
                continue;
            }

            // Special handling for specific keys
            match key.as_str() {
                // Convert const to enum
                "const" => {
                    cleaned.insert("enum".to_string(), json!([value]));
                }
                // Skip type if we have anyOf/oneOf (they define the type)
                "type" if type_yields_to_union => {
                    // Skip
                }
                // Handle type arrays (remove null)
                "type" if matches!(value, Value::Array(_)) => {
                    let cleaned_value = Self::clean_type_array(value);
                    cleaned.insert(key, cleaned_value);
                }
                "items" => {
                    let cleaned_value = Self::clean_with_defs(value, defs, strategy, ref_stack);
                    cleaned.insert(key, cleaned_value);
                }
                // Rewrite `oneOf` for targets that only have `anyOf`. See
                // `CleaningStrategy::rewrites_one_of`.
                "oneOf" if strategy.rewrites_one_of() && !has_any_of => {
                    let cleaned_value = Self::clean_union(value, defs, strategy, ref_stack);
                    cleaned.insert("anyOf".to_string(), cleaned_value);
                }
                // An `anyOf` sibling already holds the only slot the target
                // has. Dropping the `oneOf` beats clobbering it.
                "oneOf" if strategy.rewrites_one_of() => {}
                "anyOf" | "oneOf" | "allOf" => {
                    let cleaned_value = Self::clean_union(value, defs, strategy, ref_stack);
                    cleaned.insert(key, cleaned_value);
                }
                // Keep all other keys, cleaning nested objects/arrays recursively.
                _ => {
                    let cleaned_value = match value {
                        Value::Object(_) | Value::Array(_) => {
                            Self::clean_with_defs(value, defs, strategy, ref_stack)
                        }
                        other => other,
                    };
                    cleaned.insert(key, cleaned_value);
                }
            }
        }

        Value::Object(cleaned)
    }

    /// Resolve a $ref to its definition.
    fn resolve_ref(
        ref_value: &str,
        obj: &Map<String, Value>,
        defs: &HashMap<String, Value>,
        strategy: CleaningStrategy,
        ref_stack: &mut HashSet<String>,
    ) -> Value {
        // Prevent circular references
        if ref_stack.contains(ref_value) {
            eprintln!("warn: Circular $ref detected: {}", ref_value);
            return Self::preserve_meta(obj, Value::Object(Map::new()));
        }

        // Try to resolve local ref (#/$defs/Name or #/definitions/Name)
        if let Some(def_name) = Self::parse_local_ref(ref_value)
            && let Some(definition) = defs.get(def_name.as_str())
        {
            ref_stack.insert(ref_value.to_string());
            let cleaned = Self::clean_with_defs(definition.clone(), defs, strategy, ref_stack);
            ref_stack.remove(ref_value);
            return Self::preserve_meta(obj, cleaned);
        }

        // Can't resolve: return empty object with metadata
        eprintln!("warn: Cannot resolve $ref: {}", ref_value);
        Self::preserve_meta(obj, Value::Object(Map::new()))
    }

    /// Parse a local JSON Pointer ref (#/$defs/Name).
    fn parse_local_ref(ref_value: &str) -> Option<String> {
        ref_value
            .strip_prefix("#/$defs/")
            .or_else(|| ref_value.strip_prefix("#/definitions/"))
            .map(Self::decode_json_pointer)
    }

    /// Decode JSON Pointer escaping (`~0` = `~`, `~1` = `/`).
    fn decode_json_pointer(segment: &str) -> String {
        if !segment.contains('~') {
            return segment.to_string();
        }

        let mut decoded = String::with_capacity(segment.len());
        let mut chars = segment.chars().peekable();

        while let Some(ch) = chars.next() {
            if ch == '~' {
                match chars.peek().copied() {
                    Some('0') => {
                        chars.next();
                        decoded.push('~');
                    }
                    Some('1') => {
                        chars.next();
                        decoded.push('/');
                    }
                    _ => decoded.push('~'),
                }
            } else {
                decoded.push(ch);
            }
        }

        decoded
    }

    /// Try to simplify anyOf/oneOf to a simpler form.
    fn try_simplify_union(
        obj: &Map<String, Value>,
        defs: &HashMap<String, Value>,
        strategy: CleaningStrategy,
        ref_stack: &mut HashSet<String>,
    ) -> Option<Value> {
        let union_key = if obj.contains_key("anyOf") {
            "anyOf"
        } else if obj.contains_key("oneOf") {
            "oneOf"
        } else {
            return None;
        };

        let variants = obj.get(union_key)?.as_array()?;

        // Clean all variants first
        let cleaned_variants: Vec<Value> = variants
            .iter()
            .map(|v| Self::clean_with_defs(v.clone(), defs, strategy, ref_stack))
            .collect();

        // Strip null variants, for the targets that need it.
        let kept: Vec<Value> = if strategy.drops_null_variants() {
            cleaned_variants
                .into_iter()
                .filter(|v| !Self::is_null_schema(v))
                .collect()
        } else {
            cleaned_variants
        };

        // A union of one is that one, whether or not a null was removed to
        // get there.
        if kept.len() == 1 {
            return Some(Self::preserve_meta(obj, kept[0].clone()));
        }

        // Try to flatten to enum if all variants are literals
        if let Some(enum_value) = Self::try_flatten_literal_union(&kept) {
            return Some(Self::preserve_meta(obj, enum_value));
        }

        None
    }

    /// Check if a schema represents null type.
    fn is_null_schema(value: &Value) -> bool {
        if let Some(obj) = value.as_object() {
            // { const: null }
            if let Some(Value::Null) = obj.get("const") {
                return true;
            }
            // { enum: [null] }
            if let Some(Value::Array(arr)) = obj.get("enum")
                && arr.len() == 1
                && matches!(arr[0], Value::Null)
            {
                return true;
            }
            // { type: "null" }
            if let Some(Value::String(t)) = obj.get("type")
                && t == "null"
            {
                return true;
            }
        }
        false
    }

    /// Try to flatten anyOf/oneOf with only literal values to enum.
    ///
    /// Example: `anyOf: [{const: "a"}, {const: "b"}]` -> `{type: "string", enum: ["a", "b"]}`
    fn try_flatten_literal_union(variants: &[Value]) -> Option<Value> {
        if variants.is_empty() {
            return None;
        }

        let mut all_values = Vec::new();
        let mut common_type: Option<String> = None;

        for variant in variants {
            let obj = variant.as_object()?;

            // Extract literal value from const or single-item enum
            let literal_value = if let Some(const_val) = obj.get("const") {
                const_val.clone()
            } else if let Some(Value::Array(arr)) = obj.get("enum") {
                if arr.len() == 1 {
                    arr[0].clone()
                } else {
                    return None;
                }
            } else {
                return None;
            };

            // Check type consistency
            let variant_type = obj.get("type")?.as_str()?;
            match &common_type {
                None => common_type = Some(variant_type.to_string()),
                Some(t) if t != variant_type => return None,
                _ => {}
            }

            all_values.push(literal_value);
        }

        common_type.map(|t| {
            json!({
                "type": t,
                "enum": all_values
            })
        })
    }

    /// Clean type array, removing null.
    fn clean_type_array(value: Value) -> Value {
        if let Value::Array(types) = value {
            let non_null: Vec<Value> = types
                .into_iter()
                .filter(|v| v.as_str() != Some("null"))
                .collect();

            match non_null.len() {
                0 => Value::String("null".to_string()),
                1 => non_null
                    .into_iter()
                    .next()
                    .unwrap_or(Value::String("null".to_string())),
                _ => Value::Array(non_null),
            }
        } else {
            value
        }
    }

    /// Clean a name-to-subschema map (`properties`, `$defs`, ...): the keys are
    /// caller-chosen names and pass through untouched; only the values are
    /// cleaned.
    fn clean_properties(
        value: Value,
        defs: &HashMap<String, Value>,
        strategy: CleaningStrategy,
        ref_stack: &mut HashSet<String>,
    ) -> Value {
        if let Value::Object(props) = value {
            let cleaned: Map<String, Value> = props
                .into_iter()
                .map(|(k, v)| (k, Self::clean_with_defs(v, defs, strategy, ref_stack)))
                .collect();
            Value::Object(cleaned)
        } else {
            value
        }
    }

    /// Fold an `allOf` into the object that carries it, so the composition
    /// survives a target whose dialect has no `allOf` keyword.
    ///
    /// Merge rules, in the order they matter:
    ///
    /// - `required` unions. Every branch's demands hold at once — that is
    ///   what `allOf` means — so the union is exact.
    /// - `properties` merges by name; a name declared in more than one place
    ///   keeps the declaration closest to the use site.
    /// - Every other keyword is first-writer-wins, with the parent's own
    ///   entry counting as written. A branch that would contradict it is
    ///   dropped rather than allowed to loosen the parent's constraint.
    ///
    /// A branch that is a bare `$ref` has to be followed first: `allOf`
    /// composes the *contents* of its branches, and there is nothing to merge
    /// until the pointer resolves. Where the strategy keeps `$ref`, the first
    /// such branch is hoisted intact instead — that is the schemars and
    /// pydantic shape, and hoisting keeps a recursive definition recursive.
    /// Only a second one has to be resolved, having nowhere left to sit.
    ///
    /// A branch that composes further lands its own `allOf` on the parent,
    /// so this folds until none is left. That terminates: a literal branch
    /// list is finite and each pass consumes one nesting level, and a
    /// resolved branch has already been folded by the `clean_with_defs` call
    /// inside [`Self::resolve_ref`], so resolution never re-introduces one.
    /// A malformed non-array `allOf` ends the loop, discarded — no field this
    /// runs for would have accepted it.
    fn merge_all_of(
        mut obj: Map<String, Value>,
        defs: &HashMap<String, Value>,
        strategy: CleaningStrategy,
        ref_stack: &mut HashSet<String>,
    ) -> Map<String, Value> {
        while let Some(Value::Array(branches)) = obj.remove("allOf") {
            for branch in branches {
                let Value::Object(mut branch) = branch else {
                    continue;
                };

                // Where `$ref` survives cleaning, the first pointer branch is
                // hoisted onto the parent; anything after it has to be
                // inlined, since only one `$ref` slot exists.
                let hoistable_ref = !strategy.resolves_refs() && !obj.contains_key("$ref");
                let branch_ref = if hoistable_ref {
                    None
                } else {
                    branch.get("$ref").and_then(Value::as_str).map(String::from)
                };
                if let Some(ref_value) = branch_ref {
                    let Value::Object(resolved) =
                        Self::resolve_ref(&ref_value, &branch, defs, strategy, ref_stack)
                    else {
                        continue;
                    };
                    branch = resolved;
                }

                for (key, value) in branch {
                    Self::merge_all_of_entry(&mut obj, key, value);
                }
            }
        }

        obj
    }

    /// Merge one keyword of an `allOf` branch into the object it composes.
    /// See [`Self::merge_all_of`] for the rules this implements.
    fn merge_all_of_entry(target: &mut Map<String, Value>, key: String, value: Value) {
        if !target.contains_key(&key) {
            target.insert(key, value);
            return;
        }

        match (key.as_str(), target.get_mut(&key), value) {
            ("required", Some(Value::Array(present)), Value::Array(incoming)) => {
                for name in incoming {
                    if !present.contains(&name) {
                        present.push(name);
                    }
                }
            }
            ("properties", Some(Value::Object(present)), Value::Object(incoming)) => {
                for (name, subschema) in incoming {
                    if !present.contains_key(&name) {
                        present.insert(name, subschema);
                    }
                }
            }
            // A branch that composes further. Keep every branch list; the
            // caller's loop folds them on its next pass.
            ("allOf", Some(Value::Array(present)), Value::Array(incoming)) => {
                present.extend(incoming);
            }
            _ => {}
        }
    }

    /// Clean union (anyOf/oneOf/allOf).
    fn clean_union(
        value: Value,
        defs: &HashMap<String, Value>,
        strategy: CleaningStrategy,
        ref_stack: &mut HashSet<String>,
    ) -> Value {
        if let Value::Array(variants) = value {
            let cleaned: Vec<Value> = variants
                .into_iter()
                .map(|v| Self::clean_with_defs(v, defs, strategy, ref_stack))
                .collect();
            Value::Array(cleaned)
        } else {
            value
        }
    }

    /// Preserve metadata (description, title, default) from source to target.
    fn preserve_meta(source: &Map<String, Value>, mut target: Value) -> Value {
        if let Value::Object(target_obj) = &mut target {
            for &key in SCHEMA_META_KEYS {
                if let Some(value) = source.get(key) {
                    target_obj.insert(key.to_string(), value.clone());
                }
            }
        }
        target
    }
}

/// Per-provider bounds for completed cleaned-schema memos. The byte bound is
/// deliberately small because MCP schemas are externally supplied and have no
/// intrinsic size limit; the entry bound prevents many tiny trees from
/// accumulating metadata indefinitely.
const SCHEMA_CLEAN_CACHE_MAX_ENTRIES: usize = 64;
const SCHEMA_CLEAN_CACHE_MAX_BYTES: usize = 4 * 1024 * 1024;

struct SchemaCleanCacheEntry {
    /// Identity of the source schema this result was cleaned from. `Weak`
    /// so a cache entry never keeps a replaced (e.g. MCP-reconnect) schema
    /// alive on its own.
    source: std::sync::Weak<Value>,
    /// Single-flight cell for the deep-clean result. The map lock is only
    /// ever held to install or look up this cell, never while the clean
    /// itself runs: the first caller to reach [`OnceLock::get_or_init`] on a
    /// given cell performs the deep clone, and every other caller that
    /// reused the same entry (matched by `source`) blocks on that same
    /// `get_or_init` call and observes the identical `Arc` once it
    /// resolves. A `(schema, strategy)` key is therefore deep-cleaned at
    /// most once even when several threads race a cold miss together;
    /// unrelated keys use unrelated cells, so they still clean
    /// concurrently.
    cleaned: Arc<std::sync::OnceLock<Arc<Value>>>,
    /// Conservative heap-size estimate for an initialized `cleaned` tree.
    /// Shared with the initializer so it can publish the size without
    /// reacquiring the map lock.
    retained_bytes: Arc<std::sync::atomic::AtomicUsize>,
}

/// Bounded memo of [`SchemaCleanr::clean_shared`] results, keyed by source
/// schema identity and strategy.
///
/// Cleaning is a pure function of `(schema, strategy)`, but tool schemas
/// that need rewriting (`$ref`/`$defs`, `const`, unions — pervasive in
/// generated MCP schemas) would otherwise be deep-copied on every provider
/// request. Providers that clean per request embed one of these so each
/// distinct schema is cleaned once per strategy for as long as it stays
/// registered. This holds no canonical state: entries are derived values,
/// keyed by the identity of the canonical `Arc` the tool registry owns, and
/// the memoized result is byte-stable across requests (which also keeps
/// provider-side prompt caching stable).
///
/// Retired sources are pruned whenever a dirty schema uses the cache, so a
/// replaced MCP schema does not wait for capacity pressure before its cleaned
/// tree can be released. Completed entries are additionally bounded per
/// provider by both count and estimated heap bytes. When a new result would
/// exceed either bound, admission is declined for that result instead of
/// evicting the established working set. In-flight single-flight cells are
/// never evicted; concurrent callers participating in one cold miss still
/// share exactly one computation.
///
/// Only *rewritten* results are cached. A no-op clean is returned straight
/// from the pre-scan and never inserted: such an entry's `cleaned` field
/// would be the very allocation its `source` `Weak` watches, pinning it
/// forever (the dead-entry prune could never fire), and ephemeral per-call
/// `Arc`s — the default `Tool::spec()` builds a fresh one every iteration —
/// would flood the map until the overflow clear evicted the live memos this
/// cache exists to keep.
///
/// A hit requires upgrading the stored `Weak` **and** `Arc::ptr_eq` with
/// the candidate. Stale hits are impossible twice over: while an entry
/// lives, its `Weak` keeps the source `ArcInner` allocation reserved, so no
/// new schema can occupy that address; and once the source is dropped the
/// `Weak` permanently refuses to upgrade, so the entry can only miss.
pub struct SchemaCleanCache {
    entries: std::sync::Mutex<HashMap<(usize, CleaningStrategy), SchemaCleanCacheEntry>>,
    /// Counts actual deep-clean computations (`OnceLock::get_or_init`
    /// closure runs), as opposed to cache hits or single-flight waits.
    /// Test-only: lets the concurrent single-flight regression assert that
    /// N racing callers on the same key produced exactly one deep clone,
    /// not merely that they converged on the same returned pointer (a
    /// post-compute recheck could stabilize the pointer while still having
    /// done the duplicate work).
    #[cfg(test)]
    cold_compute_count: std::sync::atomic::AtomicUsize,
}

impl Default for SchemaCleanCache {
    fn default() -> Self {
        Self::new()
    }
}

impl SchemaCleanCache {
    pub fn new() -> Self {
        Self {
            entries: std::sync::Mutex::new(HashMap::new()),
            #[cfg(test)]
            cold_compute_count: std::sync::atomic::AtomicUsize::new(0),
        }
    }

    /// Memoized [`SchemaCleanr::clean_shared`]: returns the shared source
    /// `Arc` when cleaning is a no-op, and otherwise the cleaned tree —
    /// deep-computed at most once per retained (live schema, strategy) pair,
    /// even when multiple threads race a cold miss on the same key together.
    ///
    /// Single-flight: the first miss for a `(schema, strategy)` key installs
    /// a shared [`OnceLock`](std::sync::OnceLock) cell in the map before the
    /// lock is released. Any other thread that misses on the *same* key
    /// (i.e. it upgrades to the same live `source`) finds that cell already
    /// installed, reuses it, and blocks in `get_or_init` instead of starting
    /// its own deep clone — so only the winner's closure ever runs, and
    /// every caller, winner and waiters alike, ends up with the one
    /// resulting `Arc`. Misses on *different* keys install independent cells
    /// and clean fully concurrently; the map lock is never held while a clean
    /// itself runs. Capacity enforcement can decline admission only for the
    /// just-completed cell, so it cannot split an in-flight computation into
    /// competing cells or discard the established working set.
    pub fn clean_shared(&self, schema: &Arc<Value>, strategy: CleaningStrategy) -> Arc<Value> {
        if !SchemaCleanr::needs_cleaning(schema, strategy) {
            // No-op: nothing worth caching or single-flighting (see the
            // struct docs — a cached no-op would self-pin its source and
            // pollute the map with ephemeral per-call allocations).
            return Arc::clone(schema);
        }

        let key = (Arc::as_ptr(schema) as usize, strategy);
        let (cell, retained_bytes) = {
            let mut entries = self
                .entries
                .lock()
                .unwrap_or_else(std::sync::PoisonError::into_inner);
            // An actual in-flight caller still owns the source `Arc`, so a
            // dead source is also proof that its entry is safe to remove.
            entries.retain(|_, entry| entry.source.strong_count() > 0);
            if let Some(entry) = entries.get(&key)
                && let Some(live_source) = entry.source.upgrade()
                && Arc::ptr_eq(&live_source, schema)
            {
                (
                    Arc::clone(&entry.cleaned),
                    Arc::clone(&entry.retained_bytes),
                )
            } else {
                let cell = Arc::new(std::sync::OnceLock::new());
                let retained_bytes = Arc::new(std::sync::atomic::AtomicUsize::new(0));
                entries.insert(
                    key,
                    SchemaCleanCacheEntry {
                        source: Arc::downgrade(schema),
                        cleaned: Arc::clone(&cell),
                        retained_bytes: Arc::clone(&retained_bytes),
                    },
                );
                (cell, retained_bytes)
            }
        };

        // Outside the lock: `needs_cleaning` already proved above that this
        // schema requires a real rewrite, so `SchemaCleanr::clean_shared`
        // cannot take its no-op path here — it always performs (or, for
        // waiters, would have performed) the deep clone. Only the caller
        // that actually initializes `cell` runs the closure; concurrent
        // callers sharing `cell` block here and all observe that same
        // result `Arc`.
        let cleaned = Arc::clone(cell.get_or_init(|| {
            #[cfg(test)]
            self.cold_compute_count
                .fetch_add(1, std::sync::atomic::Ordering::SeqCst);
            let cleaned = SchemaCleanr::clean_shared(schema, strategy);
            retained_bytes.store(
                estimated_json_heap_bytes(&cleaned),
                std::sync::atomic::Ordering::Relaxed,
            );
            cleaned
        }));
        self.enforce_retention_bounds(key, &cell);
        cleaned
    }

    fn enforce_retention_bounds(
        &self,
        current_key: (usize, CleaningStrategy),
        current_cell: &Arc<std::sync::OnceLock<Arc<Value>>>,
    ) {
        let mut entries = self
            .entries
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner);
        entries.retain(|_, entry| entry.source.strong_count() > 0);

        let completed_count = entries
            .values()
            .filter(|entry| entry.cleaned.get().is_some())
            .count();
        let completed_bytes = entries.values().fold(0usize, |total, entry| {
            total.saturating_add(
                entry
                    .retained_bytes
                    .load(std::sync::atomic::Ordering::Relaxed),
            )
        });
        if completed_count <= SCHEMA_CLEAN_CACHE_MAX_ENTRIES
            && completed_bytes <= SCHEMA_CLEAN_CACHE_MAX_BYTES
        {
            return;
        }

        // Preserve the already-admitted working set. Clearing every completed
        // entry here makes a stable roster just one item over the cap thrash:
        // the last schema evicts all earlier schemas, then the next request
        // deep-cleans almost the entire roster again. Removing only the
        // just-completed cell implements admission control: the established
        // bounded set stays hot, while an over-budget schema is recomputed on
        // its next use. The identity check prevents an old caller from
        // removing a replacement entry installed at the same key.
        if entries.get(&current_key).is_some_and(|entry| {
            entry.cleaned.get().is_some() && Arc::ptr_eq(&entry.cleaned, current_cell)
        }) {
            entries.remove(&current_key);
        }
    }

    #[cfg(test)]
    fn len(&self) -> usize {
        self.entries
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner)
            .len()
    }

    /// Test-only: number of times the deep-clean closure actually ran
    /// (as opposed to cache hits or single-flight waits).
    #[cfg(test)]
    fn cold_compute_count(&self) -> usize {
        self.cold_compute_count
            .load(std::sync::atomic::Ordering::SeqCst)
    }
}

/// Estimate the heap retained by a JSON tree without serializing or allocating
/// a second representation. Container overhead is intentionally rounded up;
/// this is a retention guard, not an allocator accounting API.
fn estimated_json_heap_bytes(value: &Value) -> usize {
    fn heap_bytes(value: &Value) -> usize {
        match value {
            Value::Null | Value::Bool(_) | Value::Number(_) => 0,
            Value::String(text) => text.capacity(),
            Value::Array(items) => items.iter().fold(
                items
                    .capacity()
                    .saturating_mul(std::mem::size_of::<Value>()),
                |total, child| total.saturating_add(heap_bytes(child)),
            ),
            Value::Object(entries) => {
                const MAP_NODE_OVERHEAD: usize = 3 * std::mem::size_of::<usize>();
                entries.iter().fold(0usize, |total, (key, child)| {
                    total.saturating_add(
                        std::mem::size_of::<String>()
                            + std::mem::size_of::<Value>()
                            + MAP_NODE_OVERHEAD
                            + key.capacity()
                            + heap_bytes(child),
                    )
                })
            }
        }
    }

    std::mem::size_of::<Value>().saturating_add(heap_bytes(value))
}

#[cfg(test)]
mod tests {
    use super::*;

    /// `!needs_cleaning(s)` must imply `clean(s) == s` — the safety contract
    /// that lets `clean_shared` skip the deep copy.
    #[test]
    fn test_needs_cleaning_false_implies_clean_is_identity() {
        let clean_schemas = [
            json!({
                "type": "object",
                "properties": {
                    "path": { "type": "string", "description": "File path" },
                    "recursive": { "type": "boolean", "default": false },
                    "depth": { "type": "integer" }
                },
                "required": ["path"]
            }),
            json!({
                "type": "object",
                "properties": {
                    "items": { "type": "array", "items": { "type": "string" } },
                    "mode": { "type": "string", "enum": ["fast", "slow"] }
                }
            }),
            json!({ "type": "object", "properties": {} }),
        ];
        for strategy in [
            CleaningStrategy::Gemini,
            CleaningStrategy::GeminiJsonSchema,
            CleaningStrategy::Anthropic,
            CleaningStrategy::OpenAI,
            CleaningStrategy::Conservative,
        ] {
            for schema in &clean_schemas {
                assert!(
                    !SchemaCleanr::needs_cleaning(schema, strategy),
                    "expected no cleaning needed for {schema} under {strategy:?}"
                );
                assert_eq!(
                    SchemaCleanr::clean(schema.clone(), strategy),
                    *schema,
                    "clean must be identity when needs_cleaning is false ({strategy:?})"
                );
            }
        }
    }

    #[test]
    fn gemini_json_schema_keeps_refs_and_defs_intact() {
        let schema = json!({
            "type": "object",
            "properties": { "filter": { "$ref": "#/$defs/Filter" } },
            "$defs": { "Filter": { "type": "object", "properties": { "field": { "type": "string" } } } }
        });

        let cleaned = SchemaCleanr::clean(schema.clone(), CleaningStrategy::GeminiJsonSchema);

        assert_eq!(cleaned, schema, "refs and defs must round-trip untouched");
        assert!(!SchemaCleanr::needs_cleaning(
            &schema,
            CleaningStrategy::GeminiJsonSchema
        ));
    }

    #[test]
    fn gemini_json_schema_keeps_the_structure_the_openapi_subset_strips() {
        let schema = json!({
            "type": "object",
            "additionalProperties": false,
            "properties": {
                "count": { "type": "integer", "minimum": 1, "maximum": 9 },
                "tags": { "type": "array", "items": { "type": "string" }, "minItems": 1 }
            }
        });

        let cleaned = SchemaCleanr::clean(schema.clone(), CleaningStrategy::GeminiJsonSchema);
        let gemini = SchemaCleanr::clean(schema.clone(), CleaningStrategy::Gemini);

        assert_eq!(cleaned, schema);
        // `additionalProperties` is what the OpenAPI subset has no field for.
        assert!(gemini.get("additionalProperties").is_none());
        // The bounds it declares itself, so they are not a difference.
        assert_eq!(gemini["properties"]["count"]["minimum"], 1);
        assert_eq!(gemini["properties"]["count"]["maximum"], 9);
        assert_eq!(gemini["properties"]["tags"]["minItems"], 1);
    }

    #[test]
    fn gemini_json_schema_drops_keywords_outside_the_allowlist() {
        let schema = json!({
            "$schema": "https://json-schema.org/draft/2020-12/schema",
            "type": "object",
            "properties": {
                "q": { "type": "string", "minLength": 1, "multipleOf": 2 },
                "tags": { "type": "array", "items": { "type": "string" }, "uniqueItems": true }
            },
            "unevaluatedProperties": false
        });

        let cleaned = SchemaCleanr::clean(schema, CleaningStrategy::GeminiJsonSchema);

        assert!(cleaned.get("$schema").is_none());
        assert!(cleaned.get("unevaluatedProperties").is_none());
        assert!(cleaned["properties"]["q"].get("multipleOf").is_none());
        assert!(cleaned["properties"]["tags"].get("uniqueItems").is_none());
        // Measured to reach the model, so it is on the list.
        assert_eq!(cleaned["properties"]["q"]["minLength"], 1);
        assert_eq!(cleaned["properties"]["q"]["type"], "string");
    }

    #[test]
    fn the_openapi_subset_drops_a_keyword_it_has_never_heard_of() {
        // The reason this strategy filters by allowlist. `Schema` answers an
        // undeclared keyword with `Unknown name` and fails the request, and
        // no denylist over JSON Schema's open vocabulary stays ahead of that.
        let schema = json!({
            "type": "object",
            "properties": {
                "v": { "type": "string", "propertyNames": { "pattern": "^x" } }
            },
            "dependentRequired": { "v": ["w"] },
            "unevaluatedProperties": false
        });

        let cleaned = SchemaCleanr::clean_for_gemini(schema.clone());

        assert!(cleaned.get("dependentRequired").is_none());
        assert!(cleaned.get("unevaluatedProperties").is_none());
        assert!(cleaned["properties"]["v"].get("propertyNames").is_none());
        assert_eq!(cleaned["properties"]["v"]["type"], "string");

        // And the operator is told, rather than left to wonder.
        let dropped = SchemaCleanr::dropped_keywords(&schema, CleaningStrategy::Gemini);
        assert_eq!(
            dropped,
            ["dependentRequired", "propertyNames", "unevaluatedProperties"]
                .into_iter()
                .map(String::from)
                .collect()
        );
    }

    #[test]
    fn the_openapi_subset_drops_identifiers_and_tuple_items_json_schema_keeps() {
        // Both are `parametersJsonSchema` keywords. `parameters` answers them
        // with `Unknown name`, which fails the whole request, so the two
        // filters have to disagree here.
        let schema = json!({
            "type": "object",
            "properties": {
                "pair": {
                    "type": "array",
                    "prefixItems": [{ "type": "string" }, { "type": "integer" }]
                },
                "tag": { "type": "string", "$anchor": "tag" }
            }
        });

        let open_api = SchemaCleanr::clean_for_gemini(schema.clone());
        assert!(open_api["properties"]["pair"].get("prefixItems").is_none());
        assert!(open_api["properties"]["tag"].get("$anchor").is_none());

        let json_schema = SchemaCleanr::clean(schema, CleaningStrategy::GeminiJsonSchema);
        assert_eq!(
            json_schema["properties"]["pair"]["prefixItems"]
                .as_array()
                .expect("tuple typing kept")
                .len(),
            2
        );
        assert_eq!(json_schema["properties"]["tag"]["$anchor"], "tag");
    }

    #[test]
    fn dropped_keywords_names_every_constraint_the_allowlist_removes() {
        let schema = json!({
            "$schema": "https://json-schema.org/draft/2020-12/schema",
            "type": "object",
            "properties": {
                "q": { "type": "string", "minLength": 1, "multipleOf": 2 },
                "cfg": { "allOf": [{ "$ref": "#/$defs/Cfg" }] }
            },
            "$defs": { "Cfg": { "type": "object", "uniqueItems": true } }
        });

        let dropped = SchemaCleanr::dropped_keywords(&schema, CleaningStrategy::GeminiJsonSchema);

        assert_eq!(
            dropped,
            // No `allOf`: the fold consumes it, so only what the filter
            // takes out of the branch it points at is reported. No
            // `minLength` either: that one is on the list.
            ["$schema", "multipleOf", "uniqueItems"]
                .into_iter()
                .map(String::from)
                .collect()
        );
    }

    #[test]
    fn dropped_keywords_ignores_caller_chosen_names() {
        // A property named `multipleOf` is data. Reporting it as a lost
        // constraint would send someone hunting for a bug that is not there.
        // Both names are keywords this list drops, so a walk that confused
        // the two would say so.
        let schema = json!({
            "type": "object",
            "properties": {
                "multipleOf": { "type": "string" },
                "uniqueItems": { "type": "integer" }
            },
            "$defs": { "unevaluatedProperties": { "type": "boolean" } }
        });

        let dropped = SchemaCleanr::dropped_keywords(&schema, CleaningStrategy::GeminiJsonSchema);

        assert!(dropped.is_empty(), "unexpected drops: {dropped:?}");
    }

    #[test]
    fn dropped_keywords_does_not_count_inlined_refs_as_losses() {
        // Anthropic strips `$ref`/`$defs` from the output, but by resolving
        // them — nothing is lost. What the filter takes from inside a
        // definition still counts.
        let schema = json!({
            "type": "object",
            "properties": { "cfg": { "$ref": "#/$defs/Cfg" } },
            "$defs": { "Cfg": { "type": "object", "additionalProperties": false } }
        });

        assert!(
            SchemaCleanr::dropped_keywords(&schema, CleaningStrategy::Anthropic).is_empty(),
            "a resolved ref loses nothing"
        );
        assert_eq!(
            SchemaCleanr::dropped_keywords(&schema, CleaningStrategy::Conservative),
            ["additionalProperties"].into_iter().map(String::from).collect(),
            "Conservative still strips additionalProperties from the inlined definition"
        );
    }

    #[test]
    fn dropped_keywords_is_empty_when_nothing_is_filtered() {
        let schema = json!({
            "type": "object",
            "properties": { "q": { "type": "string", "description": "query" } },
            "required": ["q"]
        });

        for strategy in [
            CleaningStrategy::GeminiJsonSchema,
            CleaningStrategy::Anthropic,
            CleaningStrategy::OpenAI,
        ] {
            assert!(
                SchemaCleanr::dropped_keywords(&schema, strategy).is_empty(),
                "unexpected drops under {strategy:?}"
            );
        }
    }

    #[test]
    fn allowlist_never_mistakes_a_name_for_a_keyword() {
        // Property and definition names collide with schema keywords all the
        // time; an allowlist must not delete them. These three are outside
        // the list, so a walk that read them as keywords would drop them.
        let schema = json!({
            "type": "object",
            "properties": {
                "multipleOf": { "type": "string" },
                "uniqueItems": { "type": "string" }
            },
            "$defs": { "patternProperties": { "type": "integer" } }
        });

        let cleaned = SchemaCleanr::clean(schema.clone(), CleaningStrategy::GeminiJsonSchema);

        assert_eq!(cleaned, schema);
    }

    #[test]
    fn instance_data_is_never_read_as_schema_keywords() {
        // A `default` value's own keys are data. Cleaning must not descend
        // into them and strip entries that happen to be keyword-shaped.
        let schema = json!({
            "type": "object",
            "properties": {
                "opts": {
                    "type": "object",
                    "default": { "multipleOf": 3, "uniqueItems": true }
                }
            }
        });

        for strategy in [CleaningStrategy::Gemini, CleaningStrategy::GeminiJsonSchema] {
            let cleaned = SchemaCleanr::clean(schema.clone(), strategy);
            assert_eq!(
                cleaned["properties"]["opts"]["default"],
                json!({ "multipleOf": 3, "uniqueItems": true }),
                "default value must survive verbatim under {strategy:?}"
            );
        }
    }

    /// Every rewrite path in the cleaner must be flagged by the pre-scan.
    #[test]
    fn test_needs_cleaning_flags_every_rewrite_trigger() {
        let dirty = [
            // $ref resolution happens for every denylist strategy.
            json!({ "$ref": "#/$defs/Age", "$defs": { "Age": { "type": "integer" } } }),
            // const → enum conversion.
            json!({ "const": "fixed" }),
            // anyOf/oneOf simplification and sibling-type skipping.
            json!({ "anyOf": [{ "type": "string" }, { "type": "null" }] }),
            json!({ "oneOf": [{ "type": "string" }, { "type": "number" }] }),
            // type-array null stripping.
            json!({ "type": ["string", "null"] }),
            // Nested trigger below the top level.
            json!({
                "type": "object",
                "properties": { "role": { "const": "admin" } }
            }),
        ];
        for schema in &dirty {
            assert!(
                SchemaCleanr::needs_cleaning(schema, CleaningStrategy::OpenAI),
                "expected cleaning flagged even for the most permissive strategy: {schema}"
            );
        }
        // Strategy-specific keyword removal.
        let has_multiple_of = json!({ "type": "integer", "multipleOf": 2 });
        assert!(SchemaCleanr::needs_cleaning(
            &has_multiple_of,
            CleaningStrategy::Gemini
        ));
        assert!(!SchemaCleanr::needs_cleaning(
            &has_multiple_of,
            CleaningStrategy::Anthropic
        ));
    }

    #[test]
    fn schema_clean_cache_memoizes_dirty_schema_per_identity() {
        let cache = SchemaCleanCache::new();
        let dirty = Arc::new(json!({ "type": "string", "const": "x" }));

        let first = cache.clean_shared(&dirty, CleaningStrategy::Anthropic);
        let second = cache.clean_shared(&dirty, CleaningStrategy::Anthropic);

        assert!(
            !Arc::ptr_eq(&dirty, &first),
            "dirty schema must be rewritten"
        );
        assert!(
            Arc::ptr_eq(&first, &second),
            "repeated cleaning of the same live schema must return the memoized allocation"
        );
        assert_eq!(
            *first,
            SchemaCleanr::clean((*dirty).clone(), CleaningStrategy::Anthropic),
            "memoized result must equal the uncached cleaner output"
        );
    }

    #[test]
    fn schema_clean_cache_shares_clean_schema_without_inserting() {
        let cache = SchemaCleanCache::new();
        let clean = Arc::new(json!({
            "type": "object",
            "properties": { "path": { "type": "string" } }
        }));

        let shared = cache.clean_shared(&clean, CleaningStrategy::OpenAI);
        assert!(
            Arc::ptr_eq(&clean, &shared),
            "no-op cleaning must share the source Arc, not copy it"
        );
        assert_eq!(
            cache.len(),
            0,
            "no-op results must not be cached: a cached no-op self-pins its \
             source (cleaned aliases it, so the dead-entry prune can never \
             fire) and ephemeral per-call Arcs from the default Tool::spec() \
             would flood the map"
        );
    }

    #[test]
    fn schema_clean_cache_keys_strategies_independently() {
        let cache = SchemaCleanCache::new();
        // Dirty for Gemini (multipleOf is stripped), no-op for Anthropic.
        let schema = Arc::new(json!({ "type": "integer", "multipleOf": 2 }));

        let gemini = cache.clean_shared(&schema, CleaningStrategy::Gemini);
        let anthropic = cache.clean_shared(&schema, CleaningStrategy::Anthropic);

        assert!(!Arc::ptr_eq(&schema, &gemini));
        assert!(gemini.get("multipleOf").is_none());
        assert!(
            Arc::ptr_eq(&schema, &anthropic),
            "a strategy the schema is already clean for must still share"
        );
        assert!(
            Arc::ptr_eq(
                &gemini,
                &cache.clean_shared(&schema, CleaningStrategy::Gemini)
            ),
            "each strategy keeps its own memoized entry"
        );
    }

    #[test]
    fn schema_clean_cache_never_serves_stale_result_for_new_schema() {
        let cache = SchemaCleanCache::new();
        let original = Arc::new(json!({ "type": "string", "const": "old" }));
        let original_cleaned = cache.clean_shared(&original, CleaningStrategy::OpenAI);
        assert_eq!(original_cleaned["enum"], json!(["old"]));
        drop(original);

        // A replacement schema (e.g. MCP reconnect) cannot land at the old
        // address while the entry lives — the entry's own `Weak` keeps the
        // old `ArcInner` allocation reserved — so this exercises the plain
        // miss-then-recompute path. Address reuse only becomes possible
        // after the entry (and its `Weak`) is pruned, at which point no
        // stale entry exists to hit. Either way: fresh compute.
        let replacement = Arc::new(json!({ "type": "string", "const": "new" }));
        let replacement_cleaned = cache.clean_shared(&replacement, CleaningStrategy::OpenAI);
        assert_eq!(
            replacement_cleaned["enum"],
            json!(["new"]),
            "cache must never serve a dropped schema's cleaned result"
        );
    }

    #[test]
    fn schema_clean_cache_stays_entry_bounded_when_all_sources_live() {
        let cache = SchemaCleanCache::new();
        let sources: Vec<Arc<Value>> = (0..=SCHEMA_CLEAN_CACHE_MAX_ENTRIES)
            .map(|i| Arc::new(json!({ "type": "string", "const": format!("v{i}") })))
            .collect();
        for source in &sources {
            cache.clean_shared(source, CleaningStrategy::OpenAI);
        }
        assert!(
            cache.len() <= SCHEMA_CLEAN_CACHE_MAX_ENTRIES,
            "cache must never retain more than its cap ({}), got {}",
            SCHEMA_CLEAN_CACHE_MAX_ENTRIES,
            cache.len()
        );
    }

    #[test]
    fn schema_clean_cache_preserves_working_set_when_roster_exceeds_entry_cap() {
        let cache = SchemaCleanCache::new();
        let sources: Vec<Arc<Value>> = (0..=SCHEMA_CLEAN_CACHE_MAX_ENTRIES)
            .map(|i| Arc::new(json!({ "type": "string", "const": format!("v{i}") })))
            .collect();

        for source in &sources {
            cache.clean_shared(source, CleaningStrategy::OpenAI);
        }
        assert_eq!(
            cache.cold_compute_count(),
            sources.len(),
            "the first pass must clean each distinct dirty schema once"
        );

        for source in &sources {
            cache.clean_shared(source, CleaningStrategy::OpenAI);
        }
        assert_eq!(
            cache.cold_compute_count(),
            sources.len() + 1,
            "a stable roster one schema over the cap should recompute only the \
             unadmitted schema, not evict and rebuild the retained working set"
        );
        assert_eq!(cache.len(), SCHEMA_CLEAN_CACHE_MAX_ENTRIES);
    }

    #[test]
    fn schema_clean_cache_prunes_retired_source_without_capacity_pressure() {
        let cache = SchemaCleanCache::new();
        let retired = Arc::new(json!({ "type": "string", "const": "retired" }));
        let retired_cleaned = cache.clean_shared(&retired, CleaningStrategy::OpenAI);
        let retired_cleaned_weak = Arc::downgrade(&retired_cleaned);
        drop(retired_cleaned);
        drop(retired);
        assert!(
            retired_cleaned_weak.upgrade().is_some(),
            "the cache should own the cleaned allocation before retirement"
        );

        let active = Arc::new(json!({ "type": "string", "const": "active" }));
        cache.clean_shared(&active, CleaningStrategy::OpenAI);

        assert!(
            retired_cleaned_weak.upgrade().is_none(),
            "ordinary dirty-schema activity must release a retired source's \
             cleaned allocation without filling the cache first"
        );
        assert_eq!(cache.len(), 1);
    }

    #[test]
    fn schema_clean_cache_does_not_retain_one_oversized_external_schema() {
        let cache = SchemaCleanCache::new();
        let oversized = Arc::new(json!({
            "type": "string",
            "const": "x".repeat(SCHEMA_CLEAN_CACHE_MAX_BYTES)
        }));

        let cleaned = cache.clean_shared(&oversized, CleaningStrategy::OpenAI);
        assert_eq!(&cleaned["enum"][0], oversized.get("const").unwrap());
        assert_eq!(
            cache.len(),
            0,
            "one externally supplied schema larger than the byte budget must \
             not remain retained by the provider cache"
        );
    }

    #[test]
    fn schema_clean_cache_pressure_never_evicts_in_flight_cell() {
        let cache = SchemaCleanCache::new();
        let in_flight_source = Arc::new(json!({ "type": "string", "const": "in-flight" }));
        let in_flight_key = (
            Arc::as_ptr(&in_flight_source) as usize,
            CleaningStrategy::OpenAI,
        );
        let in_flight_cell = Arc::new(std::sync::OnceLock::new());
        cache
            .entries
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner)
            .insert(
                in_flight_key,
                SchemaCleanCacheEntry {
                    source: Arc::downgrade(&in_flight_source),
                    cleaned: Arc::clone(&in_flight_cell),
                    retained_bytes: Arc::new(std::sync::atomic::AtomicUsize::new(0)),
                },
            );

        let live_sources: Vec<Arc<Value>> = (0..=SCHEMA_CLEAN_CACHE_MAX_ENTRIES)
            .map(|i| Arc::new(json!({ "type": "string", "const": format!("v{i}") })))
            .collect();
        for source in &live_sources {
            cache.clean_shared(source, CleaningStrategy::OpenAI);
        }

        let entries = cache
            .entries
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner);
        assert!(
            entries.get(&in_flight_key).is_some_and(|entry| {
                entry.cleaned.get().is_none() && Arc::ptr_eq(&entry.cleaned, &in_flight_cell)
            }),
            "capacity enforcement must retain the installed single-flight \
             cell until its first computation finishes"
        );
    }

    /// Regression for a single-flight gap: releasing the map lock before
    /// cleaning let two concurrent cold misses on the *same* `(schema,
    /// strategy)` key both deep-clean and then race to insert, so callers
    /// could observe different allocations for what should be one memoized
    /// result. Races many threads on a common start line against one dirty
    /// schema and asserts both that the deep clean ran exactly once (via
    /// the test-only compute counter, not just pointer convergence — a
    /// post-compute recheck could stabilize the returned pointer while
    /// still having done the duplicate work) and that every racer shares
    /// that one allocation.
    #[test]
    fn schema_clean_cache_single_flights_concurrent_cold_miss_on_same_key() {
        const RACERS: usize = 32;

        let cache = Arc::new(SchemaCleanCache::new());
        // A schema with real fan-out so a duplicate deep clone is not just
        // wasted CPU cycles but a genuinely distinct tree.
        let properties: Map<String, Value> = (0..64)
            .map(|i| (format!("field_{i}"), json!({ "const": format!("v{i}") })))
            .collect();
        let dirty: Arc<Value> = Arc::new(json!({
            "type": "object",
            "properties": Value::Object(properties)
        }));

        let barrier = std::sync::Arc::new(std::sync::Barrier::new(RACERS));
        let handles: Vec<_> = (0..RACERS)
            .map(|_| {
                let cache = Arc::clone(&cache);
                let dirty = Arc::clone(&dirty);
                let barrier = std::sync::Arc::clone(&barrier);
                std::thread::spawn(move || {
                    // Every racer blocks here until all RACERS threads have
                    // started, so the misses genuinely overlap instead of
                    // serializing through thread-spawn latency.
                    barrier.wait();
                    cache.clean_shared(&dirty, CleaningStrategy::Anthropic)
                })
            })
            .collect();

        let results: Vec<Arc<Value>> = handles
            .into_iter()
            .map(|h| h.join().expect("racer thread panicked"))
            .collect();

        assert_eq!(
            cache.cold_compute_count(),
            1,
            "single-flight must deep-clean exactly once for a (schema, \
             strategy) key raced by {RACERS} concurrent callers; a count \
             greater than 1 means concurrent misses each deep-cloned \
             independently instead of sharing the first computation"
        );

        let first = &results[0];
        assert!(
            !Arc::ptr_eq(first, &dirty),
            "dirty schema must actually be rewritten by the single \
             computation, not shared as-is"
        );
        for (i, result) in results.iter().enumerate() {
            assert!(
                Arc::ptr_eq(first, result),
                "racer {i} observed a different allocation than racer 0; \
                 every concurrent caller on the same key must share the \
                 one single-flighted result"
            );
        }
    }

    #[test]
    fn test_clean_shared_returns_same_arc_when_clean() {
        let schema = Arc::new(json!({
            "type": "object",
            "properties": { "path": { "type": "string" } }
        }));
        let shared = SchemaCleanr::clean_shared(&schema, CleaningStrategy::Anthropic);
        assert!(
            Arc::ptr_eq(&schema, &shared),
            "clean schema must be shared, not copied"
        );

        let dirty = Arc::new(json!({ "type": "string", "const": "x" }));
        let cleaned = SchemaCleanr::clean_shared(&dirty, CleaningStrategy::Anthropic);
        assert!(!Arc::ptr_eq(&dirty, &cleaned));
        assert_eq!(cleaned["enum"], json!(["x"]));
    }

    #[test]
    fn test_remove_unsupported_keywords() {
        let schema = json!({
            "$schema": "https://json-schema.org/draft/2020-12/schema",
            "type": "array",
            "items": { "type": "integer", "multipleOf": 2 },
            "uniqueItems": true,
            "minItems": 1,
            "description": "A list of even numbers"
        });

        let cleaned = SchemaCleanr::clean_for_gemini(schema);

        assert_eq!(cleaned["type"], "array");
        assert_eq!(cleaned["description"], "A list of even numbers");
        assert!(cleaned.get("$schema").is_none());
        assert!(cleaned.get("uniqueItems").is_none());
        assert!(cleaned["items"].get("multipleOf").is_none());
        // `minItems` is a field on `Schema`, so it is not unsupported.
        assert_eq!(cleaned["minItems"], 1);
    }

    #[test]
    fn test_resolve_ref() {
        let schema = json!({
            "type": "object",
            "properties": {
                "age": {
                    "$ref": "#/$defs/Age"
                }
            },
            "$defs": {
                "Age": {
                    "type": "integer",
                    "minimum": 0
                }
            }
        });

        let cleaned = SchemaCleanr::clean_for_gemini(schema);

        assert_eq!(cleaned["properties"]["age"]["type"], "integer");
        assert_eq!(cleaned["properties"]["age"]["minimum"], 0);
        assert!(cleaned.get("$defs").is_none());
    }

    #[test]
    fn test_resolve_ref_decodes_json_pointer_escapes() {
        let schema = json!({
            "type": "object",
            "properties": {
                "slash": { "$ref": "#/$defs/Foo~1Bar" },
                "tilde": { "$ref": "#/$defs/Tilde~0Name" }
            },
            "$defs": {
                "Foo/Bar": { "type": "string" },
                "Tilde~Name": { "type": "integer" }
            }
        });

        let cleaned = SchemaCleanr::clean_for_anthropic(schema);

        assert_eq!(cleaned["properties"]["slash"]["type"], "string");
        assert_eq!(cleaned["properties"]["tilde"]["type"], "integer");
        assert!(cleaned.get("$defs").is_none());
    }

    #[test]
    fn test_flatten_literal_union() {
        let schema = json!({
            "anyOf": [
                { "const": "admin", "type": "string" },
                { "const": "user", "type": "string" },
                { "const": "guest", "type": "string" }
            ]
        });

        let cleaned = SchemaCleanr::clean_for_gemini(schema);

        assert_eq!(cleaned["type"], "string");
        assert!(cleaned["enum"].is_array());
        let enum_values = cleaned["enum"].as_array().unwrap();
        assert_eq!(enum_values.len(), 3);
        assert!(enum_values.contains(&json!("admin")));
        assert!(enum_values.contains(&json!("user")));
        assert!(enum_values.contains(&json!("guest")));
    }

    #[test]
    fn test_strip_null_from_union() {
        let schema = json!({
            "oneOf": [
                { "type": "string" },
                { "type": "null" }
            ]
        });

        // Still the behaviour for the denylist strategies. The Gemini fields
        // keep the branch — see the nullable-union test below.
        let cleaned = SchemaCleanr::clean(schema, CleaningStrategy::OpenAI);

        // Should simplify to just { type: "string" }
        assert_eq!(cleaned["type"], "string");
        assert!(cleaned.get("oneOf").is_none());
    }

    #[test]
    fn test_const_to_enum() {
        let schema = json!({
            "const": "fixed_value",
            "description": "A constant"
        });

        let cleaned = SchemaCleanr::clean_for_gemini(schema);

        assert_eq!(cleaned["enum"], json!(["fixed_value"]));
        assert_eq!(cleaned["description"], "A constant");
        assert!(cleaned.get("const").is_none());
    }

    #[test]
    fn test_preserve_metadata() {
        let schema = json!({
            "$ref": "#/$defs/Name",
            "description": "User's name",
            "title": "Name Field",
            "default": "Anonymous",
            "$defs": {
                "Name": {
                    "type": "string"
                }
            }
        });

        let cleaned = SchemaCleanr::clean_for_gemini(schema);

        assert_eq!(cleaned["type"], "string");
        assert_eq!(cleaned["description"], "User's name");
        assert_eq!(cleaned["title"], "Name Field");
        assert_eq!(cleaned["default"], "Anonymous");
    }

    #[test]
    fn test_circular_ref_prevention() {
        let schema = json!({
            "type": "object",
            "properties": {
                "parent": {
                    "$ref": "#/$defs/Node"
                }
            },
            "$defs": {
                "Node": {
                    "type": "object",
                    "properties": {
                        "child": {
                            "$ref": "#/$defs/Node"
                        }
                    }
                }
            }
        });

        // Should not panic on circular reference
        let cleaned = SchemaCleanr::clean_for_gemini(schema);

        assert_eq!(cleaned["properties"]["parent"]["type"], "object");
        // Circular reference should be broken
    }

    #[test]
    fn test_validate_schema() {
        let valid = json!({
            "type": "object",
            "properties": {
                "name": { "type": "string" }
            }
        });

        assert!(SchemaCleanr::validate(&valid).is_ok());

        let invalid = json!({
            "properties": {
                "name": { "type": "string" }
            }
        });

        assert!(SchemaCleanr::validate(&invalid).is_err());
    }

    #[test]
    fn test_strategy_differences() {
        let schema = json!({
            "type": "integer",
            "multipleOf": 2,
            "description": "An even number"
        });

        // Gemini: Most restrictive (removes multipleOf)
        let gemini = SchemaCleanr::clean_for_gemini(schema.clone());
        assert!(gemini.get("multipleOf").is_none());
        assert_eq!(gemini["type"], "integer");
        assert_eq!(gemini["description"], "An even number");

        // OpenAI: Most permissive (keeps multipleOf)
        let openai = SchemaCleanr::clean_for_openai(schema.clone());
        assert_eq!(openai["multipleOf"], 2); // OpenAI allows validation keywords
        assert_eq!(openai["type"], "integer");
    }

    #[test]
    fn test_nested_properties() {
        let schema = json!({
            "type": "object",
            "properties": {
                "user": {
                    "type": "object",
                    "properties": {
                        "age": {
                            "type": "integer",
                            "multipleOf": 2,
                            "minimum": 0
                        }
                    },
                    "additionalProperties": false
                }
            }
        });

        let cleaned = SchemaCleanr::clean_for_gemini(schema);

        let age = &cleaned["properties"]["user"]["properties"]["age"];
        assert!(age.get("multipleOf").is_none());
        assert_eq!(age["minimum"], 0, "a bound `Schema` declares must survive");
        assert!(
            cleaned["properties"]["user"]
                .get("additionalProperties")
                .is_none()
        );
    }

    #[test]
    fn test_type_array_null_removal() {
        let schema = json!({
            "type": ["string", "null"]
        });

        let cleaned = SchemaCleanr::clean_for_gemini(schema);

        // Should simplify to just "string"
        assert_eq!(cleaned["type"], "string");
    }

    #[test]
    fn test_type_array_only_null_preserved() {
        let schema = json!({
            "type": ["null"]
        });

        let cleaned = SchemaCleanr::clean_for_gemini(schema);

        assert_eq!(cleaned["type"], "null");
    }

    #[test]
    fn test_ref_with_json_pointer_escape() {
        let schema = json!({
            "$ref": "#/$defs/Foo~1Bar",
            "$defs": {
                "Foo/Bar": {
                    "type": "string"
                }
            }
        });

        let cleaned = SchemaCleanr::clean_for_gemini(schema);

        assert_eq!(cleaned["type"], "string");
    }

    #[test]
    fn test_skip_type_when_non_simplifiable_union_exists() {
        let schema = json!({
            "type": "object",
            "oneOf": [
                {
                    "type": "object",
                    "properties": {
                        "a": { "type": "string" }
                    }
                },
                {
                    "type": "object",
                    "properties": {
                        "b": { "type": "number" }
                    }
                }
            ]
        });

        let cleaned = SchemaCleanr::clean_for_gemini(schema);

        assert!(cleaned.get("type").is_none());
        // `Schema` has no `oneOf`, so the union travels as `anyOf`.
        assert!(cleaned.get("oneOf").is_none());
        assert_eq!(cleaned["anyOf"].as_array().expect("union kept").len(), 2);
    }

    #[test]
    fn test_clean_nested_unknown_schema_keyword() {
        // A denylist strategy carries keywords it has no opinion about, and
        // still has to clean what sits underneath them. (The Gemini
        // strategies filter by allowlist, so for them there is no such thing
        // as a keyword that survives unrecognised.)
        let schema = json!({
            "not": {
                "$ref": "#/$defs/Age"
            },
            "$defs": {
                "Age": {
                    "type": "object",
                    "additionalProperties": false
                }
            }
        });

        let cleaned = SchemaCleanr::clean(schema, CleaningStrategy::Conservative);

        assert_eq!(cleaned["not"]["type"], "object");
        assert!(cleaned["not"].get("additionalProperties").is_none());
    }

    #[test]
    fn an_all_of_pointer_is_hoisted_where_the_strategy_keeps_refs() {
        // What schemars and pydantic emit for a `$ref` with siblings. Before
        // the fold this cleaned down to a bare description.
        let schema = json!({
            "type": "object",
            "properties": {
                "cfg": {
                    "allOf": [{ "$ref": "#/$defs/Cfg" }],
                    "description": "the config"
                }
            },
            "$defs": { "Cfg": { "type": "object", "properties": {} } }
        });

        let cleaned = SchemaCleanr::clean(schema, CleaningStrategy::GeminiJsonSchema);

        let cfg = &cleaned["properties"]["cfg"];
        assert!(cfg.get("allOf").is_none());
        assert_eq!(cfg["$ref"], "#/$defs/Cfg");
        // The parent's own description is the specific one; it survives.
        assert_eq!(cfg["description"], "the config");
        assert_eq!(cleaned["$defs"]["Cfg"]["type"], "object");
    }

    #[test]
    fn an_all_of_pointer_is_resolved_where_the_strategy_inlines_refs() {
        let schema = json!({
            "type": "object",
            "properties": {
                "cfg": {
                    "allOf": [{ "$ref": "#/$defs/Cfg" }],
                    "description": "the config"
                }
            },
            "$defs": {
                "Cfg": {
                    "type": "object",
                    "properties": { "level": { "type": "integer" } }
                }
            }
        });

        let cleaned = SchemaCleanr::clean_for_gemini(schema);

        let cfg = &cleaned["properties"]["cfg"];
        assert!(cfg.get("allOf").is_none());
        assert!(cfg.get("$ref").is_none());
        assert_eq!(cfg["type"], "object");
        assert_eq!(cfg["properties"]["level"]["type"], "integer");
        assert_eq!(cfg["description"], "the config");
    }

    #[test]
    fn all_of_branches_union_required_and_merge_properties() {
        let schema = json!({
            "type": "object",
            "required": ["a"],
            "properties": { "a": { "type": "string" } },
            "allOf": [
                {
                    "properties": { "b": { "type": "number" } },
                    "required": ["b"]
                },
                {
                    "properties": { "c": { "type": "boolean" } },
                    "required": ["a", "c"]
                }
            ]
        });

        let cleaned = SchemaCleanr::clean_for_gemini(schema);

        assert!(cleaned.get("allOf").is_none());
        assert_eq!(cleaned["properties"]["a"]["type"], "string");
        assert_eq!(cleaned["properties"]["b"]["type"], "number");
        assert_eq!(cleaned["properties"]["c"]["type"], "boolean");
        let required = cleaned["required"].as_array().unwrap();
        assert_eq!(required.len(), 3, "`a` is demanded twice, carried once");
        for name in ["a", "b", "c"] {
            assert!(required.contains(&json!(name)), "{name} missing");
        }
    }

    #[test]
    fn the_parent_keyword_wins_over_an_all_of_branch() {
        let schema = json!({
            "type": "object",
            "description": "outer",
            "properties": { "a": { "type": "string" } },
            "allOf": [{
                "description": "inner",
                "properties": { "a": { "type": "integer" } }
            }]
        });

        let cleaned = SchemaCleanr::clean_for_gemini(schema);

        assert_eq!(cleaned["description"], "outer");
        assert_eq!(cleaned["properties"]["a"]["type"], "string");
    }

    #[test]
    fn a_second_all_of_pointer_is_resolved_rather_than_dropped() {
        // Only one `$ref` slot exists on the merged object, so the first
        // branch is hoisted and the rest have to be inlined to survive.
        let schema = json!({
            "type": "object",
            "properties": {
                "cfg": {
                    "allOf": [
                        { "$ref": "#/$defs/A" },
                        { "$ref": "#/$defs/B" }
                    ]
                }
            },
            "$defs": {
                "A": { "type": "object", "properties": { "a": { "type": "string" } } },
                "B": { "type": "object", "properties": { "b": { "type": "string" } } }
            }
        });

        let cleaned = SchemaCleanr::clean(schema, CleaningStrategy::GeminiJsonSchema);

        let cfg = &cleaned["properties"]["cfg"];
        assert!(cfg.get("allOf").is_none());
        assert_eq!(cfg["$ref"], "#/$defs/A");
        assert_eq!(cfg["properties"]["b"]["type"], "string");
    }

    #[test]
    fn a_recursive_all_of_pointer_does_not_run_away() {
        let schema = json!({
            "type": "object",
            "properties": { "node": { "allOf": [{ "$ref": "#/$defs/Node" }] } },
            "$defs": {
                "Node": {
                    "type": "object",
                    "properties": { "child": { "$ref": "#/$defs/Node" } }
                }
            }
        });

        let cleaned = SchemaCleanr::clean_for_gemini(schema);

        assert_eq!(cleaned["properties"]["node"]["type"], "object");
    }

    #[test]
    fn a_nullable_union_keeps_its_null_branch_for_both_gemini_fields() {
        let schema = json!({
            "anyOf": [{ "type": "string" }, { "type": "null" }]
        });

        for strategy in [CleaningStrategy::Gemini, CleaningStrategy::GeminiJsonSchema] {
            let cleaned = SchemaCleanr::clean(schema.clone(), strategy);
            let branches = cleaned["anyOf"].as_array().expect("union kept");
            assert_eq!(
                branches.len(),
                2,
                "{strategy:?} dropped the caller's way of saying optional"
            );
            assert_eq!(branches[1]["type"], "null", "{strategy:?}");
        }
    }

    #[test]
    fn a_nullable_one_of_reaches_the_openapi_subset_as_a_nullable_any_of() {
        // Both rewrites at once: the branch stays, and the keyword the model
        // does not read becomes the one it does.
        let schema = json!({
            "oneOf": [{ "type": "string", "enum": ["alpha"] }, { "type": "null" }]
        });

        let cleaned = SchemaCleanr::clean_for_gemini(schema);

        assert!(cleaned.get("oneOf").is_none());
        let branches = cleaned["anyOf"].as_array().expect("union kept");
        assert_eq!(branches.len(), 2);
        assert_eq!(branches[0]["enum"], json!(["alpha"]));
        assert_eq!(branches[1]["type"], "null");
    }

    #[test]
    fn a_union_keeps_its_type_sibling_for_the_json_schema_field() {
        let schema = json!({
            "type": "object",
            "anyOf": [
                { "properties": { "a": { "type": "string" } } },
                { "properties": { "b": { "type": "string" } } }
            ]
        });

        let json_schema = SchemaCleanr::clean(schema.clone(), CleaningStrategy::GeminiJsonSchema);
        assert_eq!(json_schema["type"], "object");
        assert_eq!(
            json_schema["anyOf"].as_array().expect("union kept").len(),
            2
        );

        let open_api = SchemaCleanr::clean_for_gemini(schema);
        assert!(open_api.get("type").is_none());
    }

    #[test]
    fn the_lossless_union_simplifications_run_for_every_target() {
        let single = json!({ "anyOf": [{ "type": "string" }] });
        let literals = json!({
            "anyOf": [
                { "type": "string", "const": "a" },
                { "type": "string", "const": "b" }
            ]
        });

        for strategy in [CleaningStrategy::Gemini, CleaningStrategy::GeminiJsonSchema] {
            let collapsed = SchemaCleanr::clean(single.clone(), strategy);
            assert_eq!(collapsed["type"], "string", "{strategy:?}");
            assert!(collapsed.get("anyOf").is_none(), "{strategy:?}");

            let flattened = SchemaCleanr::clean(literals.clone(), strategy);
            assert_eq!(flattened["enum"], json!(["a", "b"]), "{strategy:?}");
        }
    }

    #[test]
    fn a_json_schema_target_keeps_one_of_but_the_openapi_subset_does_not() {
        let schema = json!({
            "oneOf": [
                { "type": "object", "properties": { "a": { "type": "string" } } },
                { "type": "object", "properties": { "b": { "type": "string" } } }
            ]
        });

        let json_schema = SchemaCleanr::clean(schema.clone(), CleaningStrategy::GeminiJsonSchema);
        assert!(json_schema.get("oneOf").is_some());
        assert!(json_schema.get("anyOf").is_none());

        let open_api = SchemaCleanr::clean_for_gemini(schema);
        assert!(open_api.get("oneOf").is_none());
        assert_eq!(open_api["anyOf"].as_array().expect("union kept").len(), 2);
    }

    #[test]
    fn an_any_of_sibling_keeps_its_slot_when_one_of_is_rewritten() {
        let schema = json!({
            "anyOf": [
                { "type": "object", "properties": { "a": { "type": "string" } } },
                { "type": "object", "properties": { "b": { "type": "string" } } }
            ],
            "oneOf": [
                { "type": "object", "properties": { "c": { "type": "string" } } },
                { "type": "object", "properties": { "d": { "type": "string" } } }
            ]
        });

        let cleaned = SchemaCleanr::clean_for_gemini(schema);

        assert_eq!(cleaned["anyOf"][0]["properties"]["a"]["type"], "string");
        assert!(cleaned.get("oneOf").is_none());
    }

    #[test]
    fn a_permissive_target_keeps_all_of_and_one_of_untouched() {
        // Anthropic and OpenAI take full JSON Schema; neither rewrite applies.
        let schema = json!({
            "type": "object",
            "properties": {
                "cfg": { "allOf": [{ "$ref": "#/$defs/Cfg" }] },
                "pick": {
                    "oneOf": [
                        { "type": "object", "properties": { "a": { "type": "string" } } },
                        { "type": "object", "properties": { "b": { "type": "string" } } }
                    ]
                }
            },
            "$defs": { "Cfg": { "type": "object", "properties": {} } }
        });

        let cleaned = SchemaCleanr::clean(schema, CleaningStrategy::OpenAI);

        assert!(cleaned["properties"]["cfg"].get("allOf").is_some());
        assert!(cleaned["properties"]["pick"].get("oneOf").is_some());
    }

    #[test]
    fn a_nested_all_of_is_folded_all_the_way_down() {
        let schema = json!({
            "type": "object",
            "required": ["a"],
            "allOf": [
                {
                    "properties": { "a": { "type": "string" } },
                    "allOf": [{ "properties": { "b": { "type": "number" } } }]
                },
                {
                    "allOf": [{ "required": ["c"], "properties": { "c": { "type": "boolean" } } }]
                }
            ]
        });

        let cleaned = SchemaCleanr::clean_for_gemini(schema);

        assert!(cleaned.get("allOf").is_none());
        for (name, ty) in [("a", "string"), ("b", "number"), ("c", "boolean")] {
            assert_eq!(cleaned["properties"][name]["type"], ty, "{name}");
        }
        let required = cleaned["required"].as_array().unwrap();
        assert_eq!(required.len(), 2);
        assert!(required.contains(&json!("c")));
    }

    #[test]
    fn needs_cleaning_sees_an_all_of_on_both_gemini_fields() {
        let schema = json!({
            "type": "object",
            "properties": { "cfg": { "allOf": [{ "$ref": "#/$defs/Cfg" }] } },
            "$defs": { "Cfg": { "type": "object", "properties": {} } }
        });

        for strategy in [CleaningStrategy::Gemini, CleaningStrategy::GeminiJsonSchema] {
            assert!(
                SchemaCleanr::needs_cleaning(&schema, strategy),
                "{strategy:?} must not skip the fold"
            );
        }
    }
}
