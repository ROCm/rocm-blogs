# Copyright © Advanced Micro Devices, Inc., or its affiliates.
#
# SPDX-License-Identifier: MIT

"""
Hybrid medical report processing pipeline with MCP tool calling + context injection fallback.

Primary: GLM-4.7-Flash uses MCP tools to extract medical context
Fallback: When no MCP tools are called, use MeSH context injection
Final: Poro-2 uses the enriched context to generate Finnish layman translation

This architecture provides:
- GLM-4.7-Flash's function calling capabilities for knowledge extraction
- MeSH vocabulary fallback when tool calling doesn't happen
- Poro-2's Finnish language capabilities for final translation
"""

import argparse
import asyncio
import json
import logging
import re
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from pydantic_ai import Agent
from pydantic_ai.models.openai import OpenAIChatModel
from pydantic_ai.providers.openai import OpenAIProvider
from mcp import StdioServerParameters

from poro_mcp import (
    MCPToolProvider,
    load_mcp_config,
    load_system_prompt_with_dictionary,
)
from mesh_parser import get_mesh_database, MeshDatabase, MeshConcept

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(asctime)s %(message)s")
LOGGER = logging.getLogger(__name__)

# Stage 1: GLM-4.7-Flash for tool use
DEFAULT_STAGE1_ENDPOINT = "http://localhost:8042/v1"
DEFAULT_STAGE1_MODEL = "zai-org/GLM-4.7-Flash"

# Stage 2: Poro-2 for Finnish generation
DEFAULT_STAGE2_ENDPOINT = "http://localhost:8000/v1"
DEFAULT_STAGE2_MODEL = "LumiOpen/Llama-Poro-2-70B-Instruct"

DEFAULT_STAGE1_SYSTEM_PROMPT = "stage1_system_prompt.txt"
DEFAULT_STAGE2_SYSTEM_PROMPT = "stage2_system_prompt.txt"
DEFAULT_MEDICAL_DICTIONARY = "medical_dictionary.txt"
DEFAULT_MCP_CONFIG = "mcp/rdf-explorer/mcp_config.json"
DEFAULT_TEMPERATURE = 0.2
DEFAULT_CONCURRENCY = 10

# Context limits
STAGE1_MAX_CONTEXT_TOKENS = 200000
STAGE1_MAX_INPUT_CHARS = 150000
CHARS_PER_TOKEN_ESTIMATE = 4


def estimate_tokens(text: str) -> int:
    """Rough token estimate based on character count."""
    return len(text) // CHARS_PER_TOKEN_ESTIMATE


def debug_context_size(
    text: str,
    system_prompt: str,
    tools_json: str | None = None,
    case_id: str = "unknown",
) -> Dict[str, Any]:
    """Debug helper to analyze context size components."""
    text_chars = len(text)
    text_tokens_est = estimate_tokens(text)

    system_chars = len(system_prompt)
    system_tokens_est = estimate_tokens(system_prompt)

    tools_chars = len(tools_json) if tools_json else 0
    tools_tokens_est = estimate_tokens(tools_json) if tools_json else 0

    total_chars = text_chars + system_chars + tools_chars
    total_tokens_est = text_tokens_est + system_tokens_est + tools_tokens_est

    debug_info = {
        "case_id": case_id,
        "text_chars": text_chars,
        "text_tokens_est": text_tokens_est,
        "system_prompt_chars": system_chars,
        "system_prompt_tokens_est": system_tokens_est,
        "tools_chars": tools_chars,
        "tools_tokens_est": tools_tokens_est,
        "total_chars": total_chars,
        "total_tokens_est": total_tokens_est,
        "exceeds_limit": total_tokens_est > STAGE1_MAX_CONTEXT_TOKENS,
    }

    LOGGER.debug(
        f"Context size for {case_id}: "
        f"text={text_chars} chars (~{text_tokens_est} tokens), "
        f"system={system_chars} chars (~{system_tokens_est} tokens), "
        f"tools={tools_chars} chars (~{tools_tokens_est} tokens), "
        f"TOTAL ~{total_tokens_est} tokens (limit: {STAGE1_MAX_CONTEXT_TOKENS})"
    )

    if total_tokens_est > STAGE1_MAX_CONTEXT_TOKENS:
        LOGGER.warning(
            f"{case_id} EXCEEDS LIMIT: ~{total_tokens_est} tokens > {STAGE1_MAX_CONTEXT_TOKENS} tokens"
        )

    return debug_info


def truncate_text_for_context(
    text: str, max_chars: int = STAGE1_MAX_INPUT_CHARS
) -> tuple[str, bool]:
    """Truncate text to fit within context limit."""
    if len(text) <= max_chars:
        return text, False
    truncated = text[:max_chars]
    truncated += "\n\n[... TEXT TRUNCATED DUE TO LENGTH ...]"
    return truncated, True


# =============================================================================
# MeSH Term Extractor (from poro_context_injection.py)
# =============================================================================


class MeshTermExtractor:
    """
    Extracts and looks up medical terms from text using the MeSH vocabulary.
    Used as fallback when MCP tool calling doesn't produce results.
    """

    def __init__(self, db: MeshDatabase):
        self.db = db
        self._build_term_index()

    def _build_term_index(self):
        """Build index of all known terms (lowercased) for fast matching."""
        self.known_terms: Dict[str, List[str]] = {}

        for mesh_id, concept in self.db.concepts.items():
            for lang in ["fi", "en"]:
                if lang in concept.pref_labels:
                    term = concept.pref_labels[lang].lower()
                    if term not in self.known_terms:
                        self.known_terms[term] = []
                    self.known_terms[term].append(mesh_id)

                for label in concept.alt_labels.get(lang, []):
                    term = label.lower()
                    if term not in self.known_terms:
                        self.known_terms[term] = []
                    self.known_terms[term].append(mesh_id)

    def extract_terms(
        self, text: str, max_terms: int = 30
    ) -> List[Tuple[str, MeshConcept]]:
        """Extract medical terms from text and return matching MeSH concepts."""
        found_terms: Dict[str, MeshConcept] = {}
        text_lower = text.lower()

        # Finnish case suffixes to strip
        fi_suffixes = [
            "ssa",
            "ssä",
            "sta",
            "stä",
            "lla",
            "llä",
            "lta",
            "ltä",
            "lle",
            "een",
            "iin",
            "na",
            "nä",
            "ksi",
            "tta",
            "ttä",
            "ine",
            "ia",
            "iä",
            "ja",
            "jä",
            "ien",
            "jen",
            "issa",
            "issä",
            "a",
            "ä",
            "n",
            "t",
        ]

        words = re.findall(r"\b[a-zäöåA-ZÄÖÅ][a-zäöå]+\b", text_lower)

        for word in words:
            if len(found_terms) >= max_terms:
                break
            if len(word) < 4:
                continue

            # Try exact match
            if word in self.known_terms:
                mesh_ids = self.known_terms[word]
                if mesh_ids:
                    concept = self.db.concepts.get(mesh_ids[0])
                    if concept and word not in found_terms:
                        found_terms[word] = concept
                        continue

            # Try stripping Finnish suffixes
            for suffix in fi_suffixes:
                if word.endswith(suffix) and len(word) > len(suffix) + 3:
                    stem = word[: -len(suffix)]
                    if stem in self.known_terms:
                        mesh_ids = self.known_terms[stem]
                        if mesh_ids:
                            concept = self.db.concepts.get(mesh_ids[0])
                            if concept and word not in found_terms:
                                found_terms[word] = concept
                                break

        # Look for Latin/medical patterns
        latin_patterns = [
            r"\b[a-zäöå]+itis\b",
            r"\b[a-zäöå]+iitti\b",
            r"\b[a-zäöå]+oma\b",
            r"\b[a-zäöå]+osis\b",
            r"\b[a-zäöå]+oosi\b",
            r"\b[a-zäöå]+emia\b",
            r"\b[a-zäöå]+eema\b",
            r"\b[a-zäöå]+kyymi\b",
        ]

        for pattern in latin_patterns:
            if len(found_terms) >= max_terms:
                break
            for match in re.finditer(pattern, text_lower):
                term = match.group()
                if term not in found_terms:
                    results = self.db.search_by_label(term, limit=1)
                    if results:
                        found_terms[term] = results[0]

        return list(found_terms.items())

    def build_context_from_terms(
        self,
        terms: List[Tuple[str, MeshConcept]],
    ) -> Dict[str, Any]:
        """
        Build a context dictionary from extracted terms.
        Returns format compatible with stage1_extract_context output.
        """
        if not terms:
            return {
                "key_terms": [],
                "abbreviations": {},
                "layman_translations": {},
                "context": "",
                "_fallback": True,
            }

        key_terms = []
        layman_translations = {}

        for original_term, concept in terms:
            key_terms.append(original_term)

            # Get Finnish preferred label
            fi_label = concept.pref_labels.get("fi")
            # en_label = concept.pref_labels.get("en")

            if fi_label and fi_label.lower() != original_term:
                layman_translations[original_term] = fi_label

            # Also include simpler Finnish alt labels
            for alt in concept.alt_labels.get("fi", [])[:1]:
                if alt.lower() != original_term and len(alt) < len(original_term):
                    layman_translations[original_term] = alt
                    break

        return {
            "key_terms": key_terms,
            "abbreviations": {},
            "layman_translations": layman_translations,
            "context": f"Löydetty {len(terms)} lääketieteellistä termiä MeSH-sanastosta (fallback-menetelmä)",
            "_fallback": True,
        }


# =============================================================================
# Tool Call Extraction
# =============================================================================


def extract_tool_calls_from_result(result: Any) -> list[Dict[str, Any]]:
    """Extract tool call information from pydantic_ai result."""
    tool_calls = []

    messages = None
    if hasattr(result, "all_messages"):
        messages = result.all_messages()
    elif hasattr(result, "messages"):
        messages = result.messages
    elif hasattr(result, "_messages"):
        messages = result._messages

    if not messages:
        return tool_calls

    for msg in messages:
        if hasattr(msg, "parts"):
            for part in msg.parts:
                part_type = type(part).__name__

                if part_type == "ToolCallPart" or hasattr(part, "tool_name"):
                    tool_name = getattr(
                        part, "tool_name", getattr(part, "name", "unknown")
                    )
                    tool_call_id = getattr(part, "tool_call_id", None)
                    tool_args = getattr(part, "args", getattr(part, "arguments", {}))

                    args_str = (
                        json.dumps(tool_args, ensure_ascii=False) if tool_args else "{}"
                    )

                    tool_calls.append(
                        {
                            "type": "call",
                            "tool_name": tool_name,
                            "tool_call_id": tool_call_id,
                            "args_summary": args_str[:500] + "..."
                            if len(args_str) > 500
                            else args_str,
                        }
                    )

        elif hasattr(msg, "tool_calls") and msg.tool_calls:
            for tc in msg.tool_calls:
                tool_name = getattr(tc, "function", {}).get(
                    "name", getattr(tc, "name", "unknown")
                )
                tool_call_id = getattr(tc, "id", None)

                tool_calls.append(
                    {
                        "type": "call",
                        "tool_name": tool_name,
                        "tool_call_id": tool_call_id,
                    }
                )

    return tool_calls


def check_tools_were_called(tool_calls: List[Dict[str, Any]]) -> bool:
    """
    Check if any actual MCP tools were called (not just internal message parts).

    Returns True if there's at least one tool call with a non-null tool_call_id
    and the tool_name is not 'unknown'.
    """
    for tc in tool_calls:
        if tc.get("type") == "call":
            tool_name = tc.get("tool_name", "unknown")
            tool_call_id = tc.get("tool_call_id")

            # Valid tool call: has an ID and a real tool name
            if tool_call_id is not None and tool_name != "unknown":
                return True

    return False


# =============================================================================
# Stage 1: MCP Tool Calling
# =============================================================================


async def stage1_extract_context_mcp(
    text: str,
    tool_wrappers: list,
    stage1_system_prompt: str,
    endpoint: str,
    model: str,
    temperature: float = 0.2,
    case_id: str = "unknown",
) -> Tuple[Dict[str, Any], List[Dict[str, Any]]]:
    """
    Stage 1 with MCP tools: Use GLM-4.7-Flash to extract medical context.

    Returns:
        Tuple of (extracted_context, tool_calls_info)
    """
    LOGGER.info(f"Stage 1 (MCP): Extracting context for {case_id}")

    openai_model = OpenAIChatModel(
        model,
        provider=OpenAIProvider(base_url=endpoint, api_key="dummy"),
    )
    # Debug: Log context size (only when DEBUG logging is enabled)
    if LOGGER.isEnabledFor(logging.DEBUG):
        try:
            tools_json = json.dumps([str(t) for t in tool_wrappers], ensure_ascii=False)
        except Exception:
            tools_json = ""

        debug_info = debug_context_size(
            text=text,
            system_prompt=stage1_system_prompt,
            tools_json=tools_json,
            case_id=case_id,
        )
    else:
        # Skip expensive debug calculations when debug logging is disabled
        debug_info = {"total_tokens_est": 0}

    agent = Agent(
        model=openai_model,
        system_prompt=stage1_system_prompt,
        tools=tool_wrappers,
        model_settings={"temperature": temperature},
    )

    try:
        result = await agent.run(text)

        # Extract the result
        if hasattr(result, "data"):
            context = result.data
        elif hasattr(result, "output"):
            context = result.output
        else:
            context = str(result)

        # Try to parse as JSON (handle markdown code blocks)
        if isinstance(context, str):
            # Strip markdown code blocks if present
            cleaned = context.strip()
            if cleaned.startswith("```"):
                # Remove opening ```json or ``` and closing ```
                lines = cleaned.split("\n")
                # Remove first line (```json or ```) and last line (```)
                if lines[0].startswith("```"):
                    lines = lines[1:]
                if lines and lines[-1].strip() == "```":
                    lines = lines[:-1]
                cleaned = "\n".join(lines)

            try:
                context = json.loads(cleaned)
            except json.JSONDecodeError:
                context = {"context": context, "_raw_response": True}

        # Extract tool call information
        tool_calls_info = extract_tool_calls_from_result(result)

        if isinstance(context, dict):
            context["_debug"] = {
                "method": "mcp_tools",
                "tool_calls": tool_calls_info,
                "text_chars": len(text),
                "estimated_tokens": debug_info["total_tokens_est"],
            }

        return context, tool_calls_info

    except Exception as e:
        LOGGER.exception(f"Stage 1 (MCP) failed for {case_id}: {e}")
        return {
            "error": str(e),
            "context": "",
            "_debug": {
                "method": "mcp_tools",
                "tool_calls": [],
                "text_chars": len(text),
                "estimated_tokens": debug_info["total_tokens_est"],
            },
        }, []


# =============================================================================
# Stage 1 Fallback: Context Injection
# =============================================================================


def stage1_fallback_context_injection(
    text: str,
    term_extractor: MeshTermExtractor,
    case_id: str = "unknown",
) -> Dict[str, Any]:
    """
    Fallback Stage 1: Use MeSH term extraction when MCP tools weren't called.
    """
    LOGGER.info(f"Stage 1 (Fallback): Using context injection for {case_id}")

    terms = term_extractor.extract_terms(text)
    context = term_extractor.build_context_from_terms(terms)

    context["_debug"] = {
        "method": "context_injection_fallback",
        "terms_found": len(terms),
        "terms": [t[0] for t in terms],
    }

    LOGGER.info(f"Stage 1 (Fallback) complete for {case_id}: Found {len(terms)} terms")
    return context


# =============================================================================
# Stage 2: Generate Finnish with Poro-2
# =============================================================================


async def stage2_generate_finnish(
    original_text: str,
    extracted_context: Dict[str, Any],
    system_prompt: str,
    endpoint: str,
    model: str,
    temperature: float = 0.2,
) -> Dict[str, Any]:
    """Stage 2: Use Poro-2 to generate Finnish layman translation."""
    LOGGER.info("Stage 2: Generating Finnish translation with Poro-2")

    enriched_prompt = system_prompt + "\n\n"

    if extracted_context.get("layman_translations"):
        enriched_prompt += "Lisätietoja termeistä:\n"
        for med_term, layman_term in extracted_context["layman_translations"].items():
            enriched_prompt += f"- {med_term} = {layman_term}\n"
        enriched_prompt += "\n"

    if extracted_context.get("abbreviations"):
        enriched_prompt += "Lyhenteiden selvennykset:\n"
        for abbr, expansion in extracted_context["abbreviations"].items():
            enriched_prompt += f"- {abbr} = {expansion}\n"
        enriched_prompt += "\n"

    if extracted_context.get("context"):
        enriched_prompt += f"Lisäkonteksti: {extracted_context['context']}\n\n"

    openai_model = OpenAIChatModel(
        model, provider=OpenAIProvider(base_url=endpoint, api_key="dummy")
    )
    agent = Agent(
        model=openai_model,
        system_prompt=enriched_prompt,
        model_settings={"temperature": temperature},
    )

    try:
        result = await agent.run(original_text)

        if hasattr(result, "data"):
            completion = result.data
        elif hasattr(result, "output"):
            completion = result.output
        else:
            completion = str(result)

        usage = None
        if hasattr(result, "usage"):
            usage_obj = result.usage
            if usage_obj is not None and not isinstance(usage_obj, dict):
                if hasattr(usage_obj, "model_dump"):
                    usage = usage_obj.model_dump()
                elif hasattr(usage_obj, "__dict__"):
                    usage = usage_obj.__dict__
                else:
                    usage = str(usage_obj)
            else:
                usage = usage_obj

        return {"completion": completion, "usage": usage}

    except Exception as e:
        LOGGER.exception(f"Stage 2 failed: {e}")
        return {"completion": None, "usage": None, "error": str(e)}


# =============================================================================
# Fallback & Metadata Helpers
# =============================================================================


def determine_fallback_need(
    extracted_context: Dict[str, Any],
) -> Tuple[bool, Optional[str]]:
    """Decide whether fallback context injection is needed and why.

    With the medical dictionary now in the prompt, the model can correctly
    translate terms without calling tools. Fallback is only needed when:
    1. Stage 1 had an error
    2. Stage 1 returned empty results (no key_terms and no layman_translations)
    """
    stage1_error = (
        "error" in extracted_context and extracted_context.get("context") == ""
    )

    if stage1_error:
        return True, "stage1_error"

    # Check if stage1 produced useful content
    # The context might be a JSON string, so check for key_terms or layman_translations
    has_key_terms = bool(extracted_context.get("key_terms"))
    has_translations = bool(extracted_context.get("layman_translations"))

    # Also check if the context field contains parsed JSON with terms
    context_str = extracted_context.get("context", "")
    if isinstance(context_str, str) and "key_terms" in context_str:
        # Context contains JSON - stage1 produced useful output
        has_key_terms = True

    if not has_key_terms and not has_translations:
        return True, "empty_results"

    return False, None


def count_terms_found(extracted_context: Dict[str, Any]) -> int:
    """Count the number of layman translations found in the extracted context."""
    if not isinstance(extracted_context, dict):
        return 0
    return len(extracted_context.get("layman_translations", {}))


def apply_fallback_context(
    extracted_context: Dict[str, Any],
    use_fallback: bool,
    fallback_reason: Optional[str],
    text: str,
    term_extractor: Optional[MeshTermExtractor],
    case_id: str,
) -> Dict[str, Any]:
    """Run fallback context injection and merge/replace the extracted context."""
    if not use_fallback or term_extractor is None:
        return extracted_context

    LOGGER.info(f"{case_id}: Using fallback (reason: {fallback_reason})")
    fallback_context = stage1_fallback_context_injection(
        text=text,
        term_extractor=term_extractor,
        case_id=case_id,
    )

    stage1_error = fallback_reason == "stage1_error"
    if stage1_error:
        return fallback_context

    # Merge: keep MCP results but add fallback terms
    if "layman_translations" not in extracted_context:
        extracted_context["layman_translations"] = {}
    extracted_context["layman_translations"].update(
        fallback_context.get("layman_translations", {})
    )
    extracted_context["_fallback_used"] = True
    extracted_context["_fallback_reason"] = fallback_reason
    extracted_context["_fallback_terms"] = fallback_context.get("_debug", {}).get(
        "terms", []
    )
    return extracted_context


def add_context_metadata(
    extracted_context: Dict[str, Any],
    *,
    was_truncated: bool,
    original_text_len: int,
    tools_were_called: bool,
    use_fallback: bool,
    fallback_reason: Optional[str],
) -> None:
    """Attach diagnostic metadata fields to the extracted context dict (in-place)."""
    if not isinstance(extracted_context, dict):
        return
    extracted_context["_was_truncated"] = was_truncated
    extracted_context["_original_text_chars"] = original_text_len
    extracted_context["_tools_were_called"] = tools_were_called
    extracted_context["_used_fallback"] = use_fallback
    if fallback_reason:
        extracted_context["_fallback_reason"] = fallback_reason


# =============================================================================
# Main Processing Pipeline
# =============================================================================


async def process_report_hybrid(
    text: str,
    case_id: str,
    tool_wrappers: list,
    term_extractor: Optional[MeshTermExtractor],
    stage1_endpoint: str,
    stage1_model: str,
    stage1_system_prompt: str,
    stage2_endpoint: str,
    stage2_model: str,
    stage2_system_prompt: str,
    temperature: float = 0.2,
    truncate_if_needed: bool = True,
    max_input_chars: int = STAGE1_MAX_INPUT_CHARS,
) -> Dict[str, Any]:
    """
    Process a single medical report with hybrid approach:
    1. Try MCP tool calling
    2. If no tools were called, fallback to context injection
    3. Generate Finnish translation
    """

    # Truncate if needed (for Stage 1 only; keep original `text` for Stage 2/output)
    original_text_len = len(text)
    stage1_text = text
    was_truncated = False
    if truncate_if_needed and len(stage1_text) > max_input_chars:
        stage1_text, was_truncated = truncate_text_for_context(
            stage1_text, max_input_chars
        )
        LOGGER.warning(
            f"{case_id}: Text truncated from {original_text_len} to {len(stage1_text)} chars"
        )

    # Stage 1: Try MCP tool calling (operate on possibly truncated Stage 1 text)
    extracted_context, tool_calls_info = await stage1_extract_context_mcp(
        text=stage1_text,
        tool_wrappers=tool_wrappers,
        stage1_system_prompt=stage1_system_prompt,
        endpoint=stage1_endpoint,
        model=stage1_model,
        temperature=temperature,
        case_id=case_id,
    )

    tools_were_called = check_tools_were_called(tool_calls_info)
    use_fallback, fallback_reason = determine_fallback_need(extracted_context)

    extracted_context = apply_fallback_context(
        extracted_context=extracted_context,
        use_fallback=use_fallback,
        fallback_reason=fallback_reason,
        text=text,
        term_extractor=term_extractor,
        case_id=case_id,
    )

    add_context_metadata(
        extracted_context,
        was_truncated=was_truncated,
        original_text_len=original_text_len,
        tools_were_called=tools_were_called,
        use_fallback=use_fallback,
        fallback_reason=fallback_reason,
    )

    # Stage 2: Generate Finnish
    stage2_result = await stage2_generate_finnish(
        original_text=text,
        extracted_context=extracted_context,
        system_prompt=stage2_system_prompt,
        endpoint=stage2_endpoint,
        model=stage2_model,
        temperature=temperature,
    )

    terms_found = count_terms_found(extracted_context)

    return {
        "case": case_id,
        "text": text,
        "stage1_context": extracted_context,
        "completion": stage2_result.get("completion"),
        "usage": stage2_result.get("usage"),
        "error": stage2_result.get("error"),
        "status": "ok" if not stage2_result.get("error") else "error",
        # Debug fields at top level for easy access
        "tools_called": tools_were_called,
        "used_fallback": use_fallback,
        "fallback_reason": fallback_reason,
        "terms_found": terms_found,
        "was_truncated": was_truncated,
    }


def init_mesh_term_extractor(
    mesh_ttl: Optional[Path] = None,
) -> Optional[MeshTermExtractor]:
    """Load the MeSH vocabulary and return a term extractor, or None if unavailable."""
    mesh_path = mesh_ttl or Path(__file__).parent / "mesh-skos.ttl"
    if not mesh_path.exists():
        LOGGER.warning(f"MeSH file not found at {mesh_path}, fallback disabled")
        return None

    LOGGER.info(f"Loading MeSH vocabulary for fallback from {mesh_path}")
    db = get_mesh_database(mesh_path)
    extractor = MeshTermExtractor(db)
    LOGGER.info(f"MeSH fallback ready: {len(extractor.known_terms)} terms indexed")
    return extractor


# =============================================================================
# Main
# =============================================================================


async def async_main(args):
    """Main async function."""

    # Load inputs
    stage1_system_prompt = load_system_prompt_with_dictionary(
        args.stage1_system_prompt_file, args.medical_dictionary_file
    )
    stage2_system_prompt = load_system_prompt_with_dictionary(
        args.stage2_system_prompt_file, args.medical_dictionary_file
    )
    input_data = json.loads(Path(args.input).read_text(encoding="utf-8"))
    if not isinstance(input_data, list):
        raise ValueError("Input JSON must be a list of objects.")

    LOGGER.info(f"Loaded {len(input_data)} items from {args.input}")
    LOGGER.info(f"Stage 1 (MCP): {args.stage1_model} @ {args.stage1_endpoint}")
    LOGGER.info(f"Stage 2: {args.stage2_model} @ {args.stage2_endpoint}")
    LOGGER.info(f"MCP config: {args.mcp_config}")

    # Initialize MCP provider
    mcp_provider = MCPToolProvider()
    mcp_config = load_mcp_config(args.mcp_config)

    for server_name, server_config in mcp_config.get("mcpServers", {}).items():
        try:
            server_params = StdioServerParameters(
                command=server_config["command"],
                args=server_config.get("args", []),
                env=server_config.get("env"),
            )
            await mcp_provider.connect_server(server_name, server_params)
        except Exception:
            LOGGER.exception(f"Failed to connect to {server_name}")
            raise

    # Cache tool wrappers
    tool_wrappers = await mcp_provider.get_tool_wrappers()
    LOGGER.info(f"MCP tools loaded: {len(tool_wrappers)} tools")

    term_extractor = init_mesh_term_extractor(args.mesh_ttl)

    # Process items
    sem = asyncio.Semaphore(args.concurrency)
    out_f = args.output.open("w", encoding="utf-8")

    stats = {"total": 0, "mcp_used": 0, "fallback_used": 0, "errors": 0}

    async def worker(idx: int, item: Dict[str, Any]):
        async with sem:
            result = await process_report_hybrid(
                text=item["text"],
                case_id=item.get("case", f"Case{idx}"),
                tool_wrappers=tool_wrappers,
                term_extractor=term_extractor,
                stage1_endpoint=args.stage1_endpoint,
                stage1_model=args.stage1_model,
                stage1_system_prompt=stage1_system_prompt,
                stage2_endpoint=args.stage2_endpoint,
                stage2_model=args.stage2_model,
                stage2_system_prompt=stage2_system_prompt,
                temperature=args.temperature,
                truncate_if_needed=args.truncate,
                max_input_chars=args.max_input_chars,
            )

        # Update stats
        stats["total"] += 1
        if result.get("used_fallback"):
            stats["fallback_used"] += 1
        elif result.get("tools_called"):
            stats["mcp_used"] += 1
        if result.get("status") == "error":
            stats["errors"] += 1

        record = {"index": idx, **result}
        out_f.write(json.dumps(record, ensure_ascii=False) + "\n")
        out_f.flush()

        LOGGER.info(
            f"Processed {idx+1}/{len(input_data)}: {item.get('case', 'unknown')} "
            f"[{'MCP' if result.get('tools_called') else 'Fallback'}]"
        )

    try:
        tasks = [
            asyncio.create_task(worker(i, item)) for i, item in enumerate(input_data)
        ]
        await asyncio.gather(*tasks)
    finally:
        out_f.close()
        await mcp_provider.close()

    LOGGER.info(f"Processing complete. Output: {args.output}")
    LOGGER.info(
        f"Stats: {stats['total']} total, {stats['mcp_used']} MCP, "
        f"{stats['fallback_used']} fallback, {stats['errors']} errors"
    )

    return stats


# =============================================================================
# Postprocessing
# =============================================================================


def load_records(path: Path) -> List[Dict[str, Any]]:
    """Load JSONL records from file."""
    raw = path.read_text(encoding="utf-8").strip()
    records = [json.loads(line) for line in raw.splitlines() if line.strip()]
    return records


def postprocess(records: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Filter to essential fields and sort by index."""
    filtered = []
    for r in records:
        item = {
            "index": r["index"],
            "case": r["case"],
            "text": r["text"],
            "completion": r.get("completion"),
            "status": r.get("status"),
            # Debug fields (now at top level)
            "tools_called": r.get("tools_called", False),
            "used_fallback": r.get("used_fallback", False),
            "fallback_reason": r.get("fallback_reason"),
            "terms_found": r.get("terms_found", 0),
            "was_truncated": r.get("was_truncated", False),
        }
        filtered.append(item)

    filtered.sort(key=lambda r: int(r["index"]))
    return filtered


def write_output(records: List[Dict[str, Any]], path: Path) -> None:
    """Write records to JSON file."""
    with path.open("w", encoding="utf-8") as f:
        json.dump(records, f, ensure_ascii=False, indent=2)


def main():
    parser = argparse.ArgumentParser(
        description="Hybrid medical report processing: MCP tools + context injection fallback"
    )
    parser.add_argument("--input", "-i", type=Path, required=True)
    parser.add_argument("--output", "-o", type=Path, required=True)
    parser.add_argument("--stage1-endpoint", type=str, default=DEFAULT_STAGE1_ENDPOINT)
    parser.add_argument("--stage1-model", type=str, default=DEFAULT_STAGE1_MODEL)
    parser.add_argument(
        "--stage1-system-prompt-file", type=Path, default=DEFAULT_STAGE1_SYSTEM_PROMPT
    )
    parser.add_argument("--stage2-endpoint", type=str, default=DEFAULT_STAGE2_ENDPOINT)
    parser.add_argument("--stage2-model", type=str, default=DEFAULT_STAGE2_MODEL)
    parser.add_argument(
        "--stage2-system-prompt-file", type=Path, default=DEFAULT_STAGE2_SYSTEM_PROMPT
    )
    parser.add_argument(
        "--medical-dictionary-file", type=Path, default=DEFAULT_MEDICAL_DICTIONARY
    )
    parser.add_argument("--mcp-config", type=Path, default=DEFAULT_MCP_CONFIG)
    parser.add_argument(
        "--mesh-ttl", type=Path, default=None, help="Path to mesh-skos.ttl for fallback"
    )
    parser.add_argument("--temperature", type=float, default=DEFAULT_TEMPERATURE)
    parser.add_argument("--concurrency", type=int, default=DEFAULT_CONCURRENCY)
    parser.add_argument("--truncate", action="store_true", default=True)
    parser.add_argument("--no-truncate", action="store_false", dest="truncate")
    parser.add_argument("--max-input-chars", type=int, default=STAGE1_MAX_INPUT_CHARS)
    parser.add_argument(
        "--debug-context",
        action="store_true",
        help="Enable extra debug logging for context size analysis",
    )

    args = parser.parse_args()

    # Enable DEBUG level if --debug-context is set
    if args.debug_context:
        logging.getLogger().setLevel(logging.DEBUG)
        LOGGER.setLevel(logging.DEBUG)
        LOGGER.debug("Debug context logging enabled")

    # Run main processing
    stats = asyncio.run(async_main(args))

    # Postprocess results
    results_path = args.output.with_suffix(".postprocessed.json")
    LOGGER.info(f"Postprocessing results to: {results_path}")

    records = load_records(args.output)
    processed = postprocess(records)
    write_output(processed, results_path)

    # Print final summary
    mcp_pct = (stats["mcp_used"] / stats["total"] * 100) if stats["total"] > 0 else 0
    fallback_pct = (
        (stats["fallback_used"] / stats["total"] * 100) if stats["total"] > 0 else 0
    )

    LOGGER.info(
        f"Done! {stats['total']} records processed: "
        f"{stats['mcp_used']} ({mcp_pct:.1f}%) used MCP tools, "
        f"{stats['fallback_used']} ({fallback_pct:.1f}%) used fallback"
    )


if __name__ == "__main__":
    main()
