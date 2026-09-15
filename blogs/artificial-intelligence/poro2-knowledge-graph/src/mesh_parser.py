# Copyright © Advanced Micro Devices, Inc., or its affiliates.
#
# SPDX-License-Identifier: MIT

"""
MeSH SKOS TTL Parser

Parses the mesh-skos.ttl file and provides an indexed, searchable database
of medical terminology concepts.

NOTE: This uses a custom regex-based parser instead of rdflib for performance:
- mesh-skos.ttl is 654K lines (~30K concepts)
- Custom parser: ~3-5 sec load, ~200-400MB RAM
- rdflib would be 10-20x slower and use 3-5x more memory
- We only need specific SKOS predicates (prefLabel, altLabel, scopeNote,
  broader, etc.), not full RDF querying or SPARQL support
- Custom indices (_label_index, _lang_label_index) are built during parsing
  for fast substring search—rdflib would require a second pass anyway
"""

import re
import logging
from pathlib import Path
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Set

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(asctime)s %(message)s")
LOGGER = logging.getLogger(__name__)


@dataclass
class MeshConcept:
    """Represents a MeSH concept from the SKOS vocabulary."""

    uri: str
    mesh_id: str
    pref_labels: Dict[str, str] = field(default_factory=dict)  # lang -> label
    alt_labels: Dict[str, List[str]] = field(default_factory=dict)  # lang -> [labels]
    scope_note: Optional[str] = None
    broader: List[str] = field(default_factory=list)  # list of broader concept URIs
    history_note: Optional[str] = None
    note: Optional[str] = None
    created: Optional[str] = None
    modified: Optional[str] = None
    exact_match: Optional[str] = None
    types: List[str] = field(default_factory=list)

    def get_all_labels(self, lang: Optional[str] = None) -> List[str]:
        """Get all labels (pref + alt) for a given language or all languages."""
        labels = []
        if lang:
            if lang in self.pref_labels:
                labels.append(self.pref_labels[lang])
            labels.extend(self.alt_labels.get(lang, []))
        else:
            labels.extend(self.pref_labels.values())
            for alt_list in self.alt_labels.values():
                labels.extend(alt_list)
        return labels

    def to_dict(self) -> dict:
        """Convert to dictionary for JSON serialization."""
        return {
            "uri": self.uri,
            "mesh_id": self.mesh_id,
            "pref_labels": self.pref_labels,
            "alt_labels": self.alt_labels,
            "scope_note": self.scope_note,
            "broader": self.broader,
            "history_note": self.history_note,
            "note": self.note,
            "created": self.created,
            "modified": self.modified,
            "exact_match": self.exact_match,
            "types": self.types,
        }


class MeshDatabase:
    """
    In-memory database of MeSH concepts parsed from SKOS TTL.

    Provides efficient searching by label and hierarchical navigation.
    """

    def __init__(self):
        self.concepts: Dict[str, MeshConcept] = {}  # mesh_id -> concept
        self._label_index: Dict[str, Set[str]] = {}
        self._lang_label_index: Dict[str, Dict[str, Set[str]]] = {}
        self._loaded = False

    def load_from_ttl(self, ttl_path: Path) -> None:
        """Parse and load concepts from a SKOS TTL file."""
        LOGGER.info(f"Loading MeSH SKOS from {ttl_path}...")

        content = ttl_path.read_text(encoding="utf-8")

        # Parse prefixes
        prefixes = {}
        for match in re.finditer(r"@prefix\s+(\w+):\s+<([^>]+)>\s*\.", content):
            prefixes[match.group(1)] = match.group(2)

        # Split into concept blocks
        # Each concept starts with mesh:DXXXXXX and ends before the next mesh: or end
        concept_pattern = re.compile(
            r"(mesh:D\d+)\s+(.*?)(?=\nmesh:D\d+\s|$)", re.DOTALL
        )

        count = 0
        for match in concept_pattern.finditer(content):
            mesh_ref = match.group(1)
            block = match.group(2)

            mesh_id = mesh_ref.replace("mesh:", "")
            concept = self._parse_concept_block(mesh_id, block)

            if concept:
                self.concepts[mesh_id] = concept
                self._index_concept(concept)
                count += 1

                if count % 10000 == 0:
                    LOGGER.info(f"Loaded {count} concepts...")

        self._loaded = True
        LOGGER.info(f"Loaded {len(self.concepts)} MeSH concepts")

    def _parse_concept_block(self, mesh_id: str, block: str) -> Optional[MeshConcept]:
        """Parse a single concept block from TTL."""
        concept = MeshConcept(
            uri=f"http://www.yso.fi/onto/mesh/{mesh_id}", mesh_id=mesh_id
        )

        # Parse types (a meshv:..., skos:Concept)
        type_pattern = re.compile(r"a\s+([^;,]+)")
        for match in type_pattern.finditer(block):
            type_str = match.group(1).strip()
            for t in type_str.split(","):
                t = t.strip()
                if t:
                    concept.types.append(t)

        # Parse all quoted strings with language tags (handles multi-line)
        # Pattern matches "text"@lang anywhere in the block
        label_pattern = re.compile(r'"([^"]+)"@(\w+)')

        # Find prefLabel section (from skos:prefLabel to next skos: or ;)
        pref_section = re.search(r"skos:prefLabel\s+(.*?)(?=skos:|$)", block, re.DOTALL)
        if pref_section:
            for match in label_pattern.finditer(pref_section.group(1)):
                label, lang = match.groups()
                concept.pref_labels[lang] = label

        # Find altLabel section
        alt_section = re.search(r"skos:altLabel\s+(.*?)(?=skos:|$)", block, re.DOTALL)
        if alt_section:
            for match in label_pattern.finditer(alt_section.group(1)):
                label, lang = match.groups()
                if lang not in concept.alt_labels:
                    concept.alt_labels[lang] = []
                concept.alt_labels[lang].append(label)

        # Parse scopeNote
        scope_pattern = re.compile(r'skos:scopeNote\s+"([^"]+)"@en')
        scope_match = scope_pattern.search(block)
        if scope_match:
            concept.scope_note = scope_match.group(1)

        # Parse broader
        broader_pattern = re.compile(r"skos:broader\s+mesh:(D\d+)")
        for match in broader_pattern.finditer(block):
            concept.broader.append(match.group(1))

        # Parse historyNote
        history_pattern = re.compile(r'skos:historyNote\s+"([^"]+)"@en')
        history_match = history_pattern.search(block)
        if history_match:
            concept.history_note = history_match.group(1)

        # Parse note
        note_pattern = re.compile(r'skos:note\s+"([^"]+)"@en')
        note_match = note_pattern.search(block)
        if note_match:
            concept.note = note_match.group(1)

        # Parse dates
        created_pattern = re.compile(r'dct:created\s+"([^"]+)"')
        created_match = created_pattern.search(block)
        if created_match:
            concept.created = created_match.group(1)

        modified_pattern = re.compile(r'dct:modified\s+"([^"]+)"')
        modified_match = modified_pattern.search(block)
        if modified_match:
            concept.modified = modified_match.group(1)

        # Parse exactMatch
        exact_pattern = re.compile(r"skos:exactMatch\s+<([^>]+)>")
        exact_match = exact_pattern.search(block)
        if exact_match:
            concept.exact_match = exact_match.group(1)

        return concept

    def _index_concept(self, concept: MeshConcept) -> None:
        """Add concept to search indices."""
        # Index all labels (normalized to lowercase)
        for lang, label in concept.pref_labels.items():
            self._add_to_index(label, concept.mesh_id, lang)

        for lang, labels in concept.alt_labels.items():
            for label in labels:
                self._add_to_index(label, concept.mesh_id, lang)

    def _add_to_index(self, label: str, mesh_id: str, lang: str) -> None:
        """Add a label to the search indices."""
        normalized = label.lower().strip()

        # Global index
        if normalized not in self._label_index:
            self._label_index[normalized] = set()
        self._label_index[normalized].add(mesh_id)

        # Language-specific index
        if lang not in self._lang_label_index:
            self._lang_label_index[lang] = {}
        if normalized not in self._lang_label_index[lang]:
            self._lang_label_index[lang][normalized] = set()
        self._lang_label_index[lang][normalized].add(mesh_id)

    def search_by_label(
        self,
        query: str,
        lang: Optional[str] = None,
        exact: bool = False,
        limit: int = 20,
    ) -> List[MeshConcept]:
        """
        Search concepts by label.

        Args:
            query: Search term
            lang: Optional language filter (en, fi, se)
            exact: If True, only return exact matches
            limit: Maximum number of results

        Returns:
            List of matching MeshConcept objects
        """
        query_normalized = query.lower().strip()
        results: Set[str] = set()

        if exact:
            # Exact match
            if lang:
                index = self._lang_label_index.get(lang, {})
                results = index.get(query_normalized, set())
            else:
                results = self._label_index.get(query_normalized, set())
        else:
            # Substring match
            if lang:
                index = self._lang_label_index.get(lang, {})
            else:
                index = self._label_index

            for label, mesh_ids in index.items():
                if query_normalized in label:
                    results.update(mesh_ids)
                    if len(results) >= limit * 2:  # Get extra for ranking
                        break

        # Convert to concepts and rank by relevance
        concepts = []
        for mesh_id in results:
            if mesh_id in self.concepts:
                concepts.append(self.concepts[mesh_id])

        # Sort by relevance (exact matches first, then by label length)
        def relevance_key(c: MeshConcept) -> tuple:
            all_labels = c.get_all_labels(lang)
            has_exact = any(label.lower() == query_normalized for label in all_labels)
            starts_with = any(
                label.lower().startswith(query_normalized) for label in all_labels
            )
            min_len = min((len(label) for label in all_labels), default=999)
            return (not has_exact, not starts_with, min_len)

        concepts.sort(key=relevance_key)
        return concepts[:limit]

    def get_concept(self, mesh_id: str) -> Optional[MeshConcept]:
        """Get a concept by its MeSH ID (e.g., 'D000006')."""
        # Handle various input formats
        clean_id = mesh_id.replace("mesh:", "").strip()
        if not clean_id.startswith("D"):
            clean_id = f"D{clean_id}"
        return self.concepts.get(clean_id)

    def get_broader_concepts(self, mesh_id: str) -> List[MeshConcept]:
        """Get all broader (parent) concepts for a given concept."""
        concept = self.get_concept(mesh_id)
        if not concept:
            return []

        results = []
        for broader_id in concept.broader:
            broader_concept = self.concepts.get(broader_id)
            if broader_concept:
                results.append(broader_concept)
        return results

    def get_narrower_concepts(self, mesh_id: str, limit: int = 50) -> List[MeshConcept]:
        """Get narrower (child) concepts for a given concept."""
        clean_id = mesh_id.replace("mesh:", "").strip()
        results = []

        for concept in self.concepts.values():
            if clean_id in concept.broader:
                results.append(concept)
                if len(results) >= limit:
                    break

        return results

    def search_by_scope(self, query: str, limit: int = 20) -> List[MeshConcept]:
        """
        Search concepts by their scope note (definition).

        Args:
            query: Search term to find in definitions
            limit: Maximum number of results

        Returns:
            List of matching MeshConcept objects
        """
        query_lower = query.lower()
        results = []

        for concept in self.concepts.values():
            if concept.scope_note and query_lower in concept.scope_note.lower():
                results.append(concept)
                if len(results) >= limit:
                    break

        return results

    def get_hierarchy_path(self, mesh_id: str) -> List[List[MeshConcept]]:
        """
        Get all paths from a concept to root concepts.

        Returns list of paths, where each path is a list of concepts
        from the given concept up to a root.
        """
        concept = self.get_concept(mesh_id)
        if not concept:
            return []

        if not concept.broader:
            return [[concept]]

        paths = []
        for broader_id in concept.broader:
            parent_paths = self.get_hierarchy_path(broader_id)
            for path in parent_paths:
                paths.append([concept] + path)

        return paths if paths else [[concept]]


# Singleton instance for caching
_db_instance: Optional[MeshDatabase] = None


def get_mesh_database(ttl_path: Optional[Path] = None) -> MeshDatabase:
    """
    Get or create the MeSH database singleton.

    Args:
        ttl_path: Path to mesh-skos.ttl file. Only needed on first call.

    Returns:
        MeshDatabase instance
    """
    global _db_instance

    if _db_instance is None:
        if ttl_path is None:
            # Default path relative to this file
            ttl_path = Path(__file__).parent / "mesh-skos.ttl"

        _db_instance = MeshDatabase()
        _db_instance.load_from_ttl(ttl_path)

    return _db_instance


if __name__ == "__main__":
    # Test the parser
    import sys

    ttl_path = Path(__file__).parent / "mesh-skos.ttl"
    if not ttl_path.exists():
        print(f"Error: {ttl_path} not found")
        sys.exit(1)

    db = get_mesh_database(ttl_path)

    # Test searches
    print("\n=== Search for 'diabetes' ===")
    results = db.search_by_label("diabetes", limit=5)
    for c in results:
        print(f"  {c.mesh_id}: {c.pref_labels.get('en', 'N/A')}")
        if c.scope_note:
            print(f"    -> {c.scope_note[:100]}...")

    print("\n=== Search for 'sydän' (Finnish for heart) ===")
    results = db.search_by_label("sydän", lang="fi", limit=5)
    for c in results:
        print(
            f"  {c.mesh_id}: {c.pref_labels.get('fi', 'N/A')} ({c.pref_labels.get('en', 'N/A')})"
        )

    print("\n=== Get concept D000006 (Abdomen, Acute) ===")
    concept = db.get_concept("D000006")
    if concept:
        print(f"  EN: {concept.pref_labels.get('en')}")
        print(f"  FI: {concept.pref_labels.get('fi')}")
        print(f"  Scope: {concept.scope_note}")
        print(f"  Broader: {concept.broader}")

    print("\n=== Get broader concepts for D000006 ===")
    broader = db.get_broader_concepts("D000006")
    for c in broader:
        print(f"  {c.mesh_id}: {c.pref_labels.get('en', 'N/A')}")
