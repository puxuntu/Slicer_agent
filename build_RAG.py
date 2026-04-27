"""
Standalone script to build the dense vector retrieval index.
Run this to create/update the FAISS vector index under Resources/Code_RAG/v1/.
"""
import os
import sys
import logging
import time

logging.basicConfig(level=logging.INFO, format='%(asctime)s [%(levelname)s] %(message)s')
logger = logging.getLogger(__name__)

# Add lib to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# Direct import to avoid loading Slicer-dependent modules via __init__.py
import importlib.util
_spec = importlib.util.spec_from_file_location(
    "SkillIndexer",
    os.path.join(os.path.dirname(os.path.abspath(__file__)), "SlicerAIAgentLib", "SkillIndexer.py")
)
SkillIndexer = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(SkillIndexer)
IndexBuilder = SkillIndexer.IndexBuilder

SKILL_PATH = os.path.join(
    os.path.dirname(os.path.abspath(__file__)),
    "Resources", "Skills", "slicer-skill-full"
)

if __name__ == "__main__":
    if not os.path.isdir(SKILL_PATH):
        logger.error(f"Skill directory not found: {SKILL_PATH}")
        sys.exit(1)

    total_start = time.time()
    logger.info(f"Skill path: {SKILL_PATH}")
    builder = IndexBuilder(SKILL_PATH)

    if builder.index_exists():
        logger.info("Existing index found, checking for incremental updates...")
    else:
        logger.info("No existing index found, starting fresh build...")

    success = builder.build_or_update()
    total_elapsed = time.time() - total_start
    logger.info(f"[DONE] Total elapsed time: {total_elapsed:.1f}s ({total_elapsed/60:.1f} min)")

    if success:
        logger.info("Index build/update completed successfully.")
        retriever = builder.load_retriever()
        if retriever and retriever.is_ready():
            logger.info("Retriever loaded and ready.")
            # Quick sanity search
            try:
                results = retriever.search("load a volume and display it", top_k=5)
                logger.info(f"Sanity search returned {len(results)} results.")
                for i, r in enumerate(results, 1):
                    logger.info(f"  [{i}] {r.chunk.file_path} ({r.chunk.start_line}-{r.chunk.end_line}) score={r.final_score:.3f}")
            except Exception as e:
                logger.warning(f"Sanity search failed: {e}")
    else:
        logger.error("Index build/update failed.")
        sys.exit(1)
