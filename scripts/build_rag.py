"""
Standalone script to build the dense vector retrieval index.
Run this to create/update the FAISS vector index under Resources/Code_RAG/v1/.
Uses ONNX Runtime for embedding (no PyTorch/CUDA required).
"""
import os
import sys
import logging
import time

# Project root is one level up from this script
_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s',
    handlers=[
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)
sys.path.insert(0, _PROJECT_ROOT)

# Direct import to avoid loading Slicer-dependent modules via __init__.py
import importlib.util
_spec = importlib.util.spec_from_file_location(
    "SkillIndexer",
    os.path.join(_PROJECT_ROOT, "SlicerAIAgentLib", "SkillIndexer.py")
)
SkillIndexer = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(SkillIndexer)
IndexBuilder = SkillIndexer.IndexBuilder

SKILL_PATH = os.path.join(_PROJECT_ROOT, "Resources", "Skills", "slicer-skill-full")

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

    logger.info("[STAGE 1/4] Scanning and chunking files...")
    success = builder.build_or_update()
    total_elapsed = time.time() - total_start
    logger.info(f"[DONE] Total elapsed time: {total_elapsed:.1f}s ({total_elapsed/60:.1f} min)")

    if success:
        logger.info("[STAGE 4/4] Loading retriever and running sanity check...")
        retriever = builder.load_retriever()
        if retriever and retriever.is_ready():
            logger.info("Retriever loaded and ready.")
            try:
                results = retriever.search("load a volume and display it", top_k=5)
                logger.info(f"Sanity search returned {len(results)} results.")
                for i, r in enumerate(results, 1):
                    logger.info(f"  [{i}] {r.chunk.file_path} ({r.chunk.start_line}-{r.chunk.end_line}) score={r.final_score:.3f}")
            except Exception as e:
                logger.warning(f"Sanity search failed: {e}")
        logger.info("Index build/update completed successfully.")
    else:
        logger.error("Index build/update failed.")
        sys.exit(1)
