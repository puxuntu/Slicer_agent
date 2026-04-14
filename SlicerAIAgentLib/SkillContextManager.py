"""
SkillContextManager - Manages the Slicer skill knowledge base.

Resolves the skill directory relative to the extension and detects the
setup mode (full / lightweight / web) from .setup-stamp.json.
"""

import json
import logging
import os
from typing import Dict

logger = logging.getLogger(__name__)


class SkillContextManager:
    """
    Lightweight manager for the Slicer skill knowledge base.

    Features:
    - Detects skill mode from .setup-stamp.json
    - Exposes skill path for other components (e.g., SkillTools)
    """

    # Path to the skill directory (resolved relative to this file's location)
    SKILL_PATH = os.path.normpath(os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
        'Resources', 'Skills', 'slicer-skill-full'
    ))

    def __init__(self):
        """
        Initialize the skill context manager.
        """
        self.skill_path = self.SKILL_PATH
        self._skill_mode = self._detect_skill_mode()

        if not os.path.exists(self.skill_path):
            logger.warning(f"Skill not found at {self.skill_path}")
        else:
            logger.info(f"SkillContextManager initialized with skill at: {self.skill_path}")
            logger.info(f"Skill mode: {self._skill_mode}")

    def _detect_skill_mode(self) -> str:
        """
        Detect the skill mode by reading .setup-stamp.json.

        Returns:
            "full", "lightweight", "web", or "unknown"
        """
        stamp_path = os.path.join(self.skill_path, ".setup-stamp.json")
        if os.path.exists(stamp_path):
            try:
                with open(stamp_path, 'r', encoding='utf-8') as f:
                    stamp = json.load(f)
                return stamp.get("mode", "unknown")
            except Exception as e:
                logger.warning(f"Failed to read setup stamp: {e}")
        return "unknown"

    def get_skill_mode(self) -> str:
        """Get the current skill mode."""
        return self._skill_mode

    def get_status(self) -> Dict:
        """
        Get current status of the skill context manager.

        Returns:
            Dictionary with skill status information
        """
        return {
            "skill_path": self.skill_path,
            "skill_exists": os.path.exists(self.skill_path),
            "skill_mode": self._skill_mode,
        }
