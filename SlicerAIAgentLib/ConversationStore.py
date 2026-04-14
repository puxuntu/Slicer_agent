"""
ConversationStore - Manages conversation history and persistence.

Provides storage, retrieval, and export of conversation sessions.
"""

import json
import logging
import os
from datetime import datetime
from typing import Dict, List, Optional

import qt
import slicer

logger = logging.getLogger(__name__)


class ConversationStore:
    """
    Manages conversation history for the SlicerKimiAgent.
    
    Features:
    - In-memory conversation storage
    - Persistent storage to Slicer settings
    - Export to JSON for sharing
    - Conversation metadata (timestamps, token usage)
    """
    
    def __init__(self, max_history: int = 100):
        """
        Initialize the conversation store.
        
        Args:
            max_history: Maximum number of exchanges to keep in memory
        """
        self.max_history = max_history
        self.conversations: List[Dict] = []
        self.current_session_id = self._generateSessionId()
        self._loadFromSettings()
        
    def _generateSessionId(self) -> str:
        """Generate a unique session ID."""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        return f"session_{timestamp}"
        
    def addExchange(self, user_prompt: str, assistant_response: Dict):
        """
        Add a conversation exchange.
        
        Args:
            user_prompt: The user's input
            assistant_response: The assistant's response dictionary
        """
        exchange = {
            "timestamp": datetime.now().isoformat(),
            "session_id": self.current_session_id,
            "user": user_prompt,
            "assistant": assistant_response.get("message", ""),
            "reasoning_content": assistant_response.get("reasoning_content", ""),
            "code": assistant_response.get("code"),
            "tokens": assistant_response.get("tokens", 0),
            "cost": assistant_response.get("cost", 0.0),
        }
        
        self.conversations.append(exchange)
        
        # Trim history if exceeded
        if len(self.conversations) > self.max_history:
            self.conversations = self.conversations[-self.max_history:]
            
        # Auto-save every 5 exchanges
        if len(self.conversations) % 5 == 0:
            self._saveToSettings()
            
    def getCurrentSession(self) -> List[Dict]:
        """Get all exchanges from the current session."""
        return [c for c in self.conversations if c.get("session_id") == self.current_session_id]
        
    def getAllConversations(self) -> List[Dict]:
        """Get all stored conversations."""
        return self.conversations.copy()
        
    def getSessionIds(self) -> List[str]:
        """Get all unique session IDs."""
        return list(set(c.get("session_id") for c in self.conversations))
        
    def getSession(self, session_id: str) -> List[Dict]:
        """Get all exchanges for a specific session."""
        return [c for c in self.conversations if c.get("session_id") == session_id]
        
    def clear(self):
        """Clear current session history."""
        # Remove only current session conversations
        self.conversations = [c for c in self.conversations if c.get("session_id") != self.current_session_id]
        self._saveToSettings()
        logger.info(f"Cleared session: {self.current_session_id}")
        
    def clearAll(self):
        """Clear all conversation history."""
        self.conversations = []
        self._saveToSettings()
        logger.info("Cleared all conversations")
        
    def newSession(self):
        """Start a new conversation session."""
        self.current_session_id = self._generateSessionId()
        logger.info(f"Started new session: {self.current_session_id}")
        
    def exportSession(self, filepath: str, session_id: Optional[str] = None):
        """
        Export a session to JSON file.
        
        Args:
            filepath: Path to save the JSON file
            session_id: Session to export (default: current session)
        """
        session_id = session_id or self.current_session_id
        session_data = self.getSession(session_id)
        
        export_data = {
            "session_id": session_id,
            "exported_at": datetime.now().isoformat(),
            "exchanges": session_data,
            "total_tokens": sum(e.get("tokens", 0) for e in session_data),
            "total_cost": sum(e.get("cost", 0) for e in session_data),
        }
        
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(export_data, f, indent=2, ensure_ascii=False)
            
        logger.info(f"Exported session {session_id} to {filepath}")
        
    def importSession(self, filepath: str) -> str:
        """
        Import a session from JSON file.
        
        Args:
            filepath: Path to the JSON file
            
        Returns:
            The imported session ID
        """
        with open(filepath, 'r', encoding='utf-8') as f:
            data = json.load(f)
            
        imported_session_id = data.get("session_id", self._generateSessionId())
        
        for exchange in data.get("exchanges", []):
            exchange["session_id"] = imported_session_id
            exchange["imported"] = True
            self.conversations.append(exchange)
            
        self._saveToSettings()
        logger.info(f"Imported session {imported_session_id} from {filepath}")
        return imported_session_id
        
    def getStats(self) -> Dict:
        """Get conversation statistics."""
        current_session = self.getCurrentSession()
        return {
            "total_exchanges": len(self.conversations),
            "current_session_exchanges": len(current_session),
            "total_sessions": len(self.getSessionIds()),
            "total_tokens": sum(e.get("tokens", 0) for e in self.conversations),
            "total_cost": sum(e.get("cost", 0) for e in self.conversations),
            "current_session_tokens": sum(e.get("tokens", 0) for e in current_session),
            "current_session_cost": sum(e.get("cost", 0) for e in current_session),
        }
        
    def search(self, query: str) -> List[Dict]:
        """
        Search conversation history.
        
        Args:
            query: Search string
            
        Returns:
            List of matching exchanges
        """
        query_lower = query.lower()
        results = []
        
        for exchange in self.conversations:
            if (query_lower in exchange.get("user", "").lower() or
                query_lower in exchange.get("assistant", "").lower() or
                query_lower in (exchange.get("code") or "").lower()):
                results.append(exchange)
                
        return results
        
    def _saveToSettings(self):
        """Save conversations to Slicer settings."""
        try:
            settings = qt.QSettings()
            settings.beginGroup("SlicerKimiAgent/Conversations")
            
            # Save as JSON string (Slicer settings don't support complex types)
            conversations_json = json.dumps(self.conversations[-50:])  # Keep last 50
            settings.setValue("history", conversations_json)
            settings.setValue("current_session", self.current_session_id)
            settings.endGroup()
        except Exception as e:
            logger.warning(f"Failed to save conversations: {e}")
            
    def _loadFromSettings(self):
        """Load conversations from Slicer settings."""
        try:
            settings = qt.QSettings()
            settings.beginGroup("SlicerKimiAgent/Conversations")
            
            conversations_json = settings.value("history", "[]")
            if conversations_json:
                self.conversations = json.loads(conversations_json)
                
            self.current_session_id = settings.value("current_session", self._generateSessionId())
            settings.endGroup()
            
            logger.info(f"Loaded {len(self.conversations)} conversations from settings")
        except Exception as e:
            logger.warning(f"Failed to load conversations: {e}")
            self.conversations = []
