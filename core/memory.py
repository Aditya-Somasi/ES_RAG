"""
SQLite-backed session store for chat history.
Persistent across restarts with TTL-based session expiration.
"""

import os
import time
from datetime import datetime, timedelta
from typing import Optional
from langchain_community.chat_message_histories import SQLChatMessageHistory
from utils.logging import setup_logger


logger = setup_logger(__name__)

# Default database path
DEFAULT_DB_PATH = os.path.join(
    os.path.dirname(os.path.dirname(__file__)), 
    "data", 
    "chat_history.db"
)

# Session TTL in hours (sessions expire after this time of inactivity)
SESSION_TTL_HOURS = 24


class SQLiteSessionStore:
    """
    SQLite-backed session store for chat histories.
    
    Features:
    - Persistent storage across restarts
    - TTL-based session expiration
    - Thread-safe (SQLite handles locking)
    """
    
    def __init__(self, db_path: str = DEFAULT_DB_PATH, ttl_hours: int = SESSION_TTL_HOURS):
        """
        Initialize SQLite session store.
        
        Args:
            db_path: Path to SQLite database file
            ttl_hours: Session TTL in hours (default 24)
        """
        self.db_path = db_path
        self.ttl_hours = ttl_hours
        self.connection_string = f"sqlite:///{db_path}"
        
        # Track session access times for TTL
        self._session_access_times: dict[str, datetime] = {}
        
        # Ensure data directory exists
        os.makedirs(os.path.dirname(db_path), exist_ok=True)
        
        logger.info(f"SQLite session store initialized: {db_path}")
        logger.info(f"Session TTL: {ttl_hours} hours")
    
    def get_session_history(self, session_id: str) -> SQLChatMessageHistory:
        """
        Get or create chat history for a session.
        
        Args:
            session_id: Session identifier
            
        Returns:
            SQLChatMessageHistory instance for the session
        """
        # Update access time
        self._session_access_times[session_id] = datetime.now()
        
        # Create/get SQLite-backed history
        history = SQLChatMessageHistory(
            session_id=session_id,
            connection_string=self.connection_string,
        )
        
        logger.debug(f"Session accessed: {session_id}")
        return history
    
    def clear_session(self, session_id: str) -> None:
        """
        Clear chat history for a session.
        
        Args:
            session_id: Session identifier
        """
        try:
            history = SQLChatMessageHistory(
                session_id=session_id,
                connection_string=self.connection_string,
            )
            history.clear()
            
            # Remove from access tracking
            self._session_access_times.pop(session_id, None)
            
            logger.info(f"Cleared session: {session_id}")
        except Exception as e:
            logger.error(f"Failed to clear session {session_id}: {e}")
    
    def delete_session(self, session_id: str) -> None:
        """
        Delete a session entirely.
        
        Args:
            session_id: Session identifier
        """
        # SQLChatMessageHistory doesn't have a delete method,
        # so we clear it instead
        self.clear_session(session_id)
    
    def get_all_sessions(self) -> list[str]:
        """
        Get list of all session IDs.
        
        Note: This queries the database for distinct session IDs.
        
        Returns:
            List of session IDs
        """
        import sqlite3
        
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            # Check if table exists
            cursor.execute("""
                SELECT name FROM sqlite_master 
                WHERE type='table' AND name='message_store'
            """)
            
            if not cursor.fetchone():
                conn.close()
                return []
            
            cursor.execute("SELECT DISTINCT session_id FROM message_store")
            sessions = [row[0] for row in cursor.fetchall()]
            conn.close()
            
            return sessions
        except Exception as e:
            logger.error(f"Failed to get sessions: {e}")
            return []
    
    def session_exists(self, session_id: str) -> bool:
        """
        Check if a session exists.
        
        Args:
            session_id: Session identifier
            
        Returns:
            True if session exists
        """
        return session_id in self.get_all_sessions()
    
    def cleanup_expired_sessions(self) -> int:
        """
        Remove sessions that have exceeded TTL.
        
        Returns:
            Number of sessions cleaned up
        """
        import sqlite3
        
        cutoff = datetime.now() - timedelta(hours=self.ttl_hours)
        expired_count = 0
        
        # Get sessions to check
        expired_sessions = [
            session_id
            for session_id, access_time in self._session_access_times.items()
            if access_time < cutoff
        ]
        
        for session_id in expired_sessions:
            try:
                self.clear_session(session_id)
                expired_count += 1
            except Exception as e:
                logger.warning(f"Failed to cleanup session {session_id}: {e}")
        
        if expired_count > 0:
            logger.info(f"Cleaned up {expired_count} expired sessions")
        
        return expired_count
    
    def get_session_message_count(self, session_id: str) -> int:
        """
        Get number of messages in a session.
        
        Args:
            session_id: Session identifier
            
        Returns:
            Number of messages
        """
        import sqlite3
        
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            cursor.execute(
                "SELECT COUNT(*) FROM message_store WHERE session_id = ?",
                (session_id,)
            )
            count = cursor.fetchone()[0]
            conn.close()
            return count
        except Exception:
            return 0


# Global session store instance (lazy initialization)
_session_store: Optional[SQLiteSessionStore] = None


def get_session_store() -> SQLiteSessionStore:
    """Get or create the global session store instance."""
    global _session_store
    if _session_store is None:
        _session_store = SQLiteSessionStore()
    return _session_store


# For backwards compatibility
session_store = property(lambda self: get_session_store())


# Module-level function for LangChain compatibility
def get_session_history(session_id: str) -> SQLChatMessageHistory:
    """
    Get session history for use with RunnableWithMessageHistory.
    
    Args:
        session_id: Session identifier
        
    Returns:
        Chat message history instance
    """
    return get_session_store().get_session_history(session_id)