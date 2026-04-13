"""
Database storage module for managing users, sessions, and events.
Uses SQLite for local data storage.
"""

import sqlite3
import os
from datetime import datetime
from typing import Optional, List, Dict, Any
import json


class Storage:
    """SQLite database wrapper for proctoring system."""
    
    def __init__(self, db_path: str = "proctoring.db"):
        """
        Initialize database connection and create tables if they don't exist.
        
        Args:
            db_path: Path to SQLite database file
        """
        self.db_path = db_path
        self.conn = sqlite3.connect(db_path, check_same_thread=False)
        self.conn.row_factory = sqlite3.Row  # Enable column access by name
        self._init_schema()
    
    def _init_schema(self):
        """Create database tables if they don't exist."""
        cursor = self.conn.cursor()
        
        # Users table
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS users (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                name TEXT NOT NULL UNIQUE,
                enrollment_image_path TEXT NOT NULL,
                embedding_path TEXT,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        """)
        
        # Sessions table
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS sessions (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                candidate_name TEXT NOT NULL,
                exam_title TEXT NOT NULL,
                started_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                ended_at TIMESTAMP,
                reviewed BOOLEAN DEFAULT 0,
                score INTEGER,
                total_questions INTEGER,
                completion_status TEXT
            )
        """)
        
        # Add new columns to existing sessions table if they don't exist
        try:
            cursor.execute("ALTER TABLE sessions ADD COLUMN score INTEGER")
        except sqlite3.OperationalError:
            pass  # Column already exists
        try:
            cursor.execute("ALTER TABLE sessions ADD COLUMN total_questions INTEGER")
        except sqlite3.OperationalError:
            pass
        try:
            cursor.execute("ALTER TABLE sessions ADD COLUMN completion_status TEXT")
        except sqlite3.OperationalError:
            pass
        
        # Events table
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS events (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                session_id INTEGER NOT NULL,
                event_type TEXT NOT NULL,
                timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                details TEXT,
                evidence_image_path TEXT,
                FOREIGN KEY (session_id) REFERENCES sessions(id)
            )
        """)
        
        # Exam questions table
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS exam_questions (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                question_text TEXT NOT NULL,
                option_a TEXT NOT NULL,
                option_b TEXT NOT NULL,
                option_c TEXT NOT NULL,
                option_d TEXT NOT NULL,
                correct_answer TEXT NOT NULL,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        """)
        
        # Exam answers table
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS exam_answers (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                session_id INTEGER NOT NULL,
                question_id INTEGER NOT NULL,
                selected_answer TEXT NOT NULL,
                is_correct BOOLEAN NOT NULL,
                answered_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                FOREIGN KEY (session_id) REFERENCES sessions(id),
                FOREIGN KEY (question_id) REFERENCES exam_questions(id)
            )
        """)
        
        self.conn.commit()
        
        # Initialize default questions if table is empty
        self._init_default_questions()
    
    def _init_default_questions(self):
        """Initialize default programming MCQ questions if none exist."""
        cursor = self.conn.cursor()
        cursor.execute("SELECT COUNT(*) as count FROM exam_questions")
        result = cursor.fetchone()
        
        if result['count'] == 0:
            # Default programming questions
            default_questions = [
                {
                    'question_text': 'What is the output of: print(2 ** 3) in Python?',
                    'option_a': '6',
                    'option_b': '8',
                    'option_c': '9',
                    'option_d': '5',
                    'correct_answer': 'B'
                },
                {
                    'question_text': 'Which keyword is used to define a function in Python?',
                    'option_a': 'func',
                    'option_b': 'def',
                    'option_c': 'function',
                    'option_d': 'define',
                    'correct_answer': 'B'
                },
                {
                    'question_text': 'What does the len() function return for the string "Hello"?',
                    'option_a': '4',
                    'option_b': '5',
                    'option_c': '6',
                    'option_d': 'Error',
                    'correct_answer': 'B'
                },
                {
                    'question_text': 'Which data type is mutable in Python?',
                    'option_a': 'String',
                    'option_b': 'Tuple',
                    'option_c': 'List',
                    'option_d': 'Integer',
                    'correct_answer': 'C'
                },
                {
                    'question_text': 'What is the result of: [1, 2, 3] + [4, 5] in Python?',
                    'option_a': '[5, 7, 3]',
                    'option_b': '[1, 2, 3, 4, 5]',
                    'option_c': '[1, 2, 7, 5]',
                    'option_d': 'Error',
                    'correct_answer': 'B'
                }
            ]
            
            for q in default_questions:
                cursor.execute("""
                    INSERT INTO exam_questions (question_text, option_a, option_b, option_c, option_d, correct_answer)
                    VALUES (?, ?, ?, ?, ?, ?)
                """, (q['question_text'], q['option_a'], q['option_b'], q['option_c'], q['option_d'], q['correct_answer']))
            
            self.conn.commit()
    
    # User operations
    def create_user(self, name: str, enrollment_image_path: str, embedding_path: Optional[str] = None) -> int:
        """
        Create a new enrolled user.
        
        Args:
            name: User's name
            enrollment_image_path: Path to enrollment image
            embedding_path: Optional path to embedding file
            
        Returns:
            User ID
        """
        cursor = self.conn.cursor()
        try:
            cursor.execute("""
                INSERT INTO users (name, enrollment_image_path, embedding_path)
                VALUES (?, ?, ?)
            """, (name, enrollment_image_path, embedding_path))
            self.conn.commit()
            return cursor.lastrowid
        except sqlite3.IntegrityError:
            # User already exists, update instead
            cursor.execute("""
                UPDATE users 
                SET enrollment_image_path = ?, embedding_path = ?
                WHERE name = ?
            """, (enrollment_image_path, embedding_path, name))
            self.conn.commit()
            cursor.execute("SELECT id FROM users WHERE name = ?", (name,))
            return cursor.fetchone()['id']
    
    def get_user(self, name: str) -> Optional[Dict[str, Any]]:
        """
        Get user by name.
        
        Args:
            name: User's name
            
        Returns:
            User dictionary or None if not found
        """
        cursor = self.conn.cursor()
        cursor.execute("SELECT * FROM users WHERE name = ?", (name,))
        row = cursor.fetchone()
        if row:
            return dict(row)
        return None
    
    def get_all_users(self) -> List[Dict[str, Any]]:
        """Get all enrolled users."""
        cursor = self.conn.cursor()
        cursor.execute("SELECT * FROM users ORDER BY created_at DESC")
        return [dict(row) for row in cursor.fetchall()]
    
    # Session operations
    def create_session(self, candidate_name: str, exam_title: str, 
                      score: Optional[int] = None, total_questions: Optional[int] = None,
                      completion_status: Optional[str] = None) -> int:
        """
        Create a new exam session.
        
        Args:
            candidate_name: Name of the candidate
            exam_title: Title of the exam
            score: Optional exam score
            total_questions: Optional total number of questions
            completion_status: Optional completion status ('completed', 'timeout', 'abandoned')
            
        Returns:
            Session ID
        """
        cursor = self.conn.cursor()
        cursor.execute("""
            INSERT INTO sessions (candidate_name, exam_title, score, total_questions, completion_status)
            VALUES (?, ?, ?, ?, ?)
        """, (candidate_name, exam_title, score, total_questions, completion_status))
        self.conn.commit()
        return cursor.lastrowid
    
    def update_session_score(self, session_id: int, score: int, total_questions: int, 
                            completion_status: str):
        """
        Update session with exam score and completion status.
        
        Args:
            session_id: Session ID
            score: Exam score
            total_questions: Total number of questions
            completion_status: Completion status ('completed', 'timeout', 'abandoned')
        """
        cursor = self.conn.cursor()
        cursor.execute("""
            UPDATE sessions 
            SET score = ?, total_questions = ?, completion_status = ?
            WHERE id = ?
        """, (score, total_questions, completion_status, session_id))
        self.conn.commit()
    
    def end_session(self, session_id: int):
        """Mark session as ended with current timestamp."""
        cursor = self.conn.cursor()
        cursor.execute("""
            UPDATE sessions 
            SET ended_at = CURRENT_TIMESTAMP
            WHERE id = ?
        """, (session_id,))
        self.conn.commit()
    
    def get_session(self, session_id: int) -> Optional[Dict[str, Any]]:
        """Get session by ID."""
        cursor = self.conn.cursor()
        cursor.execute("SELECT * FROM sessions WHERE id = ?", (session_id,))
        row = cursor.fetchone()
        if row:
            return dict(row)
        return None
    
    def get_all_sessions(self) -> List[Dict[str, Any]]:
        """Get all sessions ordered by start time (newest first)."""
        cursor = self.conn.cursor()
        cursor.execute("""
            SELECT * FROM sessions 
            ORDER BY started_at DESC
        """)
        return [dict(row) for row in cursor.fetchall()]
    
    def mark_session_reviewed(self, session_id: int):
        """Mark a session as reviewed."""
        cursor = self.conn.cursor()
        cursor.execute("""
            UPDATE sessions 
            SET reviewed = 1
            WHERE id = ?
        """, (session_id,))
        self.conn.commit()
    
    # Event operations
    def log_event(self, session_id: int, event_type: str, details: Optional[str] = None, 
                  evidence_image_path: Optional[str] = None) -> int:
        """
        Log an anomaly event.
        
        Args:
            session_id: Session ID
            event_type: Type of event (face_missing, multiple_faces, etc.)
            details: Optional JSON string with additional details
            evidence_image_path: Optional path to evidence image
            
        Returns:
            Event ID
        """
        cursor = self.conn.cursor()
        # Convert details dict to JSON string if it's a dict
        if isinstance(details, dict):
            details = json.dumps(details)
        
        cursor.execute("""
            INSERT INTO events (session_id, event_type, details, evidence_image_path)
            VALUES (?, ?, ?, ?)
        """, (session_id, event_type, details, evidence_image_path))
        self.conn.commit()
        return cursor.lastrowid
    
    def get_session_events(self, session_id: int) -> List[Dict[str, Any]]:
        """
        Get all events for a session.
        
        Args:
            session_id: Session ID
            
        Returns:
            List of event dictionaries ordered by timestamp
        """
        cursor = self.conn.cursor()
        cursor.execute("""
            SELECT * FROM events 
            WHERE session_id = ?
            ORDER BY timestamp ASC
        """, (session_id,))
        return [dict(row) for row in cursor.fetchall()]
    
    def get_events_by_type(self, event_type: str, session_id: Optional[int] = None) -> List[Dict[str, Any]]:
        """
        Get events by type, optionally filtered by session.
        
        Args:
            event_type: Type of event to filter
            session_id: Optional session ID to filter by
            
        Returns:
            List of event dictionaries
        """
        cursor = self.conn.cursor()
        if session_id:
            cursor.execute("""
                SELECT * FROM events 
                WHERE event_type = ? AND session_id = ?
                ORDER BY timestamp ASC
            """, (event_type, session_id))
        else:
            cursor.execute("""
                SELECT * FROM events 
                WHERE event_type = ?
                ORDER BY timestamp ASC
            """, (event_type,))
        return [dict(row) for row in cursor.fetchall()]
    
    # Exam question operations
    def create_question(self, question_text: str, option_a: str, option_b: str, 
                       option_c: str, option_d: str, correct_answer: str) -> int:
        """
        Create a new exam question.
        
        Args:
            question_text: The question text
            option_a: Option A text
            option_b: Option B text
            option_c: Option C text
            option_d: Option D text
            correct_answer: Correct answer ('A', 'B', 'C', or 'D')
            
        Returns:
            Question ID
        """
        cursor = self.conn.cursor()
        cursor.execute("""
            INSERT INTO exam_questions (question_text, option_a, option_b, option_c, option_d, correct_answer)
            VALUES (?, ?, ?, ?, ?, ?)
        """, (question_text, option_a, option_b, option_c, option_d, correct_answer))
        self.conn.commit()
        return cursor.lastrowid
    
    def get_all_questions(self) -> List[Dict[str, Any]]:
        """Get all exam questions."""
        cursor = self.conn.cursor()
        cursor.execute("SELECT * FROM exam_questions ORDER BY id ASC")
        return [dict(row) for row in cursor.fetchall()]
    
    def get_question(self, question_id: int) -> Optional[Dict[str, Any]]:
        """
        Get a question by ID.
        
        Args:
            question_id: Question ID
            
        Returns:
            Question dictionary or None if not found
        """
        cursor = self.conn.cursor()
        cursor.execute("SELECT * FROM exam_questions WHERE id = ?", (question_id,))
        row = cursor.fetchone()
        if row:
            return dict(row)
        return None
    
    def question_exists(self, question_text: str) -> bool:
        """
        Check if a question with the same text already exists.
        
        Args:
            question_text: Question text to check
            
        Returns:
            True if question exists, False otherwise
        """
        cursor = self.conn.cursor()
        cursor.execute("SELECT COUNT(*) as count FROM exam_questions WHERE question_text = ?", (question_text,))
        result = cursor.fetchone()
        return result['count'] > 0
    
    # Exam answer operations
    def save_answer(self, session_id: int, question_id: int, selected_answer: str, 
                   is_correct: bool) -> int:
        """
        Save a student's answer for a question.
        
        Args:
            session_id: Session ID
            question_id: Question ID
            selected_answer: Selected answer ('A', 'B', 'C', or 'D')
            is_correct: Whether the answer is correct
            
        Returns:
            Answer ID
        """
        cursor = self.conn.cursor()
        cursor.execute("""
            INSERT INTO exam_answers (session_id, question_id, selected_answer, is_correct)
            VALUES (?, ?, ?, ?)
        """, (session_id, question_id, selected_answer, is_correct))
        self.conn.commit()
        return cursor.lastrowid
    
    def get_session_answers(self, session_id: int) -> List[Dict[str, Any]]:
        """
        Get all answers for a session.
        
        Args:
            session_id: Session ID
            
        Returns:
            List of answer dictionaries with question details
        """
        cursor = self.conn.cursor()
        cursor.execute("""
            SELECT 
                ea.id,
                ea.session_id,
                ea.question_id,
                ea.selected_answer,
                ea.is_correct,
                ea.answered_at,
                eq.question_text,
                eq.option_a,
                eq.option_b,
                eq.option_c,
                eq.option_d,
                eq.correct_answer
            FROM exam_answers ea
            JOIN exam_questions eq ON ea.question_id = eq.id
            WHERE ea.session_id = ?
            ORDER BY ea.question_id ASC
        """, (session_id,))
        return [dict(row) for row in cursor.fetchall()]
    
    def close(self):
        """Close database connection."""
        self.conn.close()
    
    def __enter__(self):
        """Context manager entry."""
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit."""
        self.close()
