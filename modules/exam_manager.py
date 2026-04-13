"""
Exam Manager module for handling exam questions, answers, and scoring.
"""

from typing import List, Dict, Any, Optional
from modules.storage import Storage


class ExamManager:
    """Manages exam questions, answers, and scoring."""
    
    def __init__(self, storage: Storage):
        """
        Initialize ExamManager with storage instance.
        
        Args:
            storage: Storage instance for database operations
        """
        self.storage = storage
    
    def get_all_questions(self) -> List[Dict[str, Any]]:
        """
        Get all exam questions.
        
        Returns:
            List of question dictionaries
        """
        return self.storage.get_all_questions()
    
    def validate_answer(self, question_id: int, selected_answer: str) -> bool:
        """
        Validate if a selected answer is correct.
        
        Args:
            question_id: Question ID
            selected_answer: Selected answer ('A', 'B', 'C', or 'D')
            
        Returns:
            True if correct, False otherwise
        """
        question = self.storage.get_question(question_id)
        if not question:
            return False
        
        correct_answer = question['correct_answer'].upper().strip()
        selected = selected_answer.upper().strip()
        
        return correct_answer == selected
    
    def calculate_score(self, session_id: int) -> Dict[str, Any]:
        """
        Calculate exam score for a session.
        
        Args:
            session_id: Session ID
            
        Returns:
            Dictionary with score, total_questions, and correct_count
        """
        answers = self.storage.get_session_answers(session_id)
        
        if not answers:
            return {
                'score': 0,
                'total_questions': 0,
                'correct_count': 0,
                'incorrect_count': 0,
                'percentage': 0.0
            }
        
        correct_count = sum(1 for answer in answers if answer['is_correct'])
        total_questions = len(answers)
        incorrect_count = total_questions - correct_count
        percentage = (correct_count / total_questions * 100) if total_questions > 0 else 0.0
        
        return {
            'score': correct_count,
            'total_questions': total_questions,
            'correct_count': correct_count,
            'incorrect_count': incorrect_count,
            'percentage': percentage
        }
    
    def save_answers(self, session_id: int, answers: Dict[int, str]) -> int:
        """
        Save multiple answers for a session.
        
        Args:
            session_id: Session ID
            answers: Dictionary mapping question_id to selected_answer
            
        Returns:
            Number of answers saved
        """
        saved_count = 0
        for question_id, selected_answer in answers.items():
            is_correct = self.validate_answer(question_id, selected_answer)
            self.storage.save_answer(session_id, question_id, selected_answer, is_correct)
            saved_count += 1
        
        return saved_count
    
    def get_exam_results(self, session_id: int) -> Dict[str, Any]:
        """
        Get complete exam results including questions, answers, and score.
        
        Args:
            session_id: Session ID
            
        Returns:
            Dictionary with exam results
        """
        session = self.storage.get_session(session_id)
        if not session:
            return {}
        
        answers = self.storage.get_session_answers(session_id)
        score_info = self.calculate_score(session_id)
        
        return {
            'session': session,
            'answers': answers,
            'score': score_info
        }

