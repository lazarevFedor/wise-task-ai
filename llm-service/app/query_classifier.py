import re
from typing import Literal
from logger import get_logger


class QueryClassifier:
    """Classifier for query type detection"""

    def __init__(self):
        """
        Initialize the QueryClassifier.

        Sets up the logger for the classifier.
        """
        self.logger = get_logger(__name__)

    def classify(self, question: str) -> Literal["definition", "explanation"]:
        """
        Detects query type based on predefined patterns in the question.

        Analyzes the input question using regular expression patterns to determine
        if it is seeking a 'definition' or an 'explanation'. Defaults to 'explanation'
        if no patterns match.

        Args:
            question (str): The input question to classify.

        Returns:
            Literal["definition", "explanation"]: The detected query type.
        """
        question_lower = question.lower().strip()
        self.logger.debug(f'Classifying question: "{question}"')

        definition_patterns = [
            r'что такое\s+',
            r'определени[ея]\s+',
            r'определи\s+',
            r'что значит\s+',
            r'что означает\s+',
            r'дай определение',
            r'как определяется',
            r'что есть\s+',
            r'расшифруй\s+',
            r'формулировка',
            r'точное определение',
            r'дайте определение',
            r'что называется'
        ]

        explanation_patterns = [
            r'объясни\s+',
            r'расскажи\s+',
            r'как работает\s+',
            r'в чём смысл\s+',
            r'зачем нужно\s+',
            r'почему\s+',
            r'какой принцип\s+',
            r'разъясни\s+',
            r'покажи на примере',
            r'как понять\s+',
            r'как устроен',
            r'в чем разница',
            r'чем отличается',
            r'как применять',
            r'для чего используется'
        ]

        for pattern in definition_patterns:
            if re.search(pattern, question_lower):
                self.logger.info(f'Question classified as DEFINITION: "{question}"')
                return 'definition'

        for pattern in explanation_patterns:
            if re.search(pattern, question_lower):
                self.logger.info(f'Question classified as EXPLANATION: "{question}"')
                return 'explanation'

        self.logger.info(f'Question classified as EXPLANATION (default): "{question}"')
        return 'explanation'
