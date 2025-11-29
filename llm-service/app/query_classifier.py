import re
from typing import Literal
from logger import get_logger
from patterns import Pattern


class QueryClassifier:
    """Classifier for query type detection"""

    def __init__(self):
        """
        Initialize the QueryClassifier.

        Sets up the logger for the classifier.
        """
        self.logger = get_logger(__name__)

    def classify(self, question: str) -> Literal['definition', 'explanation', 'wise_task']:
        """
        Detects query type based on predefined patterns in the question.

        Analyzes the input question using regular expression patterns to determine
        if it is seeking a 'definition' or an 'explanation'. Defaults to 'explanation'
        if no patterns match.

        Args:
            question (str): The input question to classify.

        Returns:
            Literal['definition', 'explanation']: The detected query type.
        """
        question_lower = question.lower().strip()
        self.logger.debug(f'Classifying question: "{question}"')

        for pattern in Pattern.wisetask_patterns:
            if re.search(pattern, question_lower):
                self.logger.debug(f'Question classified as WISE_TASK: "{question}"')
                return 'wise_task'

        for pattern in Pattern.definition_patterns:
            if re.search(pattern, question_lower):
                self.logger.debug(f'Question classified as DEFINITION: "{question}"')
                return 'definition'

        for pattern in Pattern.explanation_patterns:
            if re.search(pattern, question_lower):
                self.logger.debug(f'Question classified as EXPLANATION: "{question}"')
                return 'explanation'

        self.logger.debug(f'Question classified as EXPLANATION (default): "{question}"')
        return 'explanation'
