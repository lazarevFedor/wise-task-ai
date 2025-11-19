import pytest
import query_classifier


@pytest.fixture
def classifier():
    """Fixture to create a QueryClassifier instance."""
    return query_classifier.QueryClassifier()


class TestQueryClassifier:
    """Test suite for QueryClassifier."""

    @pytest.mark.parametrize(
        'question, expected',
        [
            ('Что такое ___', 'definition'),
            ('определение ___', 'definition'),
            ('Определи ___', 'definition'),
            ('Что значит ___', 'definition'),
            ('Что означает ___', 'definition'),
            ('Дай определение ___', 'definition'),
            ('Как определяется ___', 'definition'),
            ('Что есть ___', 'definition'),
            ('Расшифруй ___', 'definition'),
            ('Формулировка ___', 'definition'),
            ('Точное определение ___', 'definition'),
            ('Дайте определение ___', 'definition'),
            ('Что называется ___', 'definition'),
            ('Что подразумевается под ___', 'definition'),
            ('Суть что такое ___', 'definition'),
            ('Поясни термин ___', 'definition'),
            ('Что это за понятие ___', 'definition'),
            ('ЧТО ТАКОЕ ___', 'definition'),
        ],
    )
    def test_classify_definition(self, classifier, question, expected):
        """Test classification for definition patterns."""
        result = classifier.classify(question)
        assert result == expected

    @pytest.mark.parametrize(
        'question, expected',
        [
            ('Объясни, как работает ___', 'explanation'),
            ('Расскажи про ___', 'explanation'),
            ('Как работает ___', 'explanation'),
            ('В чём смысл ___', 'explanation'),
            ('Зачем нужно ___', 'explanation'),
            ('Почему используется ___', 'explanation'),
            ('Какой принцип ___', 'explanation'),
            ('Разъясни ___', 'explanation'),
            ('Покажи на примере ___', 'explanation'),
            ('Как понять ___', 'explanation'),
            ('Как устроен ___', 'explanation'),
            ('В чем разница между ___ и ___', 'explanation'),
            ('Чем отличается ___ от ___', 'explanation'),
            ('Как применять ___', 'explanation'),
            ('Для чего используется ___', 'explanation'),
            ('Приведи пример использования ___', 'explanation'),
            ('Механизм работы ___', 'explanation'),
            ('Почему именно так ___', 'explanation'),
            ('Как это связано с ___', 'explanation'),
            ('Простыми словами ___', 'explanation'),
            ('ОБЪЯСНИ ___', 'explanation'),
        ],
    )
    def test_classify_explanation(self, classifier, question, expected):
        """Test classification for explanation patterns (no definition match)."""
        result = classifier.classify(question)
        assert result == expected

    @pytest.mark.parametrize(
        'question, expected',
        [
            ('Что такое и как работает ___', 'definition'),
            ('Определи и объясни ___', 'definition'),
            ('Что значит ___, расскажи подробнее', 'definition'),
            ('Сколько стоит подписка на Spotify?', 'explanation'),
            ('Привет, как дела?', 'explanation'),
            ('Просто текст без вопроса.', 'explanation'),
            ('', 'explanation'),
            ('   ', 'explanation'),
            ('123', 'explanation'),
            ('fobsjbfoisrhbiu', 'explanation'),
        ],
    )
    def test_classify_default_or_priority(self, classifier, question, expected):
        """Test default classification and priority (definition over explanation)."""
        result = classifier.classify(question)
        assert result == expected
