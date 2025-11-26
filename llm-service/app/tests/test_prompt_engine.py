import pytest
from pathlib import Path
import tempfile
import shutil
import prompt_engine


class TestPromptEngine:
    @pytest.fixture
    def temp_prompts_dir(self):
        """Creates a temporary directory for storing prompt data"""
        temp_dir = tempfile.mkdtemp()
        prompts_dir = Path(temp_dir)

        definition_template = """<prompt>
    <role>
        Ты — строгий академический ассистент.
        Твоя задача — извлекать точные формулировки из предоставленных источников.
    </role>

    <context>
        {context}
    </context>

    <question>
        {question}
    </question>

    <rules>
        - Отвечай ТОЛЬКО информацией из контекста
        - Цитируй дословно, если возможно
        - Если информации нет — скажи «Не найдено в материалах»
        - Не интерпретируй и не добавляй примеры
        - Укажи источник при его наличии
    </rules>

    <response_template>
        <definition>
            [Дословная цитата или точная формулировка]
        </definition>
        <source>
            [Источник: учебник, страница]
        </source>
    </response_template>
</prompt>"""

        explanation_template = """<prompt>
    <role>
        Ты — опытный преподаватель, объясняющий сложные темы простым языком.
    </role>

    <context>
        {context}
    </context>

    <question>
        {question}
    </question>

    <teaching_approach>
        - Объясни тему на основе контекста, но своими словами
        - Используй аналогии для сложных концепций
        - Разбей объяснение на логические шаги
        - Подчеркни практическое применение
        - Будь краток, но содержателен (150-300 слов)
    </teaching_approach>

    <example_explanation>
        {examples_placeholder}
    </example_explanation>

    <response_structure>
        <core_idea>
            [1-2 предложения - суть темы]
        </core_idea>

        <detailed_explanation>
            [3-5 пунктов с четкой структурой]
        </detailed_explanation>

        <why_it_matters>
            [Практическая значимость и применение]
        </why_it_matters>
    </response_structure>
</prompt>"""

        (prompts_dir / 'definition.txt').write_text(definition_template,
                                                    encoding='utf-8')
        (prompts_dir / 'explanation.txt').write_text(explanation_template,
                                                     encoding='utf-8')

        yield prompts_dir
        shutil.rmtree(temp_dir)

    def test_init_loads_correct_templates(self, temp_prompts_dir):
        """Test downloading templates from prompt engine"""
        engine = prompt_engine.PromptEngine(temp_prompts_dir)

        assert 'definition' in engine.templates
        assert 'explanation' in engine.templates
        assert len(engine.templates) == 2

    def test_build_definition_prompt(self, temp_prompts_dir):
        """Test building definition prompt"""
        engine = prompt_engine.PromptEngine(temp_prompts_dir)

        context = ('Граф — это совокупность непустого множества вершин и '
                   'наборов упорядоченных или неупорядоченных пар вершин.')
        question = 'Что такое граф?'

        result = engine.build_prompt(
            'definition',
            context=context,
            question=question
        )

        assert context in result
        assert question in result
        assert 'строгий академический ассистент' in result
        assert 'Отвечай ТОЛЬКО информацией из контекста' in result
        assert '<definition>' in result

    def test_build_explanation_prompt(self, temp_prompts_dir):
        """Test building explanation prompt"""
        engine = prompt_engine.PromptEngine(temp_prompts_dir)

        context = 'Дерево — это связный граф без циклов.'
        question = 'Объясни, что такое дерево в теории графов'

        result = engine.build_prompt(
            'explanation',
            context=context,
            question=question,
        )

        assert context in result
        assert question in result
        assert 'опытный преподаватель' in result
        assert 'Объясни тему на основе контекста, но своими словами' in result
        assert '<core_idea>' in result

    def test_template_with_unknown_placeholder(self, temp_prompts_dir, caplog):
        """Test with unknown placeholder"""
        engine = prompt_engine.PromptEngine(temp_prompts_dir)

        result = engine.build_prompt(
            'definition',
            context='Контекст',
            question='Вопрос',
            unknown_placeholder='This placeholder is unknown'
        )

        assert 'unknown_placeholder' in caplog.text
        assert 'This placeholder is unknown' not in result

    def test_empty_prompts_directory(self, caplog):
        empty_dir = tempfile.mkdtemp()
        try:
            with pytest.raises(Exception) as exc_info:
                _ = prompt_engine.PromptEngine(Path(empty_dir))
            assert 'template loading failed' in str(exc_info.value).lower()
        finally:
            shutil.rmtree(empty_dir)
