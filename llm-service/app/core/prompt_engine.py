from .logger import get_logger


class PromptEngine:
    """
    Prompt Engine generates prompts with user questions and relevant contexts
    from Knowledge Data Base for Ollama LLM.
    """

    def __init__(self, prompts_dir):
        """Initialize PromptEngine:
        initialize templates dict and load prompt templates"""
        self.logger = get_logger(__name__)
        self.prompts_dir = prompts_dir  # later catch None val
        self.templates = {}

        self._load_templates()
        self.logger.info('PromptEngine: prompt engine initialized')

    def _load_templates(self):
        """Load prompt templates
        and add them into dictionary."""
        try:
            template_files = list(self.prompts_dir.rglob('*.xml'))
            if not template_files:
                self.logger.critical('PromptEngine: no template files found')
                raise Exception('No template files found in directory')

            for template_file in self.prompts_dir.rglob('*.xml'):
                template_name = template_file.stem
                with open(template_file, 'r', encoding='utf8') as f:
                    self.templates[template_name] = f.read().strip()
                self.logger.info(f'PromptEngine: template {template_name} loaded')
        except Exception as e:
            self.logger.critical(f'PromptEngine: template loading failed: {e}')
            raise

    def build_prompt(self, template_name, **placeholders):
        """Build prompt template with placeholders
        -- user question and relevant contexts"""
        if self.templates is None:
            self.logger.critical('PromptEngine: no templates loaded')
            raise Exception('PromptEngine: no templates loaded')

        template = self.templates[template_name]

        for key, value in placeholders.items():
            placeholder = f'{{{key}}}'
            if placeholder in template:
                template = template.replace(placeholder, str(value))
            else:
                self.logger.warning(f'PromptEngine: placeholder {placeholder} not found')

        return template
