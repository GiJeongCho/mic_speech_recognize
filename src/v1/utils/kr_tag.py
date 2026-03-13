import logging
from kiwipiepy import Kiwi
from typing import List, Any, Optional

logger = logging.getLogger(__name__)


class KiwiTagger:
    _instance = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(KiwiTagger, cls).__new__(cls)
            try:
                cls._instance.kiwi = Kiwi(
                    num_workers=0,
                    integrate_allomorph=True,
                    model_type='cong',
                    typo_cost_threshold=2.5,
                )
                logger.info("Kiwi initialized successfully with model_type='cong'.")
            except Exception as e:
                logger.error(f"Failed to initialize Kiwi: {e}")
                cls._instance.kiwi = None
        return cls._instance

    def split_into_sents(self, text: str) -> List[Any]:
        if not text.strip() or self.kiwi is None:
            return []
        try:
            return self.kiwi.split_into_sents(text)
        except Exception as e:
            logger.error(f"Kiwi split_into_sents failed: {e}")
            return []

    def get_ending_type(self, text: str) -> Optional[str]:
        if not text.strip() or self.kiwi is None:
            return None
        try:
            analysis = self.kiwi.analyze(text)
            if analysis:
                tokens = analysis[0][0]
                if any(t.tag == 'EF' for t in tokens):
                    return 'EF'
                if any(t.tag == 'EC' for t in tokens):
                    return 'EC'
        except Exception as e:
            logger.error(f"Kiwi analysis failed for text '{text}': {e}")
            if any(p in text for p in [".", "!", "?"]):
                return 'EF'
        return None


kiwi_tagger = KiwiTagger()
