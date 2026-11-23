import sys
from importlib import import_module
from unittest.mock import MagicMock, patch

import pytest


@pytest.fixture
def classifier_module():
    """Передать тесту наш модуль с тестовой заменой нейросети."""
    # sys.modules — хранилище импортированных модулей. Вместо настоящей
    # библиотеки Transformers помещаем туда MagicMock: её pipeline()
    # вернёт ещё один тестовый объект, не загружая библиотеку и веса.
    with patch.dict(sys.modules, {"transformers": MagicMock()}):
        # import_module обычно повторно использует модуль из sys.modules.
        # Убираем предыдущий импорт, чтобы каждый тест получил свежую модель
        # без предсказаний и истории вызовов от предыдущего теста.
        sys.modules.pop("finsight.ml.classifier", None)
        ml_package = import_module("finsight.ml")

        # При импорте Python также записывает модуль в атрибут пакета ml.
        # Сохраняем и восстанавливаем этот атрибут вместе с sys.modules.
        with patch.object(ml_package, "classifier", create=True):
            # yield передаёт модуль тесту. После теста блоки with завершатся
            # и вернут исходное состояние импортов.
            yield import_module("finsight.ml.classifier")
