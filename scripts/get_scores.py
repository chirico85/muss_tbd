from muss.resources.paths import DATASETS_DIR
from easse.cli import evaluate_system_output


def calc_metric():
    test_set = 'custom'
    # language = 'de'
    orig_sents_path = DATASETS_DIR / 'TextComplexityDE19' / 'test.complex'
    refs_sents_paths = DATASETS_DIR / 'TextComplexityDE19' / 'test.simple'
    sys_sents_path = 'current_prediction.de'

    scores = evaluate_system_output(
        test_set,
        sys_sents_path=sys_sents_path,
        orig_sents_path=orig_sents_path,
        refs_sents_paths=[refs_sents_paths],
        metrics=['sari', 'bleu', 'fkgl', 'sari_by_operation'],
        quality_estimation=False,
    )
    print(scores)

calc_metric()
