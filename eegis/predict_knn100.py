"""Predict No or Sí using the predeclared grouped kNN checkpoint."""
import argparse
import json
from pathlib import Path

import joblib
from threadpoolctl import threadpool_limits

from .epochs import read_window


def predict(window, artifacts):
    artifacts = Path(artifacts)
    metadata = json.loads((artifacts / 'best_model.json').read_text())
    if metadata.get('mode') != 'grouped' or metadata.get('class_ids') != [6, 8]:
        raise ValueError('Expected the grouped No/Sí study checkpoint')
    model_file = artifacts / metadata['file']
    model = joblib.load(model_file)
    with threadpool_limits(limits=4):
        label = int(model.predict(read_window(Path(window))[None])[0])
    return {'class_id': label, 'label': {6: 'No', 8: 'Sí'}[label],
            'model_file': str(model_file), 'protocol': 'grouped',
            'exploratory': metadata['exploratory'],
            'subject_independent': metadata['subject_independent']}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('window', type=Path, help='One original raw EEGIS CSV')
    parser.add_argument('--artifacts', type=Path,
                        default=Path('artifacts_paper_method/knn100'))
    args = parser.parse_args()
    print(json.dumps(predict(args.window, args.artifacts), ensure_ascii=False))


if __name__ == '__main__':
    main()
