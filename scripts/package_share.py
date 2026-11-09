"""Package source and measured reports for teammates, excluding data and checkpoints."""
from pathlib import Path
import hashlib
import json
import zipfile

root=Path(__file__).resolve().parents[1]
files=[root/name for name in ['.gitignore','README.md','CONTRIBUTING.md','requirements.txt','requirements-lock.txt','requirements-author.txt','requirements-author-lock.txt','provenance.json'] if (root/name).is_file()]
for directory in ['eegis','tests','scripts']:
    files.extend(sorted((root/directory).glob('*.py')))
for directory in ['artifacts','artifacts_improved','artifacts_epochs','artifacts_diagnostics','artifacts_alternatives','artifacts_ensembles','artifacts_paper_binary','artifacts_author_reproduction']:
    base=root/directory
    # Include a follow-up only once its final evaluation and verification have completed.
    if directory!='artifacts' and not (base/'verification.json').exists():
        continue
    files.extend(sorted(p for p in base.glob('*') if p.suffix in {'.json','.csv','.png'}))
    if (base/'README.md').exists():
        files.append(base/'README.md')
    if directory in {'artifacts_paper_binary','artifacts_author_reproduction'}:
        for child in ['paper_window_diagnostic','within_window_grouped']:
            diagnostic = base/child
            if (diagnostic/'verification.json').exists():
                files.extend(sorted(p for p in diagnostic.glob('*') if p.suffix in {'.json','.csv','.png'}))

# Completed methodology reviews and the verified printed-equation smoke run.
for directory in ['artifacts_author_recheck', 'artifacts_adversarial_review']:
    base = root / directory
    if (base / 'verification.json').exists():
        files.extend(sorted(p for p in base.iterdir()
                            if p.is_file() and p.suffix in {'.md', '.json', '.csv', '.png'}))

paper = root / 'artifacts_paper_method'
if (paper / 'README.md').exists():
    files.append(paper / 'README.md')
for name in ['smoke', 'knn100_check', 'knn100']:
    base = paper / name
    if (base / 'verification.json').exists():
        files.extend(sorted(p for p in base.rglob('*') if p.is_file()
                            and p.suffix in {'.md', '.json', '.csv', '.png', '.py'}
                            and 'original_checkpoints' not in p.parts))
    elif name == 'knn100':
        # Only immutable plans and an explicit status snapshot while training.
        # Mutable partial histories and adaptive fitness are not final results.
        for filename in ['study_plan.json', 'publication_status.json', 'README.md',
                         'optimization_parity.json', 'live_reference_audit.json']:
            if (base / filename).is_file():
                files.append(base / filename)
literal = paper / 'literal_check'
if (literal / 'strict_literal_status.json').exists():
    files.extend(sorted(p for p in literal.iterdir()
                        if p.is_file() and p.suffix in {'.json', '.csv'}))
for directory in ['artifacts_accuracy', 'artifacts_signal_models', 'artifacts_domain_models', 'artifacts_minirocket', 'artifacts_pretrained', 'artifacts_finetune', 'artifacts_accuracy_refit', 'artifacts_signal_audit']:
    base = root / directory
    if (base / 'verification.json').exists():
        # Per-family subdirectories are resume caches. Every candidate result
        # and CNN history is consolidated into unit-root CSVs by the reporter.
        for report_root in [base, base / 'window', base / 'recording']:
            if report_root.is_dir():
                files.extend(sorted(p for p in report_root.iterdir() if p.is_file()
                                    and p.suffix in {'.md', '.json', '.csv', '.png'}))
    else:
        for filename in ['README.md', 'plan.json', 'raw_audit.json', 'vendor_provenance.json', 'source_provenance.json', 'checkpoint_provenance.json', 'encoder_loading_audit.json']:
            if (base / filename).is_file():
                files.append(base / filename)
files=sorted(set(files))
share=root/'share';share.mkdir(exist_ok=True)
archive=share/'eegis-team-project.zip'
with zipfile.ZipFile(archive,'w',zipfile.ZIP_DEFLATED) as z:
    for path in files:z.write(path,path.relative_to(root))
manifest=[{'path':str(p.relative_to(root)),'bytes':p.stat().st_size,'sha256':hashlib.sha256(p.read_bytes()).hexdigest()} for p in files]
(share/'upload_manifest.json').write_text(json.dumps(manifest,indent=2))
print(f'{archive}: {len(files)} files, {archive.stat().st_size:,} bytes; no dataset, virtual environment, or model checkpoints')
