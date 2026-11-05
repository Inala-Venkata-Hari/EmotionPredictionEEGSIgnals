"""Package source and measured reports for teammates, excluding data and checkpoints."""
from pathlib import Path
import hashlib
import json
import zipfile

root=Path(__file__).resolve().parents[1]
files=[root/name for name in ['.gitignore','README.md','CONTRIBUTING.md','requirements.txt','requirements-lock.txt','provenance.json'] if (root/name).is_file()]
for directory in ['eegis','tests','scripts']:
    files.extend(sorted((root/directory).glob('*.py')))
for directory in ['artifacts','artifacts_improved','artifacts_epochs','artifacts_diagnostics']:
    base=root/directory
    # Include a follow-up only once its final evaluation and verification have completed.
    if directory!='artifacts' and not (base/'verification.json').exists():
        continue
    files.extend(sorted(p for p in base.glob('*') if p.suffix in {'.json','.csv','.png'}))
files=sorted(set(files))
share=root/'share';share.mkdir(exist_ok=True)
archive=share/'eegis-team-project.zip'
with zipfile.ZipFile(archive,'w',zipfile.ZIP_DEFLATED) as z:
    for path in files:z.write(path,path.relative_to(root))
manifest=[{'path':str(p.relative_to(root)),'bytes':p.stat().st_size,'sha256':hashlib.sha256(p.read_bytes()).hexdigest()} for p in files]
(share/'upload_manifest.json').write_text(json.dumps(manifest,indent=2))
print(f'{archive}: {len(files)} files, {archive.stat().st_size:,} bytes; no dataset, virtual environment, or model checkpoints')
