"""Download the official pinned LaBraM checkpoint and verify its Git blob."""
import hashlib
import json
from pathlib import Path
import re
import urllib.request


root=Path(__file__).resolve().parents[1]/'artifacts_pretrained'
provenance=json.loads((root/'source_provenance.json').read_text())
commit=provenance['source_commit']
if not re.fullmatch('[0-9a-f]{40}',commit):raise ValueError('Invalid pinned source commit')
url=f'https://raw.githubusercontent.com/935963004/LaBraM/{commit}/checkpoints/labram-base.pth'
path=root/'labram-base.pth'
if not path.exists():
    print('Downloading official LaBraM base checkpoint (96.6 MB)',flush=True)
    temporary=path.with_suffix('.download')
    with urllib.request.urlopen(url,timeout=30) as response,temporary.open('wb') as stream:
        while True:
            chunk=response.read(1024*1024)
            if not chunk:break
            stream.write(chunk)
    temporary.replace(path)
data=path.read_bytes()
blob=hashlib.sha1(b'blob '+str(len(data)).encode()+b'\0'+data).hexdigest()
if len(data)!=96612769 or blob!='221148b5af403e9178629bb61e94b2e7d89ef172':
    raise ValueError('Downloaded bytes do not match the official pinned Git blob')
(root/'checkpoint_provenance.json').write_text(json.dumps({'url':url,'bytes':len(data),'git_blob_sha':blob,
    'sha256':hashlib.sha256(data).hexdigest(),'official_git_blob_verified':True},indent=2)+'\n')
print('Official checkpoint hash verified',flush=True)
