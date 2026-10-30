import hashlib
import json
from acp.management import fingerprints


def test_code_identity_compatibility_expires_after_edit(tmp_path,monkeypatch):
    code=tmp_path/'module.py';code.write_text('value = 1\n')
    actual=hashlib.sha256(code.read_bytes()).hexdigest()
    (tmp_path/'configs').mkdir()
    (tmp_path/'configs/code_relocations.json').write_text(json.dumps({'files':{'module.py':{
        'original_sha256':'old-reviewed-identity','relocated_sha256':actual}}}))
    monkeypatch.setattr(fingerprints,'PROJECT',tmp_path)
    monkeypatch.setattr(fingerprints,'resolve_path',lambda value: value)
    assert fingerprints.digest(code)=='old-reviewed-identity'
    code.write_text('value = 2\n')
    assert fingerprints.digest(code)==hashlib.sha256(code.read_bytes()).hexdigest()
    data=tmp_path/'motion.npz';data.write_bytes(b'dataset')
    assert fingerprints.digest(data)==hashlib.sha256(b'dataset').hexdigest()
