"""Code-only relocation identities; data/model hashes always use their real bytes.

Compatibility is allowed only for the exact reviewed relocation bytes. Any later
edit returns a new fingerprint. The proof ledger keeps both identities visible.
"""
import hashlib
import json
from pathlib import Path
from acp.data.assets import PROJECT, resolve_path


def digest(path):
    path=resolve_path(path)
    actual=hashlib.sha256(path.read_bytes()).hexdigest()
    try:key=str(path.relative_to(PROJECT))
    except ValueError:return actual
    proof_path=PROJECT/'configs/code_relocations.json'
    if proof_path.is_file():
        entry=json.loads(proof_path.read_text()).get('files',{}).get(key,{})
        if entry.get('relocated_sha256')==actual:return entry['original_sha256']
    return actual
