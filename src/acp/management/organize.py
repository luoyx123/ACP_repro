"""按注册表迁移资产；保留原始记录字节与可逆移动清单。默认只预览。"""
import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import re
import sys
import tarfile

PROJECT=Path(__file__).resolve().parents[3]
sys.path.insert(0,str(PROJECT/'src'))
from acp.data.assets import registry, resolve_path, atomic_json


def rewrite_markdown(text, old_path, new_path, catalog):
    def link(match):
        label,target=match.group(1),match.group(2)
        if '://' in target or target.startswith('#'):return match.group(0)
        bare,sep,fragment=target.partition('#')
        bare=bare.strip('<>')
        absolute=Path(os.path.normpath(str(old_path.parent/bare))) if not Path(bare).is_absolute() else Path(bare)
        mapped=resolve_path(absolute,catalog=catalog)
        destination=os.path.relpath(mapped,new_path.parent)
        return f'[{label}]({destination}{sep}{fragment})'
    text=re.sub(r'\[([^\]]*)\]\(([^\n)]+)\)',link,text)
    replacements=sorted(catalog['relocations'].items(),key=lambda x:len(x[0]),reverse=True)
    # One pass prevents a shorter historical prefix matching its own replacement.
    pattern=re.compile('|'.join(re.escape(a) for a,_ in replacements))
    table=dict(replacements)
    return pattern.sub(lambda m:table[m.group(0)],text)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--apply',action='store_true')
    args=parser.parse_args();catalog=registry()
    moves=sorted(catalog['relocations'].items(),key=lambda x:len(x[0]),reverse=True)
    planned=[{'from':a,'to':b} for a,b in moves if (PROJECT/a).exists()]
    if not args.apply:
        print(json.dumps({'moves':planned},ensure_ascii=False,indent=2));return
    # Refuse to rename a working directory underneath an active writer.
    for proc in Path('/proc').iterdir():
        if not proc.name.isdigit() or int(proc.name)==os.getpid():continue
        try:argv=(proc/'cmdline').read_bytes().split(b'\0')
        except OSError:continue
        if any(x in argv for x in (b'src/acp/pipeline/datasets/mixamo.py',b'src/acp/pipeline/datasets/php.py')):
            raise SystemExit(f'先为批次 {proc.name} 保存检查点并停止，再迁移；迁移后按新目录恢复。')
    stamp=datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')
    management=PROJECT/'outputs/管理';management.mkdir(parents=True,exist_ok=True)
    backup=management/f'文档迁移前备份_{stamp}.tar.gz'
    docs=[p for p in (PROJECT/'docs').rglob('*.md')]+[PROJECT/'README.md']
    content={p:p.read_text() for p in docs}
    with tarfile.open(backup,'w:gz') as archive:
        for p in docs:archive.add(p,arcname=str(p.relative_to(PROJECT)))
    ledger={'timestamp_utc':stamp,'document_backup':str(backup.relative_to(PROJECT)),
            'artifact_policy':'目录移动；历史manifest、NPZ和模型文件字节不改写；通过read_record解析旧路径',
            'moves':[],'documents':[]}
    journal=management/f'迁移记录_{stamp}.json'
    atomic_json(journal,ledger)
    for old,new in moves:
        src,dst=PROJECT/old,PROJECT/new
        if not src.exists():continue
        if dst.exists():raise FileExistsError(f'拒绝覆盖 {dst}')
        dst.parent.mkdir(parents=True,exist_ok=True)
        src.rename(dst)
        ledger['moves'].append({'from':old,'to':new})
        atomic_json(journal,ledger)
    for old,text in content.items():
        new=resolve_path(old,catalog=catalog)
        updated=rewrite_markdown(text,old,new,catalog)
        new.write_text(updated)
        ledger['documents'].append({'from':str(old.relative_to(PROJECT)),'to':str(new.relative_to(PROJECT)),
                                    'before_sha256':hashlib.sha256(text.encode()).hexdigest(),
                                    'after_sha256':hashlib.sha256(updated.encode()).hexdigest()})
    # Any root loose file is historical material, not an unregistered run.
    for p in (PROJECT/'outputs').iterdir():
        if p.is_file() and p.name!='README.md':
            dst=management/'历史根文件'/p.name;dst.parent.mkdir(parents=True,exist_ok=True)
            p.rename(dst);ledger['moves'].append({'from':'outputs/'+p.name,'to':str(dst.relative_to(PROJECT))})
    atomic_json(journal,ledger)
    print(json.dumps({'moved':len(ledger['moves']),'documents':len(ledger['documents']),
                      'ledger':str(journal)},ensure_ascii=False,indent=2))


if __name__=='__main__':main()
