import json
import pytest
from simscan.makedata import _done


def test_empty_completion_manifest_does_not_skip_missing_scene(tmp_path):
    (tmp_path/'completed').mkdir()
    (tmp_path/'completed/scene_00000.json').write_text(json.dumps(dict(variants=1,levels=[])))
    assert not _done(tmp_path,'scene_00000')


def test_dataset_cli_reports_generation_failures_as_failed_exit(monkeypatch,tmp_path):
    from simscan.cli import main
    monkeypatch.setattr('simscan.makedata.make_dataset',lambda *a,**kw:dict(errors=1))
    with pytest.raises(SystemExit) as e:main(['make-dataset','--out',str(tmp_path),'--count','1'])
    assert e.value.code==1
