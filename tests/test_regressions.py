import numpy as np
from signlens.dataset import parse_names, read_yolo, StaticHandExtractor
from signlens.classifier import Stabilizer, Prediction
from signlens.features import extract, FLAGS_AT
from signlens.types import Hand, FrameResult

def test_yaml_dictionary_and_multiple_objects(tmp_path):
    (tmp_path/'data.yaml').write_text('names:\n  0: A\n  1: B\n')
    assert parse_names(tmp_path/'data.yaml') == ['A','B']
    (tmp_path/'train/images').mkdir(parents=True)
    (tmp_path/'train/labels').mkdir()
    (tmp_path/'train/images/x.jpg').touch()
    (tmp_path/'train/labels/x.txt').write_text('0 .2 .3 .1 .2\n1 .7 .6 .2 .3\n')
    _,s=read_yolo(tmp_path)
    assert len(s)==2 and s[1].label=='B' and s[1].bbox==(.7,.6,.2,.3)

def test_missing_detection_clears_caption_immediately():
    s=Stabilizer()
    for _ in range(7): result=s.update(Prediction('A',.9))
    assert result.label=='A'
    assert s.update(None) is None
    assert s.update(Prediction('A',.9)) is None

def test_stale_hand_is_not_recorded_as_observed():
    h=Hand(np.ones((21,3)),np.ones((21,3)),stale=True)
    v=extract(FrameResult(0,1.,{'Left':h}))
    assert v[FLAGS_AT]==0 and not v.any()

def test_extractor_rejects_wrong_hand_outside_box():
    pts=np.ones((21,3))*.9
    assert StaticHandExtractor._pick([(pts,pts,.99)],(.2,.2,.1,.1)) is None
