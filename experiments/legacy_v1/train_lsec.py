import sys,json
from pathlib import Path
from collections import Counter
import numpy as np
import joblib
from sklearn.svm import SVC
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score,f1_score,classification_report,confusion_matrix
sys.path.insert(0,str((Path(__file__).resolve().parents[2]/'src')))
from signlens.features import static_features,mirror_world

d=np.load(Path(__file__).resolve().parents[2]/'models/landmarks.npz')
mask=np.array([len(str(y))==1 and str(y).isalpha() for y in d['y']])
w,y,g,p=d['world'][mask],d['y'][mask],d['group'][mask],d['path'][mask]
classes,yi=np.unique(y,return_inverse=True)
split=np.array([Path(x).parts[-3] for x in p])
# Keep the supplied test set untouched. Use validation only for model selection.
tr,va,te=split=='train',split=='valid',split=='test'
assert all(not(set(g[a]) & set(g[b])) for a,b in [(tr,va),(tr,te),(va,te)])
X=static_features(w,'oriented'); Xm=static_features(mirror_world(w),'oriented')
models={'svm_C1':lambda:make_pipeline(StandardScaler(),SVC(C=1,probability=True,random_state=42)), 'svm_C10':lambda:make_pipeline(StandardScaler(),SVC(C=10,probability=True,random_state=42)), 'extra_trees':lambda:ExtraTreesClassifier(n_estimators=300,min_samples_leaf=2,random_state=42,n_jobs=2)}
results=[]
for name,factory in models.items():
 m=factory().fit(np.vstack([X[tr],Xm[tr]]),np.tile(yi[tr],2))
 pred=m.predict(X[va]); row={'model':name,'validation_accuracy':accuracy_score(yi[va],pred),'validation_macro_f1':f1_score(yi[va],pred,labels=np.arange(len(classes)),average='macro',zero_division=0)}
 results.append(row); print(row,flush=True)
best=max(results,key=lambda r:r['validation_macro_f1'])
m=models[best['model']]().fit(np.vstack([X[tr|va],Xm[tr|va]]),np.tile(yi[tr|va],2))
pred=m.predict(X[te]); prob=m.predict_proba(X[te]); confidence=prob.max(1); accepted=confidence>=.8
report={'scope':'Experimental single-frame letter classification; publisher labels not independently validated as LSEC. Not a translator.','classes':classes.tolist(),'samples':{s:int((split==s).sum()) for s in ['train','valid','test']},'source_groups':{s:len(set(g[split==s])) for s in ['train','valid','test']},'group_overlap':0,'selection':results,'selected':best['model'],'test_accuracy':accuracy_score(yi[te],pred),'test_macro_f1':f1_score(yi[te],pred,labels=np.arange(len(classes)),average='macro',zero_division=0),'threshold':.8,'test_accepted':int(accepted.sum()),'test_total':int(te.sum()),'test_accepted_accuracy':float((pred[accepted]==yi[te][accepted]).mean()) if accepted.any() else None,'classification_report':classification_report(yi[te],pred,labels=np.arange(len(classes)),target_names=classes,output_dict=True,zero_division=0),'confusion_matrix':confusion_matrix(yi[te],pred,labels=np.arange(len(classes))).tolist(),'limitations':['Not signer-independent: signer IDs are unavailable.','Related video frames may share different source IDs.','Only successfully extracted annotated hands evaluated.','No unknown/non-sign test set; threshold is heuristic, not a calibrated probability guarantee.','No motion, two-hand words, or linguistic validation.']}
from signlens.reporting import report_run
reports=report_run('legacy_training')
out=(Path(__file__).resolve().parents[2]/'models'/'legacy_runs'/reports.name); out.mkdir(parents=True,exist_ok=True)
joblib.dump({'model':m,'mode':'oriented','classes':classes.tolist(),'threshold':.8,'experimental':True},out/'sign_classifier.joblib')
(reports/'evaluation.json').write_text(json.dumps(report,indent=2))
(reports/'split_manifest.json').write_text(json.dumps([{'path':str(path),'source':str(group),'label':str(label),'split':str(s)} for path,group,label,s in zip(p,g,y,split)],indent=2))
print(json.dumps({k:v for k,v in report.items() if k not in ['classification_report','confusion_matrix']},indent=2),flush=True)
