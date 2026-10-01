"""Check the released frozen results, manuscript tables and package hashes."""
from pathlib import Path
import hashlib, json, re
import numpy as np
import pandas as pd

root=Path(__file__).resolve().parent
f=pd.read_csv(root/'locked_test_summary_all_runs.csv')
models={'YOLOv8s':'yolov8s','YOLO11s':'yolo11s','YOLO26s':'yolo26s','RT-DETR-L':'rtdetr_l'}
protocols={'Rastgele':'random','Batch-disjoint':'batch_disjoint'}
assert len(f)==24 and not f.duplicated(['protocol','model','seed']).any()
assert set(f.protocol)==set(protocols.values()) and set(f.model)==set(models.values())
assert f.groupby(['protocol','model']).seed.apply(lambda s:set(s)=={42,123,2026}).all()
def pair(text):return [float(x.replace(',','.').replace('−','-')) for x in re.split(r'\s*±\s*',text)]
t=pd.read_csv(root/'manuscript_tables/Tablo_3.csv')
fields=['mAP50_95','AP50','AP75','test_precision','test_recall','test_f1']
checks=0
t=pd.read_csv(root/'manuscript_tables/Tablo_1.csv')
split=pd.read_csv(root/'02_splits/split_summary.csv').set_index(['protocol','split'])
for _,r in t.iterrows():
 s=split.loc[(protocols[r.iloc[0]],{'Eğitim':'train','Doğrulama':'val','Test':'test'}[r.iloc[1]])]
 np.testing.assert_array_equal(r.iloc[2:].astype(int),s[['images','objects','small','medium','large']].astype(int));checks+=5
t=pd.read_csv(root/'manuscript_tables/Tablo_2.csv')
complexity=pd.read_csv(root/'10_reviewer_sensitivity/stage10c_checkpoint_complexity/Supplementary_Table_Model_Complexity_Manuscript.csv').set_index('Model')
for _,r in t.iterrows():
 c=complexity.loc[r.iloc[0]]
 assert int(str(r.iloc[2]).replace(',',''))==int(c['Parameters'])
 np.testing.assert_allclose([float(r.iloc[3]),float(r.iloc[4])],c[['GFLOPs @640','Frozen best.pt mean size [MiB]']].astype(float),atol=.00001,rtol=0);checks+=3
t=pd.read_csv(root/'manuscript_tables/Tablo_3.csv')
for _,r in t.iterrows():
 g=f[(f.protocol==protocols[r.iloc[0]])&(f.model==models[r.iloc[1]])]
 for j,k in enumerate(fields,2):
  np.testing.assert_allclose(pair(r.iloc[j]),[g[k].mean(),g[k].std(ddof=1)],atol=.000051,rtol=0);checks+=2
t=pd.read_csv(root/'manuscript_tables/Tablo_4.csv')
err=pd.read_csv(root/'locked_test_error_taxonomy_mean_sd.csv').set_index(['protocol','model'])
for _,r in t.iterrows():
 key=models[r.iloc[0]];e=err.loc[('batch_disjoint',key)];g=f[(f.protocol=='batch_disjoint')&(f.model==key)]
 expected=[[100*e.fn_rate_mean,100*e.fn_rate_sd],[100*e.small_fn_rate_mean,100*e.small_fn_rate_sd],[100*e.background_share_of_fp_mean,100*e.background_share_of_fp_sd],[e.mean_tp_iou_mean,e.mean_tp_iou_sd],[g.processing_ms_mean.mean(),g.processing_ms_mean.std(ddof=1)],[g.processing_fps.mean(),g.processing_fps.std(ddof=1)]]
 for j,x in enumerate(expected,1):
  np.testing.assert_allclose(pair(r.iloc[j]),x,atol=.0051 if j in [1,2,3,6] else .000051,rtol=0);checks+=2
runs=pd.read_csv(root/'10_reviewer_sensitivity/Run_Training_Details_Final.csv')
hits=pd.read_csv(root/'10_reviewer_sensitivity/stage10c_checkpoint_complexity/Checkpoint_Hash_Registry_Hits.csv')
assert len(runs)==len(hits)==24 and runs.checkpoint_hash_registered_in_final_manifest.all()
m=runs.merge(f,on=['protocol','seed'],suffixes=('_training','_test'))
m=m[m.model_key==m.model_test]
assert len(m)==24 and (m.best_pt_sha256==m.checkpoint_sha256).all()
assert set(runs.best_pt_sha256)==set(hits.sha256) and (hits.registry_hit_count>0).all()
t=pd.read_csv(root/'manuscript_tables/Tablo_5.csv')
for _,r in t.iterrows():
 g=runs[(runs.protocol==protocols[r.iloc[0]])&(runs.model_key==models[r.iloc[1]])].set_index('seed')
 assert [int(x.strip()) for x in r.iloc[2].split('/')]==g.loc[[42,123,2026],'epochs_completed'].tolist()
stats=pd.read_csv(root/'mAP_exact_tests.csv')
t=pd.read_csv(root/'manuscript_tables/Tablo_7.csv')
sensitivity=pd.read_csv(root/'10_reviewer_sensitivity/pHash0_sensitivity_mean_sd.csv').set_index('model')
for _,r in t.iterrows():
 s=sensitivity.loc[models[r.iloc[0]]]
 for j,k in [(1,'original_mAP50_95'),(2,'sensitivity_mAP50_95'),(4,'original_f1'),(5,'sensitivity_f1')]:
  np.testing.assert_allclose(pair(r.iloc[j]),[s[k+'_mean'],s[k+'_sd']],atol=.000051,rtol=0);checks+=2
 for j,k in [(3,'delta_mAP50_95_mean'),(6,'delta_f1_mean')]:
  np.testing.assert_allclose(float(r.iloc[j]),s[k],atol=.00000051,rtol=0);checks+=1
co=json.loads((root/'10_reviewer_sensitivity/batch_disjoint_test_minus_pHash0_candidate_coco.json').read_text())
assert len(co['images'])==199 and len(co['annotations'])==705
t=pd.read_csv(root/'manuscript_tables/Ek_Tablo_S1.csv')
assert len(stats)==len(t)==12
for _,r in t.iterrows():
 s=stats[(stats.protocol==protocols[r.iloc[0]])&(stats.model_A==models[r.iloc[1]])&(stats.model_B==models[r.iloc[2]])].iloc[0]
 actual=[float(str(r.iloc[j]).replace(',','.').replace('−','-')) for j in [3,4,5]]
 np.testing.assert_allclose(actual,[s.delta_A_minus_B,s.p_exact_paired,s.p_Holm_within_protocol],atol=.00000051,rtol=0);checks+=3
manifest=root/'PACKAGE_SHA256.json'
if manifest.exists():
 for path,digest in json.loads(manifest.read_text()).items():
  assert hashlib.sha256((root/path).read_bytes()).hexdigest()==digest,path
print(f'PASS: 24 runs; 24 matching checkpoint hashes; {checks} manuscript numerical cells; 12 exact tests; package SHA-256 manifest.')
