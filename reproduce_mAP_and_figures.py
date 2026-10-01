from pathlib import Path
import itertools, json, shutil
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy.stats import permutation_test
import argparse
parser=argparse.ArgumentParser(description="Reproduce exact mAP tests and revised figures from frozen run scores")
parser.add_argument("--input",type=Path,default=Path(__file__).parent/"locked_test_summary_all_runs.csv")
parser.add_argument("--out",type=Path,default=Path(__file__).parent/"reproduced")
args=parser.parse_args()
PACK=args.out; PACK.mkdir(parents=True,exist_ok=True)
f=pd.read_csv(args.input)
models=['yolov8s','yolo11s','yolo26s','rtdetr_l']
labels=dict(zip(models,['YOLOv8s','YOLO11s','YOLO26s','RT-DETR-L']))
rows=[]
for protocol in ['random','batch_disjoint']:
 p=f[f.protocol==protocol].pivot(index='seed',columns='model',values='mAP50_95').reindex([42,123,2026])
 for a,b in itertools.combinations(models,2):
  x,y=p[a].to_numpy(),p[b].to_numpy()
  # Two-sided exact paired label-swap test, seed as block: all 2**3 swaps.
  r=permutation_test((x,y),lambda x,y:np.mean(x-y),permutation_type='samples',n_resamples=np.inf,alternative='two-sided')
  u=permutation_test((x,y),lambda x,y:np.mean(x)-np.mean(y),permutation_type='independent',n_resamples=np.inf,alternative='two-sided')
  rows.append(dict(protocol=protocol,model_A=a,model_B=b,mean_A=x.mean(),mean_B=y.mean(),delta_A_minus_B=np.mean(x-y),p_exact_paired=r.pvalue,p_exact_unpaired_sensitivity=u.pvalue,n_seeds=3,n_paired_swaps=len(r.null_distribution)))
stats=pd.DataFrame(rows)
for protocol in stats.protocol.unique():
 ix=stats.index[stats.protocol==protocol];pv=stats.loc[ix,'p_exact_paired'].to_numpy();order=np.argsort(pv)
 adj=np.empty(len(pv));adj[order]=np.minimum(1,np.maximum.accumulate(pv[order]*(len(pv)-np.arange(len(pv)))))
 stats.loc[ix,'p_Holm_within_protocol']=adj
stats.to_csv(PACK/'mAP_exact_tests.csv',index=False)
f.to_csv(PACK/'locked_test_summary_all_runs.csv',index=False)
plt.rcParams.update({'font.family':'DejaVu Sans','font.size':11,'axes.labelsize':12,'legend.fontsize':10,'axes.spines.top':False,'axes.spines.right':False,'savefig.dpi':300})
colors=['#0072B2','#D55E00','#009E73'];markers=['o','s','^']
# Figure 1: seed points and mean +/- SD without bars, common scale.
fig,axes=plt.subplots(1,2,figsize=(10.2,3.8),sharey=True)
for ax,protocol,title in zip(axes,['random','batch_disjoint'],['(a) Rastgele bölme','(b) Batch-disjoint bölme']):
 for j,m in enumerate(models):
  vals=f[(f.protocol==protocol)&(f.model==m)].sort_values('seed').mAP50_95.to_numpy()
  ax.errorbar(j,vals.mean(),yerr=vals.std(ddof=1),fmt='D',color='black',capsize=5,markersize=5,zorder=3)
  for k,v in enumerate(vals):ax.scatter(j+(k-1)*.12,v,color=colors[k],marker=markers[k],s=35,zorder=4)
 ax.set_title(title,fontsize=12,pad=12);ax.set_xticks(range(4),[labels[m] for m in models],rotation=15)
 ax.set_xlim(-.5,3.5);ax.set_ylim(.22,.33);ax.grid(axis='y',alpha=.25);ax.set_axisbelow(True)
axes[0].set_ylabel('COCO mAP@[0.50:0.95]')
from matplotlib.lines import Line2D
handles=[Line2D([0],[0],color='black',marker='D',linestyle='none',label='Ortalama ± SD')]+[Line2D([0],[0],color=c,marker=mk,linestyle='none',label=f'Tohum {s}') for c,mk,s in zip(colors,markers,[42,123,2026])]
fig.legend(handles=handles,loc='upper center',bbox_to_anchor=(.5,1.02),ncol=4,frameon=False)
fig.tight_layout(rect=[0,0,1,.9])
for ext in ['png','pdf']:fig.savefig(PACK/f'Sekil_1.{ext}',bbox_inches='tight')
plt.close(fig)
# Use the final manuscript Figure 2 generator.
import subprocess, sys
subprocess.run([sys.executable, str(Path(__file__).parent/'reproduce_figure2.py'), '--input', str(args.input.resolve()), '--out', str(PACK.resolve())], check=True)
print(stats[['protocol','model_A','model_B','delta_A_minus_B','p_exact_paired','p_Holm_within_protocol','p_exact_unpaired_sensitivity']].to_string(index=False))
