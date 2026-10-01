"""Reproduce revised Figure 2 from the frozen run-level summary (no training)."""
from pathlib import Path
import argparse
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.ticker import FuncFormatter
parser=argparse.ArgumentParser()
parser.add_argument('--input',type=Path,default=Path(__file__).parent/'locked_test_summary_all_runs.csv')
parser.add_argument('--out',type=Path,default=Path('.'))
a=parser.parse_args();a.out.mkdir(parents=True,exist_ok=True)
f=pd.read_csv(a.input);f=f[f.protocol=='batch_disjoint']
models=['yolov8s','yolo11s','yolo26s','rtdetr_l']
labels=['YOLOv8s','YOLO11s','YOLO26s','RT-DETR-L']
seeds=[42,123,2026];colors=['#0072B2','#D55E00','#009E73'];markers=['o','s','^']
plt.rcParams.update({'font.family':'DejaVu Sans','font.size':16,'axes.labelsize':17,'axes.titlesize':17,'xtick.labelsize':15,'ytick.labelsize':15,'axes.spines.top':False,'axes.spines.right':False})
fig,axes=plt.subplots(1,3,figsize=(11.5,4.6))
settings=[('AP_small','(a) Küçük nesneler',.08,np.arange(0,.081,.02)),('AP_medium','(b) Orta nesneler',.12,np.arange(0,.121,.03)),('AP_large','(c) Büyük nesneler',.50,np.arange(0,.501,.10))]
for ax,(metric,title,upper,ticks) in zip(axes,settings):
 for j,m in enumerate(models):
  g=f[f.model==m].set_index('seed');assert len(g)==3 and set(g.index)==set(seeds)
  v=g.loc[seeds,metric].to_numpy();mean=v.mean();sd=v.std(ddof=1)
  assert v.max()<upper and mean+sd<upper
  ax.errorbar(j,mean,yerr=sd,fmt='D',color='black',capsize=6,markersize=7,elinewidth=1.7,zorder=3)
  for k,x in enumerate(v):ax.scatter(j+(k-1)*.13,x,color=colors[k],marker=markers[k],s=60,zorder=4)
 ax.set_title(title,loc='left',pad=10,fontweight='bold')
 ax.set_xticks(range(4),labels,rotation=30,ha='right');ax.set_xlim(-.48,3.48);ax.set_ylim(0,upper);ax.set_yticks(ticks)
 ax.yaxis.set_major_formatter(FuncFormatter(lambda x,pos:f'{x:.2f}'.replace('.',',')))
 ax.set_ylabel('COCO AP') if ax is axes[0] else None;ax.grid(axis='y',color='#dddddd',linewidth=.8);ax.set_axisbelow(True)
handles=[Line2D([0],[0],color='black',marker='D',linestyle='none',label='Ortalama ± SD')]+[Line2D([0],[0],color=c,marker=m,linestyle='none',label=f'Tohum {s}') for c,m,s in zip(colors,markers,seeds)]
fig.legend(handles=handles,loc='upper center',bbox_to_anchor=(.54,1.0),ncol=4,frameon=False,fontsize=15,columnspacing=1.1)
fig.text(.54,.075,'AP@[0.50:0.95] · Batch-disjoint kilitli test',ha='center',fontsize=14)
fig.text(.54,.025,'Panellerin düşey eksen aralıkları farklıdır.',ha='center',fontsize=13)
fig.subplots_adjust(left=.09,right=.985,top=.79,bottom=.30,wspace=.37)
for ext in ['png','pdf']:fig.savefig(a.out/f'Sekil_2.{ext}',dpi=400)
plt.close(fig)
