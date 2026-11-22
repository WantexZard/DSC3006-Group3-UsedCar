"""Render current tables only; pending C rows are excluded rather than fabricated."""
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

ROOT=Path(__file__).resolve().parents[1]
COLORS={'Lasso_price':'#A9BAC5','Lasso_log':'#3F6F8F','RF_price_mf03':'#4B8B62','RF_log_mf03':'#C8571B'}
LABELS={'Lasso_price':'Lasso / price','Lasso_log':'Lasso / log','RF_price_mf03':'RF / price 0.3','RF_log_mf03':'RF / log 0.3'}
ORDER=list(LABELS)
plt.rcParams.update({'font.family':'DejaVu Sans','font.size':11,'axes.spines.top':False,'axes.spines.right':False})

def main():
    metrics=pd.read_csv(ROOT/'results/holdout_metrics.csv'); bins=pd.read_csv(ROOT/'results/holdout_error_by_price_bin.csv')
    available=[m for m in ORDER if m in set(metrics.loc[metrics.status=='verified','model_id'])]
    data=metrics.set_index('model_id').loc[available]
    footer='Revised evaluation on the same previously viewed holdout (n = 1,208). Log predictions use exp(), without smearing.'
    if len(available)<4: footer+='\nC RF/price 0.3 is pending; legacy RF 1/3 results are excluded.'
    fig,axes=plt.subplots(1,3,figsize=(13.8,4.8))
    for ax,key,title,factor in zip(axes,['rmse_BDT','mae_BDT','r2_price'],['RMSE (million BDT)','MAE (million BDT)','R²'],[1e6,1e6,1]):
        vals=data[key].to_numpy()/factor
        bars=ax.bar(range(len(available)),vals,color=[COLORS[m] for m in available],width=.58)
        ax.set_xticks(range(len(available)),[LABELS[m] for m in available],rotation=15,ha='right')
        ax.set_title(title,fontweight='bold'); ax.set_ylim(0,max(vals)*1.18 if key!='r2_price' else 1)
        ax.bar_label(bars,labels=[f'{v:.4f}' for v in vals],padding=4,fontsize=10)
        ax.yaxis.grid(alpha=.2);ax.set_axisbelow(True)
    fig.suptitle('Angle 4 | Frozen RF max_features = 0.3',fontweight='bold',fontsize=16)
    fig.text(.02,.02,footer,fontsize=9,color='#555555')
    fig.tight_layout(rect=(0,.13,1,.94))
    for ext in ['png','svg']:fig.savefig(ROOT/f'figures/01_holdout_metrics.{ext}',dpi=240)
    plt.close(fig)
    qs=['Q1_low','Q2','Q3','Q4_high'];fig,axes=plt.subplots(1,2,figsize=(13.8,4.8))
    for mid in available:
        part=bins[bins.model_id==mid].set_index('price_bin').loc[qs]
        for ax,key,title in zip(axes,['rmse_BDT','mae_BDT'],['RMSE by true-price quartile','MAE by true-price quartile']):
            ax.plot(range(4),part[key]/1e6,marker='o',label=LABELS[mid],color=COLORS[mid],linewidth=2)
            ax.set_title(title,fontweight='bold');ax.set_ylabel('Error (million BDT)');ax.set_ylim(bottom=0)
            ax.set_xticks(range(4),['Q1 low','Q2','Q3','Q4 high']);ax.yaxis.grid(alpha=.2)
    for ax in axes:ax.legend(fontsize=9)
    fig.suptitle('Angle 4 | Absolute price errors by price range',fontweight='bold',fontsize=16)
    fig.text(.02,.02,footer,fontsize=9,color='#555555');fig.tight_layout(rect=(0,.13,1,.94))
    for ext in ['png','svg']:fig.savefig(ROOT/f'figures/02_holdout_quartiles.{ext}',dpi=240)
    plt.close(fig)
    print('Saved two figure groups in PNG and editable SVG.')
if __name__=='__main__': main()
