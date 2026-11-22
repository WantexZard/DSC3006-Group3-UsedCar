"""Reuse Lasso, import C's RF/price 0.3, and assemble comparable result tables.

No models are trained. If C's files are absent, the package is explicitly marked
pending_C. Legacy RF/price 1/3 is NEVER substituted. An optional C import requires
both prediction files and a frozen config so row/target/fold/config checks can run.
"""
from pathlib import Path
import argparse
import hashlib
import json
import shutil
import numpy as np
import pandas as pd
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from shared_preprocessing import load_development, load_holdout
from update_rf_log_03 import RF_CONFIG, HASHES, save_json, sha

COLS = ['model_id','target_scale','evaluation_split','row_id','group_id','cv_fold','y_true','y_pred_price']
NAMES = {'Lasso_price':'Lasso / price','Lasso_log':'Lasso / log-price',
         'RF_price_mf03':'RF / price (0.3)','RF_log_mf03':'RF / log-price (0.3)'}
ORDER = list(NAMES)


def validate(frame, reference, model_id, split):
    assert len(frame) == len(reference), f'{model_id}: row count mismatch'
    assert frame.row_id.is_unique, f'{model_id}: duplicate row_id'
    assert set(frame.row_id) == set(reference.row_id), f'{model_id}: missing or extra row_id'
    merged = frame.merge(reference, on='row_id', validate='one_to_one', suffixes=('', '_ref'))
    for key in ['y_true','group_id','cv_fold']:
        if key in frame:
            assert np.array_equal(merged[key].to_numpy(), merged[key+'_ref'].to_numpy()), f'{model_id}: {key} mismatch'
    assert np.isfinite(merged.y_pred_price).all()
    merged['model_id'], merged['target_scale'], merged['evaluation_split'] = model_id, ('log_price' if 'log' in model_id else 'price'), split
    return merged[COLS].sort_values('row_id').reset_index(drop=True)


def legacy_frame(path, reference, split):
    original = pd.read_csv(path, float_precision="round_trip")
    frames=[]
    for target, model_id in [('price','Lasso_price'),('log_price','Lasso_log')]:
        part = original[(original.model=='Lasso') & (original.target==target)].copy()
        part = part.rename(columns={'y_true_price_BDT':'y_true','y_pred_price_BDT':'y_pred_price'})
        normalized = validate(part, reference, model_id, split)
        assert np.array_equal(normalized.y_pred_price.to_numpy(), part.sort_values('row_id').y_pred_price.to_numpy())
        frames.append(normalized)
    return frames


def metrics(frame):
    resid = frame.y_true.to_numpy() - frame.y_pred_price.to_numpy()
    return {'n':len(frame), 'rmse_BDT':float(np.sqrt(mean_squared_error(frame.y_true,frame.y_pred_price))),
            'mae_BDT':float(mean_absolute_error(frame.y_true,frame.y_pred_price)),
            'r2_price':float(r2_score(frame.y_true,frame.y_pred_price)),
            'mean_residual_BDT':float(resid.mean()),
            'negative_prediction_count':int((frame.y_pred_price<0).sum())}


def normalize_c(path, reference, split):
    frame=pd.read_csv(path, float_precision="round_trip")
    aliases={'y_true_price_BDT':'y_true','y_pred_price_BDT':'y_pred_price','fold':'cv_fold'}
    frame=frame.rename(columns={a:b for a,b in aliases.items() if a in frame and b not in frame})
    # Require a file containing only C's selected RF/price rows to avoid guessing
    # between candidate models or selecting by test performance.
    for col in ['model_id','model','target_scale','target']:
        if col in frame:
            assert frame[col].nunique()==1, f'C file contains multiple {col} values; provide frozen RF/price only'
    for col in ['target_scale','target']:
        if col in frame:
            assert frame[col].iloc[0] in ['price','raw_price'], 'C target must be raw price'
    if split=='development_oof':
        assert 'cv_fold' in frame, 'C OOF must include fold assignments'
    assert {'row_id','y_true','y_pred_price'}.issubset(frame.columns)
    return validate(frame, reference, 'RF_price_mf03', split)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data-dir',required=True)
    parser.add_argument('--package-dir',default=str(Path(__file__).resolve().parents[1]))
    parser.add_argument('--c-oof')
    parser.add_argument('--c-holdout')
    parser.add_argument('--c-config')
    args=parser.parse_args()
    root=Path(args.package_dir); results=root/'results'; old=results/'audit'/'legacy_1over3'
    for name,expected in HASHES.items(): assert sha(Path(args.data_dir)/name)==expected
    _,yd,md,_=load_development(args.data_dir); _,yt,mt=load_holdout(args.data_dir)
    rd=md[['row_id','group_id','cv_fold']].copy(); rd['y_true']=yd
    rt=mt[['row_id','group_id','cv_fold']].copy(); rt['y_true']=yt
    dev=legacy_frame(old/'development'/'predictions.csv',rd,'development_oof')
    test=legacy_frame(old/'final'/'holdout_predictions.csv',rt,'holdout_revision')
    pd.concat(dev).to_csv(results/'reused_lasso'/'oof_predictions.csv',index=False)
    pd.concat(test).to_csv(results/'reused_lasso'/'holdout_predictions.csv',index=False)
    dev.append(validate(pd.read_csv(results/'rf_log_03'/'oof_predictions.csv', float_precision="round_trip"),rd,'RF_log_mf03','development_oof'))
    test.append(validate(pd.read_csv(results/'rf_log_03'/'holdout_predictions.csv', float_precision="round_trip"),rt,'RF_log_mf03','holdout_revision'))
    c_files=[args.c_oof,args.c_holdout,args.c_config]
    assert not any(c_files) or all(c_files), 'Supply all three C files together for verified integration'
    c_verified=False
    if all(c_files):
        config=json.loads(Path(args.c_config).read_text())
        assert config.get('target_scale', 'price') == 'price', 'C config must target raw price'
        if 'data_sha256' in config:
            assert config['data_sha256'] == HASHES, 'C data hashes mismatch'
        cfg=config.get('random_forest',config.get('rf_config',config))
        for key in ['n_estimators','max_features','min_samples_leaf','bootstrap','random_state']:
            assert cfg.get(key)==RF_CONFIG[key], f'C configuration mismatch: {key}'
        dev.append(normalize_c(args.c_oof,rd,'development_oof'))
        test.append(normalize_c(args.c_holdout,rt,'holdout_revision'))
        (results/'source_C').mkdir(exist_ok=True)
        for src,name in zip(c_files,['oof_predictions.csv','holdout_predictions.csv','model_config.json']):
            shutil.copy2(src,results/'source_C'/name)
        c_verified=True
    dev_all=pd.concat(dev).sort_values(['model_id','row_id']).reset_index(drop=True)
    test_all=pd.concat(test).sort_values(['model_id','row_id']).reset_index(drop=True)
    dev_all.to_csv(results/'predictions.csv',index=False)
    dev_all.to_csv(results/'development'/'predictions.csv',index=False)
    test_all.to_csv(results/'holdout_predictions_with_metadata.csv',index=False)
    test_all[['row_id','y_true','y_pred_price','model_id','target_scale']].to_csv(results/'test_predictions.csv',index=False)
    dev_rows=[]; test_rows=[]; fold_rows=[]
    for model_id in ORDER:
        for full,is_dev in [(dev_all,True),(test_all,False)]:
            part=full[full.model_id==model_id]
            row={'model_id':model_id,'model':NAMES[model_id],
                 'target_scale':'log_price' if 'log' in model_id else 'price',
                 'max_features':0.3 if model_id.startswith('RF') else np.nan,
                 'source':'C_frozen' if model_id=='RF_price_mf03' else ('new_RF_log_03' if model_id.startswith('RF') else 'reused_Lasso_no_refit'),
                 'status':'verified' if len(part) else 'pending_C_predictions'}
            if len(part):
                row.update(metrics(part))
                if is_dev:
                    fm=[]
                    for fold,fg in part.groupby('cv_fold'):
                        frow={'model_id':model_id,'cv_fold':int(fold),**metrics(fg)};fm.append(frow);fold_rows.append(frow)
                    row['mean_fold_rmse_BDT']=float(np.mean([f['rmse_BDT'] for f in fm]))
                    row['sd_fold_rmse_BDT']=float(np.std([f['rmse_BDT'] for f in fm],ddof=1))
            (dev_rows if is_dev else test_rows).append(row)
    dm=pd.DataFrame(dev_rows);tm=pd.DataFrame(test_rows)
    dm.to_csv(results/'summary_metrics.csv',index=False);dm.to_csv(results/'development'/'summary_metrics.csv',index=False)
    tm.to_csv(results/'holdout_metrics.csv',index=False);tm.to_csv(results/'final'/'holdout_metrics.csv',index=False)
    pd.DataFrame(fold_rows).to_csv(results/'development'/'fold_metrics.csv',index=False)
    residual=pd.concat([dm.assign(evaluation_split='development_oof'),tm.assign(evaluation_split='holdout_revision')])
    residual[['model_id','target_scale','evaluation_split','n','mean_residual_BDT','status']].to_csv(results/'mean_residuals.csv',index=False)
    # Compute true-price bins ONCE on the unique holdout rows, then join by row_id.
    labels=['Q1_low','Q2','Q3','Q4_high']
    price_bin,edges=pd.qcut(rt.y_true,4,labels=labels,retbins=True)
    bins=rt[['row_id']].copy();bins['price_bin']=price_bin
    binned=test_all.merge(bins,on='row_id',validate='many_to_one')
    rows=[]
    for (model_id,price_bin),g in binned.groupby(['model_id','price_bin'],observed=True):
        rows.append({'model_id':model_id,'price_bin':str(price_bin),**metrics(g)})
    pd.DataFrame(rows).to_csv(results/'holdout_error_by_price_bin.csv',index=False)
    cfg=json.loads((results/'rf_log_03'/'model_config.json').read_text())
    cfg.update({'c_raw_price_verified':c_verified,'lasso':'reused prior GroupKFold predictions, no refit',
                'c_raw_price_status':'row_id, y_true, fold and required RF config checks passed' if c_verified else 'C predictions/config not available; raw RF excluded from active figures',
                'historical_config':'results/development/config_used.json retained unchanged as a historical record, not current RF settings',
                'price_quartile_edges_BDT':edges.tolist(),
                'old_RF_results':'results/audit/legacy_1over3; not included in current metrics or figures'})
    if c_verified:
        cfg['c_raw_price_provenance'] = {
            'config': 'results/source_C/model_config.json',
            'input_sha256': {name: sha(Path(src)) for name, src in zip(
                ['oof', 'holdout', 'config'], c_files)},
            'random_forest': config.get('random_forest', {}),
            'selection': config.get('selection', {}),
            'software_as_reported_by_C': config.get('source_model_config', {}).get('software', {}),
            'integration_model_fits': 0,
            'comparison_limit': 'Raw-price-selected frozen RF configuration; no separate log-target tuning; source software environments differ.'}
    save_json(results/'model_config.json',cfg)
    audit={'status':'PASS' if c_verified else 'PARTIAL_PENDING_C',
           'data_hashes_match':True,'group_overlap':len(set(md.group_id)&set(mt.group_id)),
           'rf_log_oof_rows':4789,'rf_log_holdout_rows':1208,
           'lasso_predictions_unchanged':True,'rf_price_refits':0,'lasso_refits':0,
           'rf_log_refits_this_integration':0,
           'C_row_alignment_verified':c_verified,'expected_full_model_count':4,
           'available_model_count':int(dev_all.model_id.nunique()),
           'duplicate_oof_keys':int(dev_all[['model_id','row_id']].duplicated().sum()),
           'duplicate_holdout_keys':int(test_all[['model_id','row_id']].duplicated().sum()),
           'same_previously_viewed_holdout':True,'no_smearing':True}
    save_json(results/'acceptance_checks.json',audit)
    data={'status':audit['status'],'development':dm.replace({np.nan: None}).to_dict(orient='records'),
          'holdout':tm.replace({np.nan: None}).to_dict(orient='records'),
          'quartiles':rows,'c_verified':c_verified}
    save_json(results/'presentation_data.json',data)
    print(json.dumps(audit,indent=2));print(tm.to_string(index=False))

if __name__=='__main__': main()
