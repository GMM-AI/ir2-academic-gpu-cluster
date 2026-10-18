"""Figure 7 from the processed UsageLog, using the original plot code."""
from pathlib import Path
import sys,json,hashlib
ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT))
from project_paths import install_data_io_guard, PROCESSED_DATA,RESULTS_SECTION4
install_data_io_guard()
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

def annotate_cutoffs(axes):
    # Table 6's selected empirical utilization cutoffs; not algorithm-detected knees.
    specs=[
        (0,10,'Non-GPU, GPU-enabled: 10%',(0.38,0.925),1.065),
        (1,15,'Non-GPU: 15%',(0.38,0.955),1.095),
        (1,20,'GPU-enabled: 20%',(0.38,0.86),0.985),
        (2,0,'GPU-enabled: 0%',(0.28,0.925),1.065),
    ]
    curves=[[(line.get_xdata().copy(),line.get_ydata().copy()) for line in ax.lines] for ax in axes]
    artists=[]
    for idx,cutoff,label,pos,anchor_y in specs:
        ax=axes[idx]
        ax.axvline(cutoff,color='#333333',linestyle=':',linewidth=2,zorder=2)
        artists.append(ax.annotate(label,xy=(cutoff,anchor_y),xycoords='data',
            xytext=pos,textcoords='axes fraction',ha='left',va='center',fontsize=21,
            color='#222222',arrowprops={'arrowstyle':'-','color':'#333333','lw':1.5},
            bbox={'facecolor':'white','edgecolor':'none','alpha':0.96,'pad':2},zorder=8))
    for ax,before in zip(axes,curves):
        for line,(x,y) in zip(ax.lines,before):
            assert np.array_equal(line.get_xdata(),x)
            assert np.array_equal(line.get_ydata(),y)
    return artists,[{'resource':['CPU','Memory','GPU'][idx],'cutoff_pct':cutoff,'label':label} for idx,cutoff,label,_,_ in specs]

def run(style_only=False):
    sys.path.insert(0,str(ROOT/'04_Analysis/section4'))
    from figure7_original_cells import PREP as prep, PLOT as plot
    data_file='UsageLog20230401_20251130.csv'
    try:
        new=pd.read_csv(PROCESSED_DATA/data_file,low_memory=False)
    except UnicodeDecodeError:
        new=pd.read_csv(PROCESSED_DATA/data_file,low_memory=False,encoding='cp949')
    from common_monitored_cohort import restrict_to_monitored, TRAIN_N
    new=restrict_to_monitored(new)
    scopes={}
    for mode,frame in [('normalized',new)]:
        code=prep.replace('"gpu_util_avg(%)": "GPU"','"gpu_util_avg_normalized(%)": "GPU"')
        scope={'df_selected_2':frame.copy(),'np':np,'pd':pd,'plt':plt}
        exec(compile(code,'figure7_original_cells.PREP','exec'),scope)
        scopes[mode]=scope
    scope=scopes['normalized'];exec(compile(plot,'figure7_original_cells.PLOT','exec'),scope)
    figure=scope['fig']
    scope['axes'][0].set_ylabel('Cumulative Fraction of Containers',fontsize=30)
    for ax in scope['axes']:
        ax.set_yticks(np.linspace(0,1.2,7))
    scope['axes'][0].yaxis.set_major_formatter(matplotlib.ticker.FormatStrFormatter('%.1f'))
    scope['axes'][2].set_xlabel('Utilization (%)\nRelative to allocated GPUs',fontsize=27)
    annotations,cutoffs=annotate_cutoffs(scope['axes'])
    figure.set_size_inches(30,9)
    figure.tight_layout()
    figure.canvas.draw()
    renderer=figure.canvas.get_renderer()
    for artist in annotations:
        b=artist.get_window_extent(renderer)
        assert figure.bbox.contains(b.x0,b.y0) and figure.bbox.contains(b.x1,b.y1)
    figures=RESULTS_SECTION4/'figures';tables=RESULTS_SECTION4/'tables'
    figures.mkdir(parents=True,exist_ok=True);tables.mkdir(parents=True,exist_ok=True)
    name='0401_resource_utilization_container_count_cdf'
    figure.savefig(figures/(name+'.png'),dpi=180,bbox_inches='tight')
    figure.savefig(figures/(name+'.pdf'),bbox_inches='tight')
    points=[];stats=[]
    for mode,scope in scopes.items():
        gpu=scope['gpu_used_data']
        for label,group in [('GPU-Enabled AVG.',gpu),*[(g,gpu.loc[gpu['gpu_allocated(group)'].eq(g)]) for g in scope['groups_sorted']]]:
            values=group['GPU'].dropna()
            stats.append({'version':mode,'group':label,'n':len(values),'mean_pct':values.mean(),'median_pct':values.median(),'p90_pct':values.quantile(.9),'zero_pct':values.eq(0).mean()*100,'le10_pct':values.le(10).mean()*100})
        for resource in ['CPU','Memory','GPU']:
            for group,arr in [('GPU-Enabled AVG.',scope['proportions_5_gpu_used'][resource])]+([] if resource=='GPU' else [('Non-GPU',scope['proportions_5_gpu_not_used'][resource])])+[(g,v[resource]) for g,v in scope['proportions_5_by_gpu_group'].items()]:
                points.extend({'version':mode,'resource':resource,'group':group,'utilization_pct':x,'container_fraction':y} for x,y in zip(scope['thresholds_5'],arr))
    if style_only:
        assert pd.DataFrame(points).to_csv(index=False).encode()==(tables/'figure7_cdf_points.csv').read_bytes()
    else:
        pd.DataFrame(points).to_csv(tables/'figure7_cdf_points.csv',index=False)
    stats=pd.DataFrame(stats)
    if not style_only:
        stats.to_csv(tables/'figure7_gpu_summary.csv',index=False)
    validation={'status':'PASS','source_cells':'04_Analysis/section4/figure7_original_cells.py',
                'corrected_usage_sha256':hashlib.sha256((PROCESSED_DATA/data_file).read_bytes()).hexdigest(),
                'gpu_valid_average_n':int(stats.loc[stats.group.eq('GPU-Enabled AVG.'),'n'].iloc[0]),
                'common_monitored_cohort_n':TRAIN_N,
                'cohort_filter':'container_id intersection with current nonempty IR2 TimeSeries',
                'y_axis_label':figure.axes[0].get_ylabel(),
                'y_tick_step':0.2,
                'selected_cutoff_annotations':cutoffs,
                'annotation_meaning':'Selected utilization cutoffs from Table 6, not automatically detected knees',
                'annotation_bounds_checked':True,
                'y_ticks':figure.axes[0].get_yticks().tolist(),
                'y_tick_labels':[tick.get_text() for tick in figure.axes[0].get_yticklabels()]}
    assert validation['y_axis_label']=='Cumulative Fraction of Containers'
    assert np.allclose(validation['y_ticks'],np.linspace(0,1.2,7))
    assert validation['y_tick_labels']==['0.0','0.2','0.4','0.6','0.8','1.0','1.2']
    assert all(np.allclose(ax.get_yticks(),np.linspace(0,1.2,7)) for ax in figure.axes)
    assert all(not any(tick.get_text() for tick in ax.get_yticklabels()) for ax in figure.axes[1:])

    (tables/'figure7_validation.json').write_text(json.dumps(validation,ensure_ascii=False,indent=2))
    print(stats.to_string(index=False));print(json.dumps(validation,ensure_ascii=False,indent=2))
    plt.close('all')
    return stats,validation

if __name__=='__main__':run(style_only='--style-only' in sys.argv)
