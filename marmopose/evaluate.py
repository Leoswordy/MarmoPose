import os
import seaborn as sns
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from textwrap import dedent
from typing import Dict, Any, List

import sleap


def plot_2d_distance(bodyparts: List[str], metrics: Dict[str, Any]) -> None:
    """
    Plot 2D distance between points.

    Args:
        bodyparts: List of bodyparts names.
        metrics: Dictionary of metrics data.
    """
    dist = metrics['dist.dists']

    df_errors = pd.DataFrame(dist, columns=bodyparts)
    df_melted = (df_errors.melt(var_name='BodyParts', value_name='Error(px)')
                 .dropna().query('`Error(px)` <= 40'))


    plt.figure(figsize=(15, 8))
    sns.boxplot(x='BodyParts', y='Error(px)', data=df_melted, fliersize=0.5, showfliers=False)
    sns.stripplot(x='BodyParts', y='Error(px)', data=df_melted, color='grey', size=1, jitter=True)
    plt.title(f'Distance between 2D predictions and ground truth')
    plt.xticks(rotation=45) 
    plt.yticks(np.arange(0, df_melted['Error(px)'].max()+1, step=5))
    plt.tight_layout()
    plt.show()


def evaluate(config: Dict[str, Any], model_dir: str, label_path: str, verbose: bool = True) -> None:
    """
    Evaluate a model with given labels.

    Args:
        config: Configuration parameters dictionary.
        model_dir: Directory where model files are located.
        label_path: Path to labels.
        verbose: Enable verbose mode, defaults to True.
    """
    output_path = f'/Users/leosword/Library/CloudStorage/Nutstore-1203442707@qq.com/MarmoPose/data/predictions/{os.path.basename(model_dir)}_{os.path.basename(label_path)}'
    labels_gt = sleap.load_file(label_path)
    if os.path.exists(output_path):
        if verbose: print(f'Loading labels from: {output_path}')
        labels_pr = sleap.load_file(output_path)
    else:
        if verbose: print(f'Evaluating {label_path} using {model_dir}')
        progress_reporting = 'none' if not verbose else 'rich'
        predictor = sleap.load_model(model_dir, batch_size=4, progress_reporting=progress_reporting)
        labels_pr = predictor.predict(labels_gt)
        labels_pr.save(output_path, with_images=False, embed_all_labeled=False)

    metrics = sleap.nn.evals.evaluate(labels_gt, labels_pr)
    print(dedent(f"""
                 {'vis.precision:':<15} {metrics['vis.precision']}
                 {'vis.recall:':<15} {metrics['vis.recall']}
                 {'dist.avg:':<15} {metrics['dist.avg']}
                 {'oks.mOKS:':<15} {metrics['oks.mOKS']}
                 {'oks.mAP:':<15} {metrics['oks_voc.mAP']}
                 """))
    plot_2d_distance(config['bodyparts'], metrics)