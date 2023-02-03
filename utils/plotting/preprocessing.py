def metric_to_ylabel(metric):
    if metric == 'spearman_rho_correlation':
        return 'Spearman\'s ρ'
    elif metric == 'pearson_corr_correlation':
        return 'Pearson correlation coefficient'
    elif metric == 'zero-shot':
        return 'Zero-shot odd-one-out accuracy'
    elif metric == 'probing':
        return 'Probing odd-one-out accuracy'
    elif metric == 'accuracy':
        return 'Odd-one-out accuracy'
    else:
        raise ValueError()


def reduce_best_final_layer(results, metric, module=None):
    final_layer_indices = []
    for name, group in results.groupby('model'):
        if 'seed1' in name:
            continue

        if module is None:
            final_layer_indices.append(group[group[metric].max() == group[metric]].index[0])
        else:
            if module not in group.module.values.tolist():
                print(name, f'Warning: no {module} layer')
                final_layer_indices.append(group.index[0])
            else:
                final_layer_indices.append(group[group.module == module].index[0])
    final_layer_results = results.iloc[final_layer_indices]
    return final_layer_results
