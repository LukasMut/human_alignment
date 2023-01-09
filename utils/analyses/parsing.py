import os
import pickle
import pandas as pd
from utils.analyses.training_mapping import Mapper


def exclude_models(results):
    blacklist = ['resnext50_32x4d', 'resnext101_32x8d', 'efficientnet_b1', 'resnet50', 'resnet101', 'resnet152']
    return results[~results.model.isin(blacklist)]


def exclude_clip_models(results):
    blacklist = ['clip_ViT-B/16', 'clip_ViT-L/14', 'clip_RN101', 'clip_RN50x4',
                 'clip_RN50x16', 'clip_RN50x64',
                 'OpenCLIP_ViT-B-32_openai',
                 'OpenCLIP_RN50_openai',
                 'OpenCLIP_RN101_openai',
                 'OpenCLIP_RN50x4_openai',
                 'OpenCLIP_RN50x16_openai',
                 'OpenCLIP_RN50x64_openai',
                 'OpenCLIP_ViT-B-16_openai',
                 'OpenCLIP_ViT-L-14_openai']
    return results[~results.model.isin(blacklist)]


def exclude_ecoset_models(results):
    blacklist = [
        'Alexnet_ecoset',
        'Resnet50_ecoset',
        'VGG16_ecoset'
    ]
    return results[~results.model.isin(blacklist)]


def parse_results_dir(base_dir, modules=['logits', 'penultimate']):
    data = []
    for dir in os.listdir(base_dir):
        path = os.path.join(base_dir, dir)
        for module in modules:
            if os.path.exists(os.path.join(path, module)):
                with open(os.path.join(path, module, 'results.pkl'), 'rb') as f:
                    df = pickle.load(f)
                    df['module'] = module
                    data.append(df)
    results = pd.concat(data)
    mapper = Mapper(results)
    results['training'] = mapper.get_training_objectives()
    # results = results[results.source != 'vit_best']
    return results
