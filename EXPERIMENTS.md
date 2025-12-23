# Experiments for Captcha OCR

Tracks all experiments performed with paths to their configs, wandb logs, kaggle notebook version number for runs and results.

The motivation behind the set of experiments is to try minimize the number of factors that affect a result in an individual run. 
- The choice of the image encoder size is determined through experiments on fixed datasets so that downstream effects like pooling or sequence models being inadequate are not a factor.

## Datasets
A number of datasets are generated. Here is a list:

| Dataset name | Dataset Config | Number of Captchas | words list | font_root | random_capitalize | remarks |
| --- | --- | --- | --- | --- | --- | --- |
| fixed_clean_dataset_train | experiments/dataset_configs/clean_dataset_train.yaml | 50k | experiments/diverse_words.tsv | train_font_library | False |clean dataset, 5 letter alphanumeric words, no noise, no distortions, black text on white background |
| fixed_clean_dataset_val | experiments/dataset_configs/clean_dataset_val.yaml | 10k | experiments/diverse_words.tsv | val_font_library | False | clean dataset, 5 letter alphanumeric words, no noise, no distortions, black text on white background |
| fixed_noisy_dataset_train | experiments/dataset_configs/noisy_dataset_train.yaml | 50k | experiments/diverse_words.tsv | train_font_library | False | noisy dataset, 5 letter alphanumeric words, noise, distortions, randomly colored text on a light background |
| fixed_noisy_dataset_train_large | experiments/dataset_configs/noisy_dataset_train.yaml | 400k | experiments/diverse_words.tsv | train_font_library | False | noisy dataset, 5 letter alphanumeric words, noise, distortions, randomly colored text on a light background |
| fixed_noisy_dataset_val | experiments/dataset_configs/noisy_dataset_val.yaml | 10k | experiments/diverse_words.tsv | val_font_library | False | noisy dataset, 5 letter alphanumeric words, noise, distortions, randomly colored text on a light background |
| variable_clean_dataset_train | experiments/dataset_configs/clean_dataset_train.yaml | 200k | experiments/wiki_words_20k.txt | train_font_library | False | clean dataset, words of length between 4 and 12, mean around 7, no noise, no distortions, black text on white background |
| variable_clean_dataset_val | experiments/dataset_configs/clean_dataset_val.yaml | 10k | experiments/wiki_words_20k.txt | val_font_library | False | clean dataset, words of length between 4 and 12, mean around 7, no noise, no distortions, black text on white background |
| variable_clean_dataset_val_generation | experiments/dataset_configs/clean_dataset_val.yaml | 10k | experiments/wiki_words_2k.txt | val_font_library | True | clean dataset, words of length between 4 and 12, mean around 7, no noise, no distortions, black text on white background |
| variable_noisy_dataset_train | experiments/dataset_configs/noisy_dataset_train.yaml | 200k | experiments/wiki_words_20k.txt | train_font_library | False | noisy dataset, words of length between 4 and 12, mean around 7, noise, distortions, randomly colored text on a light background |
| variable_noisy_dataset_train_large | experiments/dataset_configs/noisy_dataset_train.yaml | 800k | experiments/wiki_words_20k.txt | train_font_library | True | noisy dataset, words of length between 4 and 12, mean around 7, noise, distortions, randomly colored text on a light background |
| variable_noisy_dataset_val | experiments/dataset_configs/noisy_dataset_val.yaml | 10k | experiments/wiki_words_20k.txt | val_font_library | False | noisy dataset, words of length between 4 and 12, mean around 7, noise, distortions, randomly colored text on a light background |
| variable_noisy_dataset_val_generation | experiments/dataset_configs/noisy_dataset_val.yaml | 10k | experiments/wiki_words_2k.txt | val_font_library | True | noisy dataset, words of length between 4 and 12, mean around 7, noise, distortions, randomly colored text on a light background |

These datasets enable comparison of the following:
- Effect of dataset size on training (for fixed and variable datasets) and (noisy and clean datasets)
- Effect of fixed or variable datasets
- Effect of noisy or clean datasets


## Classification

### Determining the size of dataset

All of the following experiments are performed on resnet-base (as defined in the training config below). In the interest of not having too many experiments, we experiment only on fixed length datasets: fixed_clean and fixed_noisy. For variable length datasets, the idea is to use heuristics from here and the number of characters to get a ball-park figure for the number of images required. This would depend on the sequence model used downstream, and hence is deferred to the generation experiments.

Since each of these are done in a kaggle notebook with a different version, the training config remains the same. In a persistent set up, we would be able to define different configs for different experiments that point to the appropriate dataset as required.

| Experiment Name | Config Path | Wandb Link| Kaggle Notebook Link | Kaggle Notebook Version | Results |
| --- | --- | --- | --- | --- | --- |
| fixed_clean_small | experiments/training_configs/classification/resnet_base.yaml | | | | |
| fixed_clean_large | experiments/training_configs/classification/resnet_base.yaml | | | | |
| fixed_noisy_small | experiments/training_configs/classification/resnet_base.yaml | | | | |
| fixed_noisy_large | experiments/training_configs/classification/resnet_base.yaml | | | | |

### Determining the size of model

All of the following experiments are performed on fixed_noisy_train_large

| Experiment Name | Config Path | Wandb Link| Kaggle Notebook Link | Kaggle Notebook Version | Results |
| --- | --- | --- | --- | --- | --- |
| resnet_small | experiments/training_configs/classification/resnet_small.yaml | | | | |
| resnet_base | experiments/training_configs/classification/resnet_base.yaml | | | | |
| resnet_large | experiments/training_configs/classification/resnet_large.yaml | | | | |
| --- | --- | --- | --- | --- | --- |

### Comparing architectures

There are primarily two kinds of experiments in this subsection:
- Comparing between ConvNext and ResNet
- Determining a 

### Determining the effect of asymmetric compression

## Generation