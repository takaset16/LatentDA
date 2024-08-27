# LatentDA


## Paper
[Optimal Layer Selection for Latent Data Augmentation](https://arxiv.org/abs/2408.13426)

## How to Use
Run run.sh (interactive job) or run_sweep001.sh (batch job) on ABCI

## Settings 
--n_model: neural network model   
--n_data: dataset  
--num_epochs: number of training epochs  
--batch_size_training: batch size during training  
--num_samples: number of training samples (default:0)  
--n_aug: kind of data augmentation (0: no data augmentation)  
--flag_random_layer: 1 for randomly selecting DA layers  
--flag_adalase: 1 for AdaLASE  
--initial_als_rate: a hyperparameter for AdaLASE  
--flag_adversarial adversarialにする場合1  
--epoch_random 学習開始後何エポックの間ランダムにDA層を選ぶかの指定  
--iter_interval ステップ平均を計算するステップ数  
--flag_alstest AdaLASE計算にテストデータを利用する場合1、訓練データ利用する場合0 (default:1)  
--flag_als_acc AdaLASE計算に精度を利用する場合1、誤差を利用する場合0 (default:0)  
--flag_wandb weights and biasesを用いる場合1 (default:0)  
--save_file 結果のファイルを出力しない場合0 (default:1)  
--gpu_multi マルチGPUを使う場合1  
--loop 繰り返し実行の指定  
--mean_visual weights and biasesで値の推移を表示するときに平均をとるときの幅(default:1)  
--flag_defaug 入力データにデフォルトでDAを加える場合1(default:1)  
--flag_sign ALSおよびGradient descent ALSでsign関数を利用する場合1 (default: 0)  
  
