# LatentDA

ABCIでrun.sh(インタラクティブノードで実行)かrun_cifar10_full1.sh(バッチジョブ)を実行  
  
**[よくいじるハイパーパラメータ]**  
--n_model ニューラルネットワークモデル   
--n_data データセット  
--num_epochs エポック数  
--batch_size_training 学習時のバッチサイズ  
--num_samples 利用する訓練サンプル数 (0で全データ利用)(default:0)  
--n_aug Data augmentationの種類 (0でDAなし)  
--flag_random_layer ALSを行わず、ランダムにDA層を選ぶ場合1  
--flag_als ALSの指定 (0でALS行わない, 1で通常ALS, 2でnaive, 3でgreedy)  
--als_rate ALSの更新の係数  
--flag_adversarial adversarialにする場合1  
--epoch_random 学習開始後何エポックの間ランダムにDA層を選ぶかの指定  
--iter_interval ステップ平均を計算するステップ数  
--flag_alstest ALS計算にテストデータを利用する場合1、訓練データ利用する場合0 (default:1)  
--flag_als_acc ALS計算に精度を利用する場合1、誤差を利用する場合0 (default:0)  
--flag_wandb weights and biasesを用いる場合1 (default:0)  
--save_file 結果のファイルを出力しない場合0 (default:1)  
--gpu_multi マルチGPUを使う場合1  
--loop 0 繰り返し実行の指定  
  
  
**--n_aug 9でガウシアンノイズ</span>**  
  
**--flag_wandb 1でweights and biases利用**  
wandb login  
