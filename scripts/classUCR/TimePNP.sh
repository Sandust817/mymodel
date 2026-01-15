# 指定使用第0块GPU（可根据需求修改为0,1 或 1等）
export CUDA_VISIBLE_DEVICES=0

# 核心参数定义
model_name="TimePNP"
# 可选：若需要统一实验ID，可在这里定义（与原有脚本保持一致性）
# exp_id="UCR_TimeMIL_Exp"

# 2. 批量执行训练（UCR数据集列表，包含经典UCR时间序列分类数据集）
# 注：该列表包含UCR常用基准数据集，可根据你的本地数据集存放情况增删
datasets=(
    "ACSF1" "Adiac" "AllGestureWiimoteY"
    "ArrowHead" "BME" "Beef" "BeetleFly" "BirdChicken"
    "CBF" "Car" "Chinatown" "ChlorineConcentration" "CinCECGTorso" 
    "Coffee" "Computers" "CricketX" "CricketY" "CricketZ"
    "DiatomSizeReduction" "DistalPhalanxOutlineAgeGroup" "DistalPhalanxOutlineCorrect" "DistalPhalanxTW"
    "DodgerLoopDay" "DodgerLoopGame" "DodgerLoopWeekend" "ECG200" "ECG5000"
    "ECGFiveDays" "EOGHorizontalSignal" "Earthquakes" "ElectricDevices" "EthanolLevel"
    "FaceAll" "FaceFour" "FacesUCR" "FiftyWords"
    "Fish" "FordA" "FordB" "FreezerRegularTrain" "FreezerSmallTrain"
    "Fungi" "GestureMidAirD1" "GestureMidAirD2" "GestureMidAirD3" "GesturePebbleZ1"
    "GesturePebbleZ2" "GunPoint" "GunPointAgeSpan" "GunPointMaleVersusFemale" "GunPointOldVersusYoung"
    "Ham" "HandOutlines" "Herring" "HouseTwenty"
    "InsectEPGRegularTrain" "InsectEPGSmallTrain" "InsectWingbeatSound" "ItalyPowerDemand"
    "LargeKitchenAppliances" "Lightning2" "Lightning7" "Mallat" "Meat"
    "MedicalImages" "MelbournePedestrian" "MiddlePhalanxOutlineAgeGroup" "MiddlePhalanxOutlineCorrect"
    "MixedShapesRegularTrain" "MixedShapesSmallTrain" "MoteStrain" "NonInvasiveFetalECGThorax1" "NonInvasiveFetalECGThorax2"
    "OSULeaf" "PickupGestureWiimoteZ" "PigCVP" "Plane"
    "PowerCons" "ProximalPhalanxOutlineAgeGroup" "ProximalPhalanxOutlineCorrect" "ProximalPhalanxTW" "RefrigerationDevices"
    "Rock" "ScreenType" "SemgHandGenderCh2" "SemgHandMovementCh2" "SemgHandSubjectCh2"
    "ShakeGestureWiimoteZ" "ShapeletSim" "ShapesAll" "SmallKitchenAppliances" "SmoothSubspace"
    "SonyAIBORobotSurface1" "StarLightCurves" "Strawberry" "SwedishLeaf"
    "Symbols" "SyntheticControl" "ToeSegmentation1" "ToeSegmentation2" "Trace"
    "TwoLeadECG" "TwoPatterns" "UMD" "UWaveGestureLibraryAll" "UWaveGestureLibraryX"
    "UWaveGestureLibraryY" "UWaveGestureLibraryZ" "Wafer" "Wine" "WordSynonyms"
    "Worms" "WormsTwoClass"
)

for dataset in "${datasets[@]}"; do
    python -u run.py \
      --task_name classification \
      --is_training 1 \
      --root_path "/root/data/UCR/${dataset}/" \
      --model_id "${dataset}" \
      --model "${model_name}" \
      --data UCR \
      --e_layers 3 \
      --batch_size 16 \
      --dropout 0.2 \
      --d_model 128 \
      --d_ff 256 \
      --top_k 3 \
      --des "Exp" \
      --itr 1 \
      --learning_rate 0.001 \
      --train_epochs 150 \
      --patience 30
done
datasets=(
    "AllGestureWiimoteX" "AllGestureWiimoteZ"
     "Crop" "EOGVerticalSignal" "Haptics" "InlineSkate" "MiddlePhalanxTW"
    "OliveOil" "PLAID" "PhalangesOutlinesCorrect" "Phoneme"
    "PigAirwayPressure" "PigArtPressure" "SonyAIBORobotSurface2" "Yoga"
)

for dataset in "${datasets[@]}"; do
    python -u run.py \
      --task_name classification \
      --is_training 1 \
      --root_path "/root/data/UCR/${dataset}/" \
      --model_id "${dataset}" \
      --model "${model_name}" \
      --data UCR \
      --e_layers 2 \
      --batch_size 16 \
      --dropout 0.2 \
      --d_model 128 \
      --d_ff 256 \
      --top_k 3 \
      --des "Exp" \
      --itr 1 \
      --learning_rate 0.001 \
      --train_epochs 150 \
      --patience 30
done
