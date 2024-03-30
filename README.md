# 以pytorch套件實作LSTM模型進行時序性天氣資料預測

## 使用相關技術:
語言: Python
套件: Pytorch, numpy, matplotlib

## 資料集:
[Markdown Live Preview]([https://markdownlivepreview.com/](https://www.kaggle.com/datasets/sumanthvrao/daily-climate-time-series-data)
資料集包含印度新德里自2013/1/1~2017/4/24之天氣資料, 共有四項參數: meantemp, humidity, wind_speed, meanpressure
此次預測只取meantemp做預測

## 專案內容
預測資料集中最後三成時間段的每日平均溫度資料

## 實作細節:
將前7成的資料分為訓練資料做訓練, 以每一組90天對第91天做預測, 訓練完畢後, 針對後三成的測試資料做測試.
視覺化後, 橫軸為日期, 縱軸為溫度, 可清楚看見預測準度. 同時以MSE, MAE做準度評估.
LSTM模型我採用1000個hidden_size, 6層, 訓練40個epoch

## 預測結果:
=======
<img width="312" alt="Visualized predict result" src="https://github.com/Welonbai/ntustEEMachineLearningCourse/assets/62245152/1bc0c836-655d-4f11-bc07-78933d654f0c">
<img width="381" alt="trainingAndTestingLoss" src="https://github.com/Welonbai/ntustEEMachineLearningCourse/assets/62245152/5d20bf14-d2e5-404a-b32f-db07d22aee36">
<img width="763" alt="mseAndMae" src="https://github.com/Welonbai/ntustEEMachineLearningCourse/assets/62245152/f29462e3-d38f-4de4-8d2e-f957d33ee7fb">

