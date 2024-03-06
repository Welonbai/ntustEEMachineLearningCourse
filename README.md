 # 臺科大電機所機器學習期末專案

## 以pytorch套件實作LSTM模型進行時序性天氣資料預測

## 資料集:
[Markdown Live Preview]([https://markdownlivepreview.com/](https://www.kaggle.com/datasets/sumanthvrao/daily-climate-time-series-data)
資料集包含印度新德里自2013/1/1~2017/4/24之天氣資料, 共有四項參數: meantemp, humidity, wind_speed, meanpressure
此次預測只取meantemp做預測

## 實作細節:
將前7成的資料分為訓練資料做訓練, 以每一組90天對第91天做預測, 訓練完畢後, 針對後三成的測試資料做測試.
視覺化後, 橫軸為日期, 縱軸為溫度, 可清楚看見預測準度. 同時以MSE, MAE做準度評估.
LSTM模型我採用1000個hidden_size, 6層, 訓練40個epoch

## 預測結果:
![plot](image/Visualized predict result)
