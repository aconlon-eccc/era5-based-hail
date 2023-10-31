import ML.RF as rf

model = rf.rf(source_ds='v0_ml_dataset.csv', max_depths=[None, 10, 20])
