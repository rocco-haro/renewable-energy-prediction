import stackedLSTM as st

x = st.StackedLSTM(1, "prod_Data/training_Data12.csv")
x.networkParams(1)
x.train(target_loss=0.5)
x.test()
#
# print("Attempting to load..")
# y = st.StackedLSTM('varyingData/moving/temperature')
# y.networkParams(loading=True)
# y.restoreModel("temperature" )
