import stackedLSTM as st

# x = st.StackedLSTM(dataFileTarget='varyingData/moving/temperature', modelName="temperature")
# x.networkParams()
# x.train(target_loss=0.5)
# #x.test()

print("Attempting to load..")
y = st.StackedLSTM('varyingData/moving/temperature')
y.networkParams(loading=True)
y.restoreModel("temperature" )
