import stackedLSTM as st

x = st.StackedLSTM('varyingData/moving/temperature')
x.networkParams()
x.train(0.0025)
#x.test()

print("Attempting to load..")
y = st.StackedLSTM('varyingData/moving/temperature')
y.networkParams(loading=True)
y.restoreModel("model" )
