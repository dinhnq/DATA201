import numpy
import pandas
from matplotlib import pyplot
import wavio

ecoli = pandas.read_csv('../../res/data/TaranakiStWharf.csv', skiprows=0, header=None, parse_dates=[8])
ecoli.head()

iloc8_ecoli = ecoli.iloc[:,8]
iloc9_ecoli = ecoli.iloc[:,9]
#pyplot.plot(iloc9_ecoli, '.')
#pyplot.show()

x_values = iloc8_ecoli
y_values = iloc9_ecoli

#pyplot.plot(x_values, y_values, '.')
#pyplot.show()

# red if ecoli > 350, amber if it is > 150, green otherwise
inds_amber = numpy.squeeze(numpy.where((iloc9_ecoli > 150) & (iloc9_ecoli <= 350)))
inds_red = numpy.squeeze(numpy.where(iloc9_ecoli > 350))
#inds_green = numpy.squeeze(numpy.where(iloc9_ecoli <= 150))

#pyplot.plot(x_values, y_values, '.g', markersize=10)

x_values = ecoli.iloc[inds_amber,8]
y_values = ecoli.iloc[inds_amber,9]
#pyplot.plot(x_values, y_values, '.k', markersize=10)

x_values = ecoli.iloc[inds_red,8]
y_values = ecoli.iloc[inds_red,9]
#pyplot.plot(x_values, y_values, '.r', markersize=10)

#pyplot.show()

co2 = pandas.read_csv('../../res/data/daily_flask_co2_mlo.csv', skiprows=70, header=None, parse_dates=[0])
co2.head()

co2nz = pandas.read_csv('../../res/data/daily_flask_co2_nzd.csv', skiprows=70, header=None, parse_dates=[0])
co2nz.head()

#Does it match the NZ data? We should plot them both.
#pyplot.plot(co2.iloc[:,-1])
#pyplot.plot(co2nz.iloc[:,-1])

# Why don't they match
#pyplot.plot(co2.iloc[:,0], co2.iloc[:,-1])
#pyplot.plot(co2nz.iloc[:,0], co2nz.iloc[:,-1])

#pyplot.show()

# Inside a computer all data is numbers
# Image and sound data
img = pyplot.imread('../../res/cute.jpeg')
arr_w_h = numpy.shape(img)

#pyplot.imshow(img, cmap='gray')
#pyplot.show()

img = pyplot.imread('../../res/hakatere.jpeg')
#pyplot.imshow(img, cmap='gray')
#pyplot.show()

sound = wavio.read('../../res/tril1.wav')
#pyplot.plot(sound.data[:100])
#pyplot.show()
print(sound.data)
print(sound.data[:100])
print(sound.data[:100].T)

#pyplot.plot(sound.data)
#pyplot.show()

# turn sound into a histogram of power (a spectrogram)
squeze_data = numpy.squeeze(sound.data)
pyplot.specgram(squeze_data)
# pyplot.show()

# Anscombe's dataset

# This has two purposes here:
# - to explore a dataset and see how important plotting can be
# - to see some basic NumPy syntax

data = numpy.array([
[10.0    ,8.04   ,10.0   ,9.14   ,10.0   ,7.46   ,8.0    ,6.58 ],
[8.0     ,6.95   ,8.0    ,8.14   ,8.0    ,6.77   ,8.0    ,5.76 ],
[13.0    ,7.58   ,13.0   ,8.74   ,13.0   ,12.74  ,8.0    ,7.71 ],
[9.0     ,8.81   ,9.0    ,8.77   ,9.0    ,7.11   ,8.0    ,8.84 ],
[11.0    ,8.33   ,11.0   ,9.26   ,11.0   ,7.81   ,8.0    ,8.47 ],
[14.0    ,9.96   ,14.0   ,8.10   ,14.0   ,8.84   ,8.0    ,7.04 ],
[6.0     ,7.24   ,6.0    ,6.13   ,6.0    ,6.08   ,8.0    ,5.25 ],
[4.0     ,4.26   ,4.0    ,3.10   ,4.0    ,5.39   ,19.0   ,12.50],
[12.0    ,10.84  ,12.0   ,9.13   ,12.0   ,8.15   ,8.0    ,5.56 ],
[7.0     ,4.82   ,7.0    ,7.26   ,7.0    ,6.42   ,8.0    ,7.91 ],
[5.0     ,5.68   ,5.0    ,4.74   ,5.0    ,5.73   ,8.0    ,6.89 ],
])

# Basic statistics
mean = numpy.mean(data, axis=0)
std = numpy.std(data, axis=0)
print('means: ',mean)
print('standard deviations: ',std)

corrcoef_cols_1_2 = numpy.corrcoef(data[:,0], data[:,1])
corrcoef_cols_7_8 = numpy.corrcoef(data[:,6], data[:,7])
print('correlation coefficients: col 1& 2: {} and col 7& 8: {}'.format(corrcoef_cols_1_2, corrcoef_cols_7_8))

polyfit_col_1_2 = numpy.polyfit(data[:,0], data[:,1], deg=1)
polyfit_col_3_4 = numpy.polyfit(data[:,2], data[:,3], deg=1)
polyfit_col_5_6 = numpy.polyfit(data[:,4], data[:,5], deg=1)
polyfit_col_7_8 = numpy.polyfit(data[:,6], data[:,7], deg=1)

print(polyfit_col_1_2)
print(polyfit_col_3_4)

# So are they all the same then?
x = numpy.arange(5,14,0.1)

pyplot.subplot(221)
pyplot.plot(data[0,:], data[1,:],'.',markersize=12)
f1 = numpy.poly1d(polyfit_col_1_2)
pyplot.plot(x, f1(x))

pyplot.subplot(222)
pyplot.plot(data[2,:], data[3,:],'.',markersize=12)
f2 = numpy.poly1d(polyfit_col_3_4)
pyplot.plot(x, f2(x))

pyplot.subplot(223)
pyplot.plot(data[4,:], data[5,:],'.',markersize=12)
f3 = numpy.poly1d(polyfit_col_5_6)
pyplot.plot(x, f3(x))

pyplot.subplot(224)
pyplot.plot(data[6,:], data[7,:],'.',markersize=12)
f4 = numpy.poly1d(polyfit_col_7_8)
pyplot.plot(x, f4(x))

pyplot.show()