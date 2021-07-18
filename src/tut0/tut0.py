import numpy
import pandas
from matplotlib import pyplot

# numpy practices
A = numpy.ones((6,4))
B = numpy.eye(6,4)
print(B)

AB = 2*A + B
print(AB)

dot_AB = numpy.dot(A,B.T)
print(dot_AB)

C = numpy.arange(3,30)
C = numpy.reshape(C, (3,3,3))
print(C)

D = numpy.ones((4,4))
D[:,1] = 2
D[:,2] = 3
D[:,3] = 4
D = numpy.dot(D[[0,2],:], D[:,[1,3]])
numpy.where(D>20)
print(D)

# pandas practices
step = 3
a = numpy.arange(2,196,step)
b = numpy.arange(3,196,step)

ab = 2  # total of matrices
rank_col = 1
total_col = int(196/step * ab + rank_col)
indices = numpy.zeros(total_col, dtype=int)
indices[0] = rank_col
indices[1::2] = a
indices[2::2] = b

girls = pandas.read_excel("../../res/data/Names.xlsx", sheet_name='Girls\' Names', skiprows=4, index_col=None, usecols=indices, engine='openpyxl')
girls = girls.loc[1:100]
# loc is label-based                loc[row_label, column_label]
# iloc is integer position-based    loc[row_position, column_position]
girls.head()

start_col = 2
end_col = total_col
step_to_yr = 2
for yr in range(start_col, end_col, step_to_yr):
    index = yr-1
    col_name = girls.columns[index]
    girls_per_yr = girls.iloc[:, yr].sum()
    print(col_name, girls_per_yr)

total = pandas.read_csv('../../res/data/BirthsNZ.csv', skiprows=[0,2])
total.head()
total.tail()

birth_data = total.iloc[:28,:]
birth_data = total.iloc[:,[0,-1]]
total.describe()

#pyplot.plot(total.iloc[:,-1])
#pyplot.show()

# find the unique names in the list
popular = pandas.unique(girls.iloc[:,1::2].values.ravel())
print(girls.iloc[:,1::2])

# count how often each one of those occurs
popular_size = numpy.shape(popular)
counts = numpy.zeros(popular_size)
i = 0
for name in popular:
    ind = numpy.where(girls==name)
    for j in range(len(ind[0])):
        x_index = ind[0][j]
        y_index = ind[1][j] + 1

        counts[i] += girls.iloc[x_index, y_index]
    i += 1

print(counts)

# if your list of popular names: popular and the counts: popcounts
newshape_popular = numpy.reshape(popular,(len(popular), 1))
newshape_popcounts = numpy.reshape(counts, (len(counts), 1))
newdata = numpy.concatenate((newshape_popular, newshape_popcounts), axis=1)
print(newdata)

nd1 = numpy.vstack((popular,counts))
print(nd1)

# put that into a new dataframe,
# track the popularity of the name 'Christine' over time
# making a plot of it
popularity = pandas.DataFrame(data=newdata,index=None,columns=['Name','Count'])
popularity_Christine = numpy.where(girls=='Christine')
print(popularity_Christine)

# write a lambda function to get the last letter
char_as_int = lambda x: ord(x)-97
get_last_letter = lambda x: x[-1]

print(char_as_int(get_last_letter('abc')))

# write a loop to count how common each letter is
names = popularity.Name.values
char_counts = numpy.zeros(26)
for i in range(len(names)):
    char_counts[char_as_int(get_last_letter(str(names[i])))] += 1

print(char_counts)

# plot them
char_a_index = ord('a')
char_z_index = ord('z')
alphabet_range = 26
pyplot.bar(numpy.arange(alphabet_range), char_counts, tick_label=[chr(x) for x in range(char_a_index,char_z_index+1)])
pyplot.show()