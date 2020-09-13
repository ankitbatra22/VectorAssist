import numpy
v1 = []
v2 = []
dot = 0

one = (input("Enter the first vector: "))
two = (input("Enter the second vector: "))

v1 = one.split(",")
v1 = list(map(int, v1))

v2 = two.split(",")
v2 = list(map(int, v2))
print(type(v2))
print(v2)

for i in range(len(v1)):
    dot += (v1[i] * v2[i])

print("the dot product of the two vectors is:", dot)



