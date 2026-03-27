x = [[0,0],[0,1],[1,0],[1,1]]
y = [0,0,0,1]
w = [0,0]
b = 0
lr = 0.1

def step(sum):
    return 1 if sum >= 0 else 0

for i in range(100):
    for x1,y1 in zip(x,y):
        y_pred = step(w[0]*x1[0] + w[1]*x1[1] + b)
        error = y1 - y_pred

        w[0] += lr*error*x1[0]
        w[1] += lr*error*x1[1]

        b +=lr*error

print("Weight: ", w)
print("Bias: ", b)

for x1 in x:
    print(f"{x1}->{step(w[0]*x1[0] + w[1]*x1[1] + b)}")
#
# def OR(A, B):
#     return A | B
#
#
# print("Output of 0 OR 0 is", OR(0, 0))
# print("Output of 0 OR 1 is", OR(0, 1))
# print("Output of 1 OR 0 is", OR(1, 0))
# print("Output of 1 OR 1 is", OR(1, 1))
#
# def AND(A, B):
#     return A & B
#
#
# print("Output of 0 OR 0 is", AND(0, 0))
# print("Output of 0 OR 1 is", AND(0, 1))
# print("Output of 1 OR 0 is", AND(1, 0))
# print("Output of 1 OR 1 is", AND(1, 1))
#
# # Function to simulate XOR Gate
# def XOR(A, B):
#     return A ^ B
#
#
# print("Output of 0 XOR 0 is", XOR(0, 0))
# print("Output of 0 XOR 1 is", XOR(0, 1))
# print("Output of 1 XOR 0 is", XOR(1, 0))
# print("Output of 1 XOR 1 is", XOR(1, 1))