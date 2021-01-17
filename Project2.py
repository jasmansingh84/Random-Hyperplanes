import sys
import math
import random

def dotProduct(w, x):
    dp = 0.0
    for wi, xi in zip(w, x):
        dp += wi * xi
    return dp
    
def sign(x):
    if(x > 0):
        return 1
    elif(x < 0):
        return -1
    return 0
    
def standardize_data(traindata, testdata):
    rows = len(traindata)
    cols = len(traindata[0])  
    for i in range(0, cols, 1):
        result = 0
        for j in range(0, rows, 1):
            result += math.pow(traindata[j][i], 2)
        result = math.sqrt(result)
        if(result != 0):
            for a in range(0, rows, 1):
                traindata[a][i] /= result
            for b in range(0, len(testdata), 1):
                testdata[b][i] /= result
    return [traindata, testdata]


labels=[]
datafile = sys.argv[1]
f=open(datafile)
data=[]
l=f.readline()
while (l != ''):
    a=l.split()
    labels.append(int(a[0]))
    l2=[]
    for j in range(1, len(a), 1):
        l2.append(float(a[j]))
    data.append(l2)
    l=f.readline()
f.close()

labelfile = sys.argv[2]
f=open(labelfile)
testdata= []
l=f.readline()
while(l != ''):
    a=l.split()
    l2=[]
    for j in range(1,len(a),1):
        l2.append(float(a[j]))
    testdata.append(l2)
    l=f.readline()


noRows=len(data)
noCols =len(data[0])

k = int(sys.argv[3])

w = []
for i in range(0, k, 1):
    w.append([])
    for j in range(0, noCols-1, 1):
        w[i].append(random.uniform(-1, 1))
                
z=[]                        
for i in range (0,len(data),1):
    z.append([])	
    for j in range(0, k, 1):
        t=dotProduct(w[j],data[i])
        min = 1000000000
        maximum = 0
        for a in range(0, noRows, 1):
            dp = dotProduct(w[j],data[a])
            if dp>maximum:
                maximum = dp
            if dp<min:
                min = dp
        w0 = random.uniform(maximum,min)        
        #print(i,t)
        z[i].append(sign(t+w0))  
	
        
z1 = []  
for i in range (0,len(testdata),1):
    z1.append([])
    for j in range(0, k, 1):
        t=dotProduct(w[j],testdata[i])
        min = 1000000000
        maximum = 0
        for a in range(0, len(testdata), 1):
            dp = dotProduct(w[j],testdata[a])
            if dp>maximum:
                maximum = dp
            if dp<min:
                min = dp
        w0 = random.uniform(maximum,min)        
        #print(i,t)
        z1[i].append(sign(t+w0))


standardize_data(data,testdata)
standardize_data(z,z1)
for i in range(len(data)):
    data[i].append(1)
for i in range(len(testdata)):
    testdata[i].append(1)
for i in range(len(z)):
    z[i].append(1)
for i in range(len(z1)):
    z1[i].append(1)

eta = 0.001
theta = 0.001

v=[]
for i in range(len(data[0])):
    v.append(random.uniform(-0.01, 0.01))

error=0.0
for i in range (len(data)):
    error += max( 0,1-labels[i]*dotProduct(v,data[i]))

flag = 0
k1=0

while(flag != 1):
    k1+=1
    delf = []
    for i in range(len(data[0])):
        delf.append(0)
    for i in range(len(data)):
        d_p = dotProduct(v, data[i])
        for j in range (len(data[0])):
            if(d_p*labels[i]<1):
                delf[j]+=-1*data[i][j]*labels[i]
            else:
                delf[j]+=0
##update
    for j in range(len(data[0])):
        v[j] = v[j] - eta*delf[j]

##compute error
    curr_error = 0
    for i in range (len(data)):
        curr_error += max( 0,1-labels[i]*dotProduct(v,data[i]))
    #print(error,k1)
    if error - curr_error < 0.001:
        flag = 1
    error = curr_error

#predictions
out=open("original_output.txt","w")
for i in range(0, len(testdata)):
    d_p = dotProduct(v, testdata[i])
    if(d_p > 0):
        out.write("1\n")
    else:
        out.write("-1\n")
        
        
        

#prediction for data
v=[]
for i in range(len(z[0])):
    v.append(random.uniform(-0.01, 0.01))

error=0.0
for i in range (len(data)):
    error += max( 0,1-labels[i]*dotProduct(v,z[i]))


flag = 0
k1=0

while(flag != 1):
    k1+=1
    delf = []
    for i in range(k):
        delf.append(0)
    for i in range(len(z)):
        d_p = dotProduct(v, z[i])
        for j in range (k):
            if(d_p*labels[i]<1):
                delf[j]+=-1*z[i][j]*labels[i]
            else:
                delf[j]+=0
##update
    for j in range(k):
        v[j] = v[j] - eta*delf[j]

##compute error
    curr_error = 0
    for i in range (len(z)):
        curr_error += max( 0,1-labels[i]*dotProduct(v,z[i]))
    #print(error,k1)
    if error - curr_error < 0.001:
        flag = 1
    error = curr_error

#predictions
out=open("01space_output.txt","w")
for i in range(0, len(z1)):
    d_p = dotProduct(v, z1[i])
    if(d_p > 0):
        out.write("1\n")
    else:
        out.write("-1\n")