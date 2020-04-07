import numpy as np
import math
import matplotlib.pyplot as plt


# Function to provide an estimate of
# number of gaussian distributions required
def gaussianRequired(image):
    red = image[:, :, 2].ravel()
    green = image[:, :, 1].ravel()
    blue = image[:, :, 0].ravel()
    data = [red, green, blue]

    fig = plt.figure()
    plt1 = fig.add_subplot(221)
    plt2 = fig.add_subplot(222)
    plt3 = fig.add_subplot(223)
    plt1.hist(red, 256, [0, 256])
    plt1.set_title('Histogram of Red channel of the image')
    plt2.hist(green, 256, [0, 256])
    plt2.set_title('Histogram of green channel of the image')
    plt3.hist(blue, 256, [0, 256])
    plt3.set_title('Histogram of blue channel of the image')
    plt.show()

def gaussian(x, u, cov):
    x.reshape(len(x),1)
    u.reshape(len(u),1)
    cov.reshape(len(cov),len(cov[0]))

    detCov = np.linalg.det(cov)
    invCov = np.linalg.inv(cov)
    coeff = 1.0 / math.sqrt(((2 * math.pi) ** 3) * detCov)
    val = coeff * np.exp(-0.5*np.dot(np.dot((x-u).T,cov),(x-u)))
    return val


def EM(data, numClusters=3):
    ########################################################
    #           initialization step
    ########################################################
    _,channel= data.shape 

    cov = np.zeros((numClusters,channel,channel))                               #initializing covariance matrix
    for i in range(numClusters):
        np.fill_diagonal(cov[i],np.random.randint(2,10))  
    

    u = np.random.randint(min(data[:,0]),max(data[:,0]),(numClusters,channel)).astype(float)  #initializing means
    Pc = [1.0 / numClusters] * numClusters                                      # probabilities of clusters

    # Pc_x = np.zeros((len(data), numClusters,channel))                         # Initializing probability of c given x
    Pc_x = np.zeros((len(data), numClusters))                                   # Initializing probability of c given x
    logLikelihood = []

    for iter in range(1000):
        ######################################################
                  # E Step
        ######################################################
        for i in range(len(data)):
            for c in range(numClusters):
                Pc_x[i,c] = Pc[c]*gaussian(data[i,:],u[c],cov[c])


        normalize = np.sum(Pc_x, axis=1)
        Pc_x = Pc_x / normalize[:, np.newaxis]

        #####################################################
                 # M Step
        #####################################################
        m_c = np.sum(Pc_x, axis=0)

        # Probabilities of clusters
        Pc = m_c / np.sum(m_c)

        # New mean
        for i in range(numClusters):
            for j in range(len(data)):
                u[i] += Pc_x[j,i]*data[j,:]
            u[i] = u[i]/m_c

        # Variance
        for j in range(numClusters):
            for i in range(len(data)):
                cov[j] += (Pc_x[i, j] * np.dot((data[i,:] - u[j]), (data[i,:] - u[j]).T))
            cov[j] = cov[j]/m_c

        logLikelihood.append(np.sum(np.log(normalize)))
        if len(logLikelihood) >= 2:
            if abs(logLikelihood[-1] - logLikelihood[-2] < 0.001):
                break

    return u,cov,Pc



if __name__ == '__main__':
    

    mean = [0, 3, 6]
    sigma = [2, 0.5, 3]

    data1 = np.random.normal(0, 2, (50, 1))
    data2 = np.random.normal(3, 0.5, (50, 1))
    data3 = np.random.normal(6, 3, (50, 1))
    data = np.concatenate((data1, data2, data3), axis=0).flatten()
    
    dat = np.column_stack((data,data,data)).reshape(len(data),3)

    EM(dat,3)
    # Actual parameters
    # print("Actual parameters ")
    # for i in range(3):
        # print('Mean(u): {}, Standard Deviation(sigma): {}'.format(mean[i], sigma[i]))

    # print("")
    # print("=====================")
    # print("Model parameters")

    # u, cov, Pc = EM(data, 3)
    # for i in range(3):
        # print('Mean(u):{}, Standard Deviation(sigma): {}'.format(u[i], cov[i]))
    # print(Pc)
