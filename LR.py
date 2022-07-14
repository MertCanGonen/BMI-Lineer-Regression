from typing import List

class LineerRegressionModel:
    
    def __init__(self, learning_rate: float, epoch: int):
        self.learning_rate = learning_rate
        self.epoch = epoch

    m1 = 1
    m2 = 1
    b = 0

    def calculateLoss(self, i1, i2, i3, x_train: List[int], y_train: List[int], z_train: List[int]): #her bir epoch icin loss hesapla
        m1 = i1
        m2 = i2
        b = i3
        length = len(x_train) #lengths of x, y and z train are equals
        lossSum = 0
        for i in range(length):
            lossSum = lossSum + ( (m1 * x_train[i]) + (m2 * y_train[i]) + b - (z_train[i]) )**2
        lossSum = lossSum / length
        return lossSum

    def fit(self, x_train: List[int], y_train: List[int], z_train: List[int], x_test: List[int], y_test: List[int], z_test: List[int]):
        length = len(x_train) #lengths of x, y and z train are equals
        lossSumTraining = 0
        lossSumTesting = 0
        lossTraining = []
        lossTesting = []
        accuracyTraining = []
        accuracyTesting = []
        result = []
        for i in range (1000):
            lossSumTraining = self.calculateLoss(self.m1, self.m2, self.b, x_train, y_train, z_train)
            lossSumTesting = self.calculateLoss(self.m1, self.m2, self.b, x_test, y_test, z_test)

            lossTraining.append(lossSumTraining)
            lossTesting.append(lossSumTesting)

            #calculate m1, m2 and b again
            tmpB = 0
            for i in range(length):
                tmpB = tmpB + self.m1*x_train[i] + self.m2*y_train[i] + self.b - z_train[i]
            tmpB = (2 * tmpB) / length
            self.b = self.b - ( self.learning_rate * tmpB )

            tmpM1 = 0
            for i in range(length):
                tmpM1 = tmpM1 + ( self.m1*x_train[i] + self.m2*y_train[i] + self.b - z_train[i] ) * x_train[i]
            tmpM1 = (2 * tmpM1) / length
            self.m1 = self.m1 - ( self.learning_rate * tmpM1 )

            tmpM2 = 0
            for i in range(length):
                tmpM2 = tmpM2 + ( self.m1*x_train[i] + self.m2*y_train[i] + self.b - z_train[i] ) * y_train[i]
            tmpM2 = (2 * tmpM2) / length
            self.m2 = self.m2 - ( self.learning_rate * tmpM2 )

            resultOfEpochTraining = self.predict(x_train, y_train); #o anki m1, m2 ve b'ye gore training datasi icin predict results dondu.            
            err = 0 #Mean Error
            for i in range (len(y_train)):
                err = err + (z_train[i] - resultOfEpochTraining[i])
            err = err / len(y_train)
            accuracyTraining.append(err)

            resultOfEpochTesting = self.predict(x_test, y_test); #o anki m1, m2 ve b'ye gore test datasi icin predict results dondu.
            err2 = 0 #Mean Error
            for i in range (len(y_test)):
                err2 = err2 + (z_test[i] - resultOfEpochTraining[i])
            err2 = err2 / len(y_test)
            accuracyTesting.append(err2)

        result.append(lossTraining)
        result.append(accuracyTraining)
        result.append(lossTesting)
        result.append(accuracyTesting)
        return result

    def predict(self, x_test: List[int], y_test: List[int]):
        result = []
        length = len(x_test)
        for i in range(length):
            result.append(self.m1*x_test[i] + self.m2*y_test[i] + self.b)
        return result        

    def values(self):
        result = []
        result.append(self.m1)
        result.append(self.m2)
        result.append(self.b)
        return result

