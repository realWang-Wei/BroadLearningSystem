import numpy as np
from sklearn import preprocessing
import pandas as pd
from sklearn.model_selection import train_test_split
import datetime

def show_accuracy(predictLabel,Label):
    Label = np.ravel(Label).tolist()
    predictLabel = predictLabel.tolist()
    count = 0
    for i in range(len(Label)):
        if Label[i] == predictLabel[i]:
            count += 1
    return (round(count/len(Label),5))

class node_generator(object):
	def __init__(self, isenhance = False):
		self.Wlist = []
		self.blist = []
		self.function_num = 0
		self.isenhance = isenhance

	def sigmoid(self, x):
		return 1.0/(1 + np.exp(-x))

	def relu(self, x):
		return np.maximum(x, 0)

	def tanh(self, x):
		return (np.exp(x) - np.exp(-x))/(np.exp(x) + np.exp(-x))

	def linear(self, x):
		return x

	def orth(self, W):
		"""
		目前看来，这个函数应该配合下一个generator函数是生成权重的
		"""
		for i in range(0, W.shape[1]):
			w = np.mat(W[:,i].copy()).T
			w_sum = 0
			for j in range(i):
				wj = np.mat(W[:,j].copy()).T
				w_sum += (w.T.dot(wj))[0,0]*wj
			w -= w_sum
			w = w/np.sqrt(w.T.dot(w))
			W[:,i] = np.ravel(w)

		return W

	def generator(self, shape, times):
		for i in range(times):
			W = 2*np.random.random(size=shape)-1
			if self.isenhance == True:
				W = self.orth(W)   # 只在增强层使用
			b = 2*np.random.random() -1
			yield (W, b)

	def generator_nodes(self, data, times, batchsize, function_num):
		# 按照bls的理论，mapping layer是输入乘以不同的权重加上不同的偏差之后得到的
		# 若干组，所以，权重是一个列表，每一个元素可作为权重与输入相乘
		self.Wlist = [elem[0] for elem in self.generator((data.shape[1], batchsize), times)]
		self.blist = [elem[1] for elem in self.generator((data.shape[1], batchsize), times)]

		self.function_num = {'linear':self.linear,
						'sigmoid': self.sigmoid,
						'tanh':self.tanh,
						'relu':self.relu }[function_num]  # 激活函数供不同的层选择
		# 下面就是先得到一组mapping nodes，再不断叠加，得到len(Wlist)组mapping nodes
		nodes = self.function_num(data.dot(self.Wlist[0]) + self.blist[0])
		for i in range(1, len(self.Wlist)):
			nodes = np.column_stack((nodes, self.function_num(data.dot(self.Wlist[i])+self.blist[i])))
		return nodes	

	def transform(self,testdata):
		testnodes = self.function_num(testdata.dot(self.Wlist[0])+self.blist[0])
		for i in range(1,len(self.Wlist)):
			testnodes = np.column_stack((testnodes, self.function_num(testdata.dot(self.Wlist[i])+self.blist[i])))
		return testnodes

class scaler:
    def __init__(self):
        self._mean = 0
        self._std = 0
    
    def fit_transform(self,traindata):
        self._mean = traindata.mean(axis = 0)
        self._std = traindata.std(axis = 0)
        return (traindata-self._mean)/(self._std+0.001)
    
    def transform(self,testdata):
        return (testdata-self._mean)/(self._std+0.001)

class broadNet(object):
	def __init__(self, map_num=10,enhance_num=10,map_function='linear',enhance_function='linear',batchsize='auto'):
		self.map_num = map_num
		self.enhance_num = enhance_num
		self.batchsize = batchsize
		self.map_function = map_function
		self.enhance_function = enhance_function

		self.W = 0
		self.pseudoinverse = 0
		self.normalscaler = scaler()
		self.onehotencoder = preprocessing.OneHotEncoder(sparse = False)
		self.mapping_generator = node_generator()
		self.enhance_generator = node_generator(isenhance = True)

	def fit(self, data, label):
		if self.batchsize == 'auto':
			self.batchsize = data.shape[1]

		data = self.normalscaler.fit_transform(data)
		label = self.onehotencoder.fit_transform(np.mat(label).T)

		mappingdata = self.mapping_generator.generator_nodes(data, self.map_num, self.batchsize,self.map_function)
		enhancedata = self.enhance_generator.generator_nodes(mappingdata, self.enhance_num, self.batchsize,self.enhance_function)

		print('number of mapping nodes {0}, number of enhence nodes {1}'.format(mappingdata.shape[1],enhancedata.shape[1]))
		print('mapping nodes maxvalue {0} minvalue {1} '.format(round(np.max(mappingdata),5),round(np.min(mappingdata),5)))
		print('enhence nodes maxvalue {0} minvalue {1} '.format(round(np.max(enhancedata),5),round(np.min(enhancedata),5)))

		inputdata = np.column_stack((mappingdata, enhancedata))
		print('input shape ', inputdata.shape)
		pseudoinverse = np.linalg.pinv(inputdata)
		# 新的输入到输出的权重
		print('pseudoinverse shape:', pseudoinverse.shape)
		self.W = pseudoinverse.dot(label)

	def decode(self,Y_onehot):
		Y = []
		for i in range(Y_onehot.shape[0]):
			lis = np.ravel(Y_onehot[i,:]).tolist()
			Y.append(lis.index(max(lis)))
		return np.array(Y)

	def accuracy(self,predictlabel,label):
		label = np.ravel(label).tolist()
		predictlabel = predictlabel.tolist()
		count = 0
		for i in range(len(label)):
			if label[i] == predictlabel[i]:
				count += 1
		return (round(count/len(label),5))

	def predict(self, testdata):
		testdata = self.normalscaler.transform(testdata)
		test_mappingdata = self.mapping_generator.transform(testdata)
		test_enhancedata = self.enhance_generator.transform(test_mappingdata)

		test_inputdata = np.column_stack((test_mappingdata,test_enhancedata))    
		return self.decode(test_inputdata.dot(self.W))   

if __name__ == '__main__':
    data = pd.read_csv('./balance-scale.csv')  
  
    le = preprocessing.LabelEncoder()
    for item in data.columns:
        data[item] = le.fit_transform(data[item])


    label = data['Class'].values
    data = data.drop('Class',axis=1)
    data = data.values
    print(data.shape,max(label)+1)

    traindata,testdata,trainlabel,testlabel = train_test_split(data,label,test_size=0.2,random_state = 0)
    print(traindata.shape,trainlabel.shape,testdata.shape,testlabel.shape)


    bls = broadNet(map_num = 10, 
               enhance_num = 10,
               map_function = 'relu',
               enhance_function = 'relu',
               batchsize = 100)

    starttime = datetime.datetime.now()
    bls.fit(traindata,trainlabel)
    endtime = datetime.datetime.now()
    print('the training time of BLS is {0} seconds'.format((endtime - starttime).total_seconds()))

    predictlabel = bls.predict(testdata)
    print(show_accuracy(predictlabel,testlabel))
