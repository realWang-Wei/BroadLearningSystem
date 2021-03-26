import numpy as np
import pandas as pd
import datetime
from sklearn import preprocessing  # 用来转化为独热编码
from sklearn.model_selection import train_test_split
from scipy import linalg as LA     # 用来求正交基

# 在第一次求权重时，并未使用岭回归，还是直接求了伪逆，对于小型数据集这种方法足够了

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

	def generator(self, shape, times):
		# times是多少组mapping nodes
		for i in range(times):
			W = 2*np.random.random(size=shape)-1
			if self.isenhance == True:
				W = LA.orth(W)   # 求正交基，只在增强层使用。也就是原始输入X变成mapping nodes的W和mapping nodes变成enhancement nodes的W要正交
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

	def update(self,otherW, otherb):
		# 权重更新
		self.Wlist += otherW
		self.blist += otherb

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
	def __init__(self, map_num=10,enhance_num=10,DESIRED_ACC = 0.99, EPOCH = 10,STEP = 1, map_function='linear',enhance_function='linear',batchsize='auto'):
		self.map_num = map_num    # 多少组mapping nodes
		self.enhance_num = enhance_num  # 多少组engance nodes
		self.batchsize = batchsize
		self.map_function = map_function
		self.enhance_function = enhance_function
		self.DESIRED_ACC = DESIRED_ACC
		self.EPOCH = EPOCH
		self.STEP = STEP

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
		# 求伪逆
		self.pseudoinverse = np.linalg.pinv(inputdata)
		# 新的输入到输出的权重
		print('pseudoinverse shape:', self.pseudoinverse.shape)
		self.W = self.pseudoinverse.dot(label)

		# 查看当前的准确率
		Y = self.predict(data)
		accuracy = self.accuracy(Y,label)
		print("inital setting, number of mapping nodes {0}, number of enhence nodes {1}, accuracy {2}".format(mappingdata.shape[1],enhancedata.shape[1],round(accuracy,5)))
		# 如果准确率达不到要求并且训练次数小于设定次数，重复添加enhance_nodes
		epoch_now = 0
		while accuracy < self.DESIRED_ACC and epoch_now < self.EPOCH:
			Y = self.addingenhance_predict(data, label, self.STEP, self.batchsize)
			accuracy = self.accuracy(Y, label)
			epoch_now += 1
			print("enhencing {3}, number of mapping nodes {0}, number of enhence nodes {1}, accuracy {2}".format(len(self.mapping_generator.Wlist)*self.batchsize,
            																				                     len(self.enhance_generator.Wlist)*self.batchsize,
                                                                                                                 round(accuracy,5),
                                                                                                                 epoch_now))


	def decode(self,Y_onehot):
		Y = []
		for i in range(Y_onehot.shape[0]):
			lis = np.ravel(Y_onehot[i,:]).tolist()
			Y.append(lis.index(max(lis)))
		return np.array(Y)

	def accuracy(self,predictlabel,label):
		#print('predictlabel shape', predictlabel.shape)bbb
		#print('label shape:', label.shape)
		labels = []
		for i in range(len(label)):
			labels.append(np.argmax(label[i]))
		labels = np.ravel(labels).tolist()
		predictlabel = predictlabel.tolist()
		count = 0
		for i in range(len(labels)):
			if labels[i] == predictlabel[i]:
				count += 1
		return (round(count/len(labels),5))

	def predict(self, testdata):
		#print(self.W.shape)
		testdata = self.normalscaler.transform(testdata)
		test_inputdata = self.transform(testdata)
		return self.decode(test_inputdata.dot(self.W))

	def transform(self,data):
		mappingdata = self.mapping_generator.transform(data)
		enhancedata = self.enhance_generator.transform(mappingdata)
		return np.column_stack((mappingdata,enhancedata))

	def addingenhance_nodes(self, data, label, step = 1, batchsize = 'auto'):
		if batchsize == 'auto':
			batchsize = data.shape[1]

		mappingdata = self.mapping_generator.transform(data)
		inputdata = self.transform(data)
		localenhance_generator = node_generator()
		extraenhance_nodes = localenhance_generator.generator_nodes(mappingdata,step,batchsize,self.enhance_function)

		D = self.pseudoinverse.dot(extraenhance_nodes)
		C = extraenhance_nodes - inputdata.dot(D)
		BT = np.linalg.pinv(C) if (C == 0).any() else  np.mat((D.T.dot(D)+np.eye(D.shape[1]))).I.dot(D.T).dot(self.pseudoinverse)

		self.W = np.row_stack((self.W-D.dot(BT).dot(label),BT.dot(label)))
		self.enhance_generator.update(localenhance_generator.Wlist,localenhance_generator.blist)
		self.pseudoinverse =  np.row_stack((self.pseudoinverse - D.dot(BT),BT))

	def addingenhance_predict(self, data, label, step = 1, batchsize = 'auto'):
		self.addingenhance_nodes(data, label, step, batchsize)
		test_inputdata = self.transform(data)
		return self.decode(test_inputdata.dot(self.W))


if __name__ == '__main__':

    # load the data
    train_data = pd.read_csv('D://GitHub/MNIST/data/train.csv')
    test_data = pd.read_csv('D://GitHub/MNIST/data/test.csv')
    samples_data = pd.read_csv('D://GitHub/MNIST/data/sample_submission.csv')

    label = train_data['label'].values
    data = train_data.drop('label', axis=1)
    data = data.values
    print(data.shape, max(label) + 1)

    traindata,valdata,trainlabel,vallabel = train_test_split(data,label,test_size=0.2,random_state = 0)
    print(traindata.shape,trainlabel.shape,valdata.shape,vallabel.shape)


    bls = broadNet(map_num = 10,        # 初始时多少组mapping nodes
                    enhance_num = 10,   # 初始时多少enhancement nodes
                    EPOCH = 10,         # 训练多少轮
                    map_function = 'relu',
                    enhance_function = 'relu',
                    batchsize = 100,    # 每一组的神经元个数
                    DESIRED_ACC = 0.96, # 期望达到的准确率
                    STEP = 5            # 一次增加多少组enhancement nodes
                   )
    starttime = datetime.datetime.now()
    bls.fit(traindata,trainlabel)
    endtime = datetime.datetime.now()
    print('the training time of BLS is {0} seconds'.format((endtime - starttime).total_seconds()))


    predictlabel = bls.predict(valdata)
    print(show_accuracy(predictlabel,vallabel))


    predicts = bls.predict(test_data)

    # save as csv file
    samples = samples_data['ImageId']
    result = {'ImageId':samples,
              'Label': predicts }
    result = pd.DataFrame(result)

    result.to_csv('D://GitHub/MNIST/data/mnist_bls_addenhancenodes.csv', index=False)
